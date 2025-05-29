import time
import threading
import json
import requests
import websocket
import socket
import os
import re
from datetime import datetime
from lib.config import MONITORING_WEBSOCKET_URL, MONITORING_INTERVAL_SECONDS


class MonitoringService:
    def __init__(self, app, monitoring_endpoint_url="http://localhost:8888/monitoring"):
        self.app = app
        self.monitoring_endpoint_url = monitoring_endpoint_url
        self.websocket_url = MONITORING_WEBSOCKET_URL
        self.interval_seconds = MONITORING_INTERVAL_SECONDS
        self.running = False
        self.thread = None
        self.ws = None
        self.node_name = self._infer_node_name()
        # Track the IDs of requests we've already sent to avoid resending
        self.sent_request_ids = set()
        
    def start(self):
        """Start the monitoring service if websocket URL is configured"""
        if not self.websocket_url:
            print("💡 Monitoring websocket URL not configured, monitoring service disabled")
            return
            
        print(f"🔍 Starting monitoring service (interval: {self.interval_seconds}s, websocket: {self.websocket_url})")
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the monitoring service"""
        print("🛑 Stopping monitoring service...")
        self.running = False
        if self.ws:
            self.ws.close()
        if self.thread:
            self.thread.join(timeout=5)
            
    def _infer_node_name(self):
        """Infer the node name from the systemd service file"""
        systemd_file_path = "/etc/systemd/system/uomi.service"
        
        try:
            if not os.path.exists(systemd_file_path):
                print(f"💡 Systemd service file not found at {systemd_file_path}, using empty node name")
                return ""
                
            with open(systemd_file_path, 'r') as file:
                content = file.read()
                
            # Look for --name parameter in the ExecStart line
            # Pattern matches --name "node-name" or --name node-name
            name_pattern = r'--name\s+["\']?([^"\'\s\\]+)["\']?'
            match = re.search(name_pattern, content)
            
            if match:
                node_name = match.group(1)
                print(f"🏷️ Inferred node name from systemd file: {node_name}")
                return node_name
            else:
                print(f"💡 Could not find --name parameter in {systemd_file_path}, using empty node name")
                return ""
                
        except Exception as e:
            print(f"❌ Failed to read systemd service file: {e}")
            return ""
            
    def _connect_websocket(self):
        """Connect to the websocket server"""
        try:
            # Set a larger buffer size to handle the increased data volume
            self.ws = websocket.WebSocket()
            self.ws.connect(
                self.websocket_url,
                sockopt=((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),),
            )
            print(f"✅ Connected to monitoring websocket: {self.websocket_url}")
            
            # Reset the sent request tracking on new connections
            # This ensures we don't miss any data if the connection was broken
            self.sent_request_ids = set()
            print("🔄 Reset request tracking on new connection")
            
            return True
        except Exception as e:
            print(f"❌ Failed to connect to websocket: {e}")
            self.ws = None
            return False
            
    def _send_monitoring_data(self, data):
        """Send monitoring data through websocket"""
        if not self.ws:
            if not self._connect_websocket():
                return False
                
        try:
            # Add node name to the data
            enhanced_data = data.copy() if data else {}
            enhanced_data['node_name'] = self.node_name
            
            # Extract and filter request data to only send new requests
            if 'requests' in enhanced_data and 'all_requests' in enhanced_data['requests']:
                all_requests = enhanced_data['requests']['all_requests']
                new_requests = []
                
                for req in all_requests:
                    # Ensure each request has an ID
                    if 'id' not in req:
                        timestamp = req.get('timestamp', datetime.now().isoformat())
                        input_hash = hash(req.get('input', '')) & 0xffffffff
                        req['id'] = f"{timestamp}_{input_hash}"
                    
                    # Only include requests we haven't sent before
                    if req['id'] not in self.sent_request_ids:
                        new_requests.append(req)
                        self.sent_request_ids.add(req['id'])
                
                # Replace all_requests with only the new ones
                enhanced_data['requests']['new_requests'] = new_requests
                
                # Keep track of how many total requests exist
                enhanced_data['requests']['total_request_count'] = len(all_requests)
                
                # Remove the full all_requests array to reduce payload size
                del enhanced_data['requests']['all_requests']
                
                # Log the delta information
                new_req_count = len(new_requests)
                if new_req_count > 0:
                    print(f"📊 Sending {new_req_count} new requests (total tracked: {len(self.sent_request_ids)})")
            
            message = {
                "type": "monitoring",
                "timestamp": datetime.now().isoformat(),
                "data": enhanced_data
            }
            
            # Convert to JSON string
            message_json = json.dumps(message)
            
            # Check if message is too large (>1MB) and potentially split
            if len(message_json.encode('utf-8')) > 1000000:  # 1MB threshold
                print(f"⚠️ Large monitoring payload detected ({len(message_json.encode('utf-8'))/1000000:.2f}MB), may impact performance")
            
            # Send the data
            self.ws.send(message_json)
            return True
        except Exception as e:
            print(f"❌ Failed to send monitoring data: {e}")
            self.ws = None
            return False
            
    def _get_monitoring_data(self):
        """Fetch monitoring data from the local endpoint"""
        try:
            response = requests.get(self.monitoring_endpoint_url, timeout=15)  # Increased timeout for larger data
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Monitoring endpoint returned status {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Failed to fetch monitoring data: {e}")
            return None
            
    def _monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread"""
        consecutive_failures = 0
        max_failures = 5
        retry_delay_seconds = 120  # 2 minutes between retry attempts
        
        while self.running:
            try:
                # Get monitoring data
                monitoring_data = self._get_monitoring_data()
                
                if monitoring_data:
                    # Send data through websocket
                    if self._send_monitoring_data(monitoring_data):
                        consecutive_failures = 0
                        node_info = f" (node: {self.node_name})" if self.node_name else ""
                        uptime_info = f"uptime: {monitoring_data.get('uptime', {}).get('total_seconds', 0):.0f}s"
                        # The new_requests count is now handled in _send_monitoring_data
                        print(f"📊 Sent monitoring data{node_info} ({uptime_info})")
                    else:
                        consecutive_failures += 1
                else:
                    consecutive_failures += 1
                    
                # Check if we should implement backoff due to too many failures
                if consecutive_failures >= max_failures:
                    print(f"❌ Too many consecutive failures ({max_failures}), backing off for {retry_delay_seconds} seconds")
                    # Wait for the retry delay before attempting again
                    for _ in range(retry_delay_seconds // self.interval_seconds):
                        if not self.running:
                            break
                        time.sleep(self.interval_seconds)
                    
                    # Reset failure counter to give it another chance
                    consecutive_failures = max_failures // 2
                    print(f"🔄 Retrying connection after backoff period")
                    continue
                    
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                consecutive_failures += 1
                
            # Wait for the next iteration
            time.sleep(self.interval_seconds)
            
        print("🛑 Monitoring service stopped")
