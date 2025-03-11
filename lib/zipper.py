import zlib
import base64

def zip_string(input_string):
    """
    Compresses a string using zlib and encodes it with base64 for safe storage/transmission.
    
    Args:
        input_string (str): The string to compress
        
    Returns:
        str: The compressed string in base64 encoding
    """
    # Convert string to bytes
    bytes_data = input_string.encode('utf-8')
    
    # Compress the bytes
    compressed_data = zlib.compress(bytes_data)
    
    # Convert to base64 for safe storage/transmission
    base64_compressed = base64.b64encode(compressed_data)
    
    # Return as string
    return base64_compressed.decode('ascii')

def unzip_string(compressed_string):
    """
    Decompresses a string that was compressed with zip_string.
    
    Args:
        compressed_string (str): The compressed string in base64 encoding
        
    Returns:
        str: The original uncompressed string
    """
    # Convert from base64 string to bytes
    base64_bytes = compressed_string.encode('ascii')
    
    # Decode base64
    compressed_data = base64.b64decode(base64_bytes)
    
    # Decompress
    original_data = zlib.decompress(compressed_data)
    
    # Convert bytes back to string
    return original_data.decode('utf-8')