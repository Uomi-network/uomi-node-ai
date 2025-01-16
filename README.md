# UOMI NODE AI

Install conda environment and dependencies for AI service by running commands defined in **install.sh script**.

Setup and start uomi-ai service:

```bash
cp uomi-ai.service /etc/systemd/system/ # edit the file to set the correct path to the uomi-ai service
systemctl enable uomi-ai
systemctl start uomi-ai
```

View uomi-ai service logs in real-time:

```bash
journalctl -f -u uomi-ai
```