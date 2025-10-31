import os

# Memory-friendly defaults for small instances (Render Free ~512MB)

# Bind is provided by CLI (-b 0.0.0.0:$PORT), but keep a fallback
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# Keep 1 worker to minimize RAM; gthread gives simple concurrency
workers = 1
worker_class = "gthread"
threads = 2

# Allow longer requests (first-time telemetry downloads)
timeout = 180
keepalive = 5

# Recycle workers periodically to mitigate leaks in native libs
max_requests = 80
max_requests_jitter = 20

# Avoid preloading heavy modules into the master process
preload_app = False

# Use tmpfs when available; avoids excessive disk usage in /tmp
worker_tmp_dir = "/dev/shm"

loglevel = "info"


