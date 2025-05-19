import os
import multiprocessing

bind = "0.0.0.0:8443"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
keyfile = os.path.join(os.path.dirname(__file__), 'key.pem')
certfile = os.path.join(os.path.dirname(__file__), 'cert.pem')
# Remove the ssl_version line as it may be causing issues
#ssl_version = 'TLSv1_2'
secure_scheme_headers = {'X-Forwarded-Proto': 'https'}