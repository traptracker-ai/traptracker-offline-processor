"""Production entry point. Served by waitress in the container."""
import os
from app import app

if __name__ == '__main__':
    from waitress import serve
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '8501'))
    threads = int(os.environ.get('WAITRESS_THREADS', '8'))
    serve(app, host=host, port=port, threads=threads)
