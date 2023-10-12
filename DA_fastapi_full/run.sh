#!/bin/sh
exec gunicorn --timeout 600 --worker-class uvicorn.workers.UvicornWorker run:app -b 0.0.0.0:8080 --worker=3 --log-file=-