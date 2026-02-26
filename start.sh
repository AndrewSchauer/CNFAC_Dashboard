#!/bin/bash
exec gunicorn CMAH_dash:server --bind "0.0.0.0:${PORT:-10000}" --timeout 120 --workers 1
