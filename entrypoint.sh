#!/bin/bash
set -e  # exit on any error
exec gunicorn --config gunicorn_config.py flask_app:app