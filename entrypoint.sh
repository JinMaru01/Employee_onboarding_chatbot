#!/bin/bash
exec gunicorn --config gunicorn_config.py flask_app:app