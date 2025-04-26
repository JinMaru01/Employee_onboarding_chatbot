import yaml
import os

def load_config(config_path="_config/settings.yml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
