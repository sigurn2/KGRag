import yaml
import logging


logger = logging.getLogger('kgrag')
file_path = "../config.yaml"


def read_config():
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print("file not found")
    except yaml.YAMLError as e:
        print(f"error occurs while reading {e}")
