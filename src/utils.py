import yaml
import logging
import re
from hashlib import md5

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


def safe_unicode_decode(content):
    # Regular expression to find all Unicode escape sequences of the form \uXXXX
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    # Function to replace the Unicode escape with the actual character
    def replace_unicode_escape(match):
        # Convert the matched hexadecimal value into the actual Unicode character
        return chr(int(match.group(1), 16))

    # Perform the substitution
    decoded_content = unicode_escape_pattern.sub(
        replace_unicode_escape, content.decode("utf-8")
    )

    return decoded_content

def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()
