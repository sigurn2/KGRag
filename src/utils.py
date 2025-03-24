import yaml
import logging
import re
from hashlib import md5
import json
import os
from dataclasses import dataclass
import asyncio
from pathlib import Path
import numpy as np

logger = logging.getLogger('kgrag')

script_path = Path(__file__).resolve()
project_path = script_path.parent.parent

file_path = project_path / "config.yaml"


class UnlimitedSemaphore:
    """A context manager that allows unlimited access."""

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable
    concurrent_limit: int = 16

    def __post_init__(self):
        if self.concurrent_limit != 0:
            self._semaphore = asyncio.Semaphore(self.concurrent_limit)
        else:
            self._semaphore = UnlimitedSemaphore()

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        async with self._semaphore:
            return await self.func(*args, **kwargs)


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


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)
