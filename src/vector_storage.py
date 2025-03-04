from dataclasses import dataclass
from nano_vectordb import NanoVectorDB
from utils import logger
from tqdm.asyncio import tqdm as tqdm_async
import asyncio
import numpy as np
import os
from utils import compute_mdhash_id

@dataclass
class NanoVectorDBStorage:
    cosine_better_than_threshold: float = 0.2
    
