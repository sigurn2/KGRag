import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
import pandas as pd
from kg import Neo4JStorage
from llm import silcon_compelete, siliconcloud_embedding
from storage import JsonKVStorage, NanoVectorDBStorage
from utils import logger, compute_mdhash_id

script_path = Path(__file__).resolve()
project_path = script_path.parent.parent


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class KGrag:
    # config = read_config()
    working_dir: str = field(
        default_factory=lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    llm: callable = silcon_compelete
    embedding: callable = siliconcloud_embedding
    kv_storage = JsonKVStorage
    # corpus: json = {}
    graph_storage = Neo4JStorage
    vector_storage = NanoVectorDBStorage
    corpus: Literal["2wiki", "hotpotqa", "musique"] = "2wiki"

    def __post_init__(self):
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
        self.storage_config = {"working_dir": self.working_dir}
        self.doc_cache = JsonKVStorage("doc", self.storage_config)
        self.chunk_cache = JsonKVStorage("chunk", self.storage_config)
        self.dataset = None
        if self.corpus == "2wiki":
            self.dataset = pd.read_json(
                project_path / "data" / "2wiki_corpus.json"
            )
        if self.corpus == "hotpotqa":
            self.dataset = pd.read_json(
                project_path / "data" / "hotpotqa_corpus.json"
            )
        if self.corpus == "musique":
            self.dataset = pd.read_json(
                project_path / "data" / "misque_corpus.json"
            )
        self.insert()

    def insert(self):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self._ainsert())

    async def _ainsert(self):
        try:
            if self.dataset is None:
                logger.error("corpus is None")
                return

            new_chunks = {
                compute_mdhash_id(row["title"], prefix="chunk-"): {
                    **row,
                    "full_doc_id": id,
                }
                for id, row in self.dataset.iterrows()
            }
            _add_chunk_keys = await self.chunk_cache.filter_keys(
                list(new_chunks.keys())
            )
            if not len(_add_chunk_keys):
                logger.warning("all file have been processed")
                return
            inserting_chunks = {
                k: v for k, v in new_chunks.items() if k in _add_chunk_keys
            }
            await self.chunk_cache.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.doc_cache,
            self.chunk_cache,
        ]:
            if storage_inst is None:
                continue
            tasks.append(storage_inst.index_done_callback())
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    WORKING_DIR = "test"
    corpus = "musique"
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    rag = KGrag(corpus=corpus, working_dir=WORKING_DIR)
