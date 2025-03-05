from dataclasses import dataclass,field
from utils import logger,read_config, compute_mdhash_id
from llm import silcon_compelete, siliconcloud_embedding
from kg import Neo4JStorage
from vector_storage import NanoVectorDBStorage
from prompt import prompts
import asyncio
from base import StorageNameSpace
from typing import cast
import json
from storage import JsonKVStorage
from datetime import datetime
from tqdm.asyncio import tqdm as tqdm_async
import os
from operate import chunking_by_passage

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
    working_dir: str = field(default_factory=lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
    llm: callable = silcon_compelete
    embedding: callable = silcon_compelete
    kv_storage = JsonKVStorage
    
    corpus: list[str] = field(default_factory=list)
    # graph_storage = Neo4JStorage
    # vector_storage = NanoVectorDBStorage
     
    
    
    
    def __post_init__(self):
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
        self.storage_config = {
            'working_dir':  self.working_dir
        }
        self.doc_cache = JsonKVStorage('doc', self.storage_config)
        self.chunk_cache = JsonKVStorage('chunk', self.storage_config)
    
    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))
    async def ainsert(self, string_or_strings):
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            
            new_docs = {
                compute_mdhash_id(c.strip(), prefix='doc-'): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.doc_cache.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning('all file have been processed')
                return
            
            logger.info(f'[new docs] inserting {len(new_docs)} docs')
            
            inserting_chunks = {}
            for doc_key, doc in tqdm_async(
                new_docs.items(), desc='Chunking documents', unit='doc'
            ):
                chunks = {
                    compute_mdhash_id(dp['content'], prefix='chunk-'):
                        {
                            **dp,
                            'full_doc_id': doc_key,
                        }
                    for dp in doc['content']
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.chunk_cache.filter_keys(list(inserting_chunks.keys()))
            inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}
            if not len(inserting_chunks):
                logger.warning('all file have been processed')
                return
            logger.info(f'[new chunks] inserting {len(inserting_chunks)} chunks')
            await self.chunk_cache.upsert(inserting_chunks)
            await self.doc_cache.upsert(new_docs)
        finally:
            await self._insert_done()
            
                
    async def entity_extraction(self, text):
        entities = await self.llm(prompt=text,system_prompt=prompts['entity_extraction'])
        return entities
    
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

    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    rag = KGrag(working_dir=WORKING_DIR)
    with open('/mnt/home/liangdongqi/KGRag/data/misque_corpus.json') as f:
        rag.insert(f.read())