from dataclasses import dataclass
import os
from utils import write_json, load_json, logger,compute_mdhash_id
from base import BaseKVStorage, BaseVectorStorage
from nano_vectordb import NanoVectorDB
import numpy as np
import asyncio
from tqdm.asyncio import tqdm_asyncio as tqdm_async


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config['working_dir']
        self._file_name = os.path.join(working_dir, f'kv_store_{self.namespace}.json')
        self._data = load_json(self._file_name) or {}
        logger.info(f'load kv {self.namespace}')

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_ids(self, idx, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in idx]
        return [
            (
                {
                    k: v for k, v in self._data[id].items() if k in fields
                }
                if self._data.get(id, None)
                else None
            )
            for id in idx
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i: i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        if len(embeddings) == len(list_data):
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
            results = self._client.upsert(datas=list_data)
            return results
        else:
            # sometimes the embedding is not returned correctly. just log it.
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")

    async def delete_entity(self, entity_name: str):
        try:
            entity_id = [compute_mdhash_id(entity_name, prefix="ent-")]

            if self._client.get(entity_id):
                self._client.delete(entity_id)
                logger.info(f"Entity {entity_name} have been deleted.")
            else:
                logger.info(f"No entity found with name {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def delete_relation(self, entity_name: str):
        try:
            relations = [
                dp
                for dp in self.client_storage["data"]
                if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
            ]
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                self._client.delete(ids_to_delete)
                logger.info(
                    f"All relations related to entity {entity_name} have been deleted."
                )
            else:
                logger.info(f"No relations found for entity {entity_name}.")
        except Exception as e:
            logger.error(
                f"Error while deleting relations for entity {entity_name}: {e}"
            )

    async def index_done_callback(self):
        self._client.save()
