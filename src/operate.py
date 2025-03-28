import asyncio
import json
from typing import TypedDict
from base import BaseVectorStorage, BaseKVStorage
from prompt import prompts
from pathlib import Path
import pandas as pd
from llm import silcon_compelete
from tqdm.asyncio import tqdm_asyncio as tqdm_async

script_path = Path(__file__).resolve()
project_path = script_path.parent.parent

TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {
        "title": str,
        "content": str,
        "full_doc_id": str,
    },
)


async def extract_entities(
        chucks: dict[str, TextChunkSchema],
        knowledge_graph_inst,
        chunk_entity_jdb: BaseKVStorage,
        entity_vdb: BaseVectorStorage,
        relation_vdb: BaseVectorStorage,
        llm_func: callable,
):
    ordered_chunks = list(chucks.items())
    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp['content']

        entities = await llm_func(
            prompt=content,
            system_prompt=prompts["keywords_extraction"],
        )

        triples = await llm_func(
            prompt=entities,
            system_prompt=prompts["triples_extraction"],
        )
        pass

    results = []
    for result in tqdm_async(
            asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
            total=len(ordered_chunks),
            desc="Extracting entities from chunks",
            unit="chunk"
    ):
        results.append(await result)


async def main():
    data_folder = project_path / 'src' / 'test' / 'kv_store_chunk.json'
    df = pd.read_json(data_folder)
    chunks = {}

    for idx, row in df.iterrows():
        chunk_key = str(idx)  # You can customize how you generate chunk keys
        chunks[chunk_key] = {
            "title": row["title"],
            "content": row["content"],
            "full_doc_id": row["full_doc_id"],
        }

    await extract_entities(chunks, None, None, None, None, silcon_compelete)


if __name__ == '__main__':
    asyncio.run(main())
