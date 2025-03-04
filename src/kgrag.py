from dataclasses import dataclass,field
from utils import read_config, compute_mdhash_id
from llm import silcon_compelete, siliconcloud_embedding
from kg import Neo4JStorage
from vector_storage import NanoVectorDBStorage
from prompt import prompts
import asyncio
import json
@dataclass
class KGrag:
    # config = read_config()
    llm: callable = silcon_compelete
    embedding: callable = silcon_compelete
    kv_storage = None
    corpus: list[str] = field(default_factory=list)
    # graph_storage = Neo4JStorage
    # vector_storage = NanoVectorDBStorage
    
    
    
    def generate_triple(self, file_paths):
        self.corpus = file_paths
        for path in self.corpus:
            with open(path, "r") as file:
                doc = json.loads(file.read())
                for d in doc:
                    title = d['title']
                    content = d['content']
                    triples = self.llm(content)
                    id = compute_mdhash_id(title,prefix='doc')
                    
                    pass
                
    async def entity_extraction(self, text):
        entities = await self.llm(prompt=text,system_prompt=prompts['entity_extraction'])
        return entities
                
async def main():
    rag = KGrag()
    entities = rag.generate_triple(['data/2wiki_corpus.json'])
    pass

if __name__ == "__main__":
    asyncio.run(main())