from dataclasses import dataclass
import os
from utils import write_json,load_json,logger
from base import BaseKVStorage
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
            return [self._data.get(id,None) for id in idx]
        return [
            (
                {
                    k: v for k, v in self._data[id].items() if k in fields
                }
                if self._data.get(id,None)
                else None
            )
            for id in idx
        ]
    
    async def filter_keys(self, data:list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])
    
    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}