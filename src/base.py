from dataclasses import dataclass,field
from typing import TypedDict, Union, Literal, Generic, TypeVar



@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict
    
    
T = TypeVar("T")
    
@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    # embedding_func: EmbeddingFunc

    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError

