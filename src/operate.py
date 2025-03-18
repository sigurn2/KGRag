from typing import TypedDict

TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {
        "title": str,
        "content": str,
        "full_doc_id": str,
    },
)

async def extract_entities(
    chucks: dict[str, TextChunkSchema]
)