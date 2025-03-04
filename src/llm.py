import os
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    AsyncAzureOpenAI,
)
from utils import logger, read_config, safe_unicode_decode
from functools import lru_cache
import numpy as np
import struct
import aioboto3
import aiohttp
import base64




@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    **kwargs,
) -> str:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    open_async_client = (
        AsyncOpenAI if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # 添加日志输出
    logger.debug("===== Query Input to LLM =====")
    logger.debug(f"Query: {prompt}")
    logger.debug(f"System prompt: {system_prompt}")
    logger.debug("Full context:")

    if "response_format" in kwargs:
        response = await open_async_client.beta.chat.completions.parse(
            model=model, messages=messages, **kwargs
        )
    else:
        response = await open_async_client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

    # if hasattr(response, "__aiter__"):
    #
    #     async def inner():
    #         async for chunk in response:
    #             content = chunk.choices[0].delta.content
    #             if content is None:
    #                 continue
    #             if r"\u" in content:
    #                 content = safe_unicode_decode(content.encode("utf-8"))
    #             yield content
    #
    #     return inner()
    # else:
    content = response.choices[0].message.content
    if r"\u" in content:
        content = safe_unicode_decode(content.encode("utf-8"))
    return content


async def silcon_compelete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await openai_complete_if_cache(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )
    return result

# @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def siliconcloud_embedding(
    texts: list[str],
    model: str = "BAAI/bge-large-zh-v1.5",
    base_url: str = "https://api.siliconflow.cn/v1/embeddings",
    max_token_size: int = 512,
    api_key: str = None,
) -> np.ndarray:
    if api_key and not api_key.startswith("Bearer "):
        api_key = "Bearer " + api_key

    headers = {"Authorization": api_key, "Content-Type": "application/json"}

    truncate_texts = [text[0:max_token_size] for text in texts]

    payload = {"model": model, "input": truncate_texts, "encoding_format": "base64"}

    base64_strings = []
    async with aiohttp.ClientSession() as session:
        async with session.post(base_url, headers=headers, json=payload) as response:
            content = await response.json()
            if "code" in content:
                raise ValueError(content)
            base64_strings = [item["embedding"] for item in content["data"]]

    embeddings = []
    for string in base64_strings:
        decode_bytes = base64.b64decode(string)
        n = len(decode_bytes) // 4
        float_array = struct.unpack("<" + "f" * n, decode_bytes)
        embeddings.append(float_array)
    return np.array(embeddings)

async def main():
    config = read_config()
    open_ai_config = config.get("openai")
    api_key = open_ai_config.get("api_key")
    base_url = open_ai_config.get("base_url")
    texts = ["这是一个测试文本"]
    result = await siliconcloud_embedding(texts,api_key=api_key)
    print(result)


if __name__ == "__main__":
    asyncio.run((main()))
