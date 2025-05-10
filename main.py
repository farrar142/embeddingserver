import asyncio
import pickle
import os
from typing import Callable, Coroutine, Generic, ParamSpec, TypeVar
from asgiref.wsgi import WsgiToAsgi
from flask import Flask, request, jsonify
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from redis.asyncio import from_url

model_name = os.getenv("MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

embedding = HuggingFaceEmbeddings(
    model_name=model_name,
)

app = Flask(__name__)

Vector = list[float]

T = TypeVar("T")
P = ParamSpec("P")


class Result(Generic[T]):
    def __init__(self, value: T, cached: bool):
        self.value, self.cached = value, cached


class VectorRedis(Generic[T]):

    def __init__(self, redis_url: str):
        self.redis = from_url(redis_url)

    async def get(self, key: str) -> T | None:
        pickled_value = await self.redis.get(key)
        if pickled_value is not None:
            return pickle.loads(pickled_value)
        return None

    async def set(self, key: str, value: T, expire: int = 3600):
        pickled_value = pickle.dumps(value)
        await self.redis.set(key, pickled_value, ex=expire)

    def cache(self, func: Callable[P, Coroutine[None, None, T]]) -> "Cache[P,T]":

        return Cache(self, func)


class Cache(Generic[P, T]):
    def __init__(
        self, redis: VectorRedis[T], func: Callable[P, Coroutine[None, None, T]]
    ):
        self.func = func
        self.redis = redis

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Result[T]:
        key = f"{self.func.__name__}:{args}:{kwargs}"
        if cached := await self.redis.get(key):
            return Result(cached, True)
        result = await self.func(*args, **kwargs)
        await self.redis.set(key, result)
        return Result(result, False)

    async def purge(self, *args: P.args, **kwargs: P.kwargs) -> None:
        key = f"{self.func.__name__}:{args}:{kwargs}"
        await self.redis.redis.delete(key)


redis = VectorRedis[Vector]("redis://redis:6379")


@redis.cache
async def aembed_documents(document: str) -> Vector:
    return (await embedding.aembed_documents([document]))[0]


@app.post("/embedding")
async def get_embeddings():
    json = request.get_json()
    if not json or "content" not in json:
        return jsonify({"error": "Invalid request"}), 400
    content = request.get_json()["content"]
    if not isinstance(content, str):
        return jsonify({"error": "Content must be a string"}), 400
    results = await aembed_documents(content)
    return jsonify(results.value)


async def test():
    await aembed_documents.purge("t1")
    t1 = await aembed_documents("t1")
    assert t1.cached == False
    t1 = await aembed_documents("t1")
    assert t1.cached == True
    await aembed_documents.purge("t1")
    t1 = await aembed_documents("t1")
    assert t1.cached == False
    await aembed_documents.purge("t1")


if __name__ == "__main__":
    asyncio.run(test())
    import uvicorn

    asgi_app = WsgiToAsgi(app)
    uvicorn.run(asgi_app, host="0.0.0.0", port=8000)
