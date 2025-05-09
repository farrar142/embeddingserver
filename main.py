import asyncio
import pickle
import os
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


class VectorRedis:
    def __init__(self, redis_url: str):
        self.redis = from_url(redis_url)

    async def get(self, key: str) -> Vector | None:
        pickled_value = await self.redis.get(key)
        if pickled_value is not None:
            return pickle.loads(pickled_value)
        return None

    async def set(self, key: str, value: Vector, expire: int = 3600):
        pickled_value = pickle.dumps(value)
        await self.redis.set(key, pickled_value, ex=expire)


redis = VectorRedis("redis://redis:6379")


async def aembed_documents(document: str) -> Vector:
    if cached := await redis.get(document):
        return cached
    embedding_vector = (await embedding.aembed_documents([document]))[0]
    asyncio.create_task(redis.set(document, embedding_vector))
    return embedding_vector


@app.post("/embedding")
async def get_embeddings():
    json = request.get_json()
    if not json or "content" not in json:
        return jsonify({"error": "Invalid request"}), 400
    content = request.get_json()["content"]
    if not isinstance(content, str):
        return jsonify({"error": "Content must be a string"}), 400
    results = await aembed_documents(content)
    return jsonify(results)


if __name__ == "__main__":
    import uvicorn

    asgi_app = WsgiToAsgi(app)
    uvicorn.run(asgi_app, host="0.0.0.0", port=8000)
