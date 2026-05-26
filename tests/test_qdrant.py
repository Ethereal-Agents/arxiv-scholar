from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, PointStruct, SparseVector

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="test",
    vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    sparse_vectors_config={"bm25": SparseVectorParams()}
)

client.upsert(
    collection_name="test",
    points=[
        PointStruct(
            id=1,
            vector={
                "": [0.1, 0.2, 0.3, 0.4],
                "bm25": SparseVector(indices=[1, 2], values=[0.5, 0.6])
            },
            payload={"text": "hello world"}
        )
    ]
)
print("Upsert successful")
