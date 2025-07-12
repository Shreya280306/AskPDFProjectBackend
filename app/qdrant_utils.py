

from qdrant_client.models import Filter, FieldCondition, PayloadSchemaType, PointStruct, Distance, VectorParams
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain.schema import Document
from collections import Counter
import uuid

# # Qdrant configuration
QDRANT_URL = "https://a26f47f6-04bd-4433-bc1d-30c41f48f0cd.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ZHDktyBPFyC7qCHT-zmPwr8URuKZ3_PwB-D4xuHCwj0"
QDRANT_COLLECTION = "pdfs_embeddings_sk"

Document(
    page_content="This is a chunk of text from a PDF.",
    metadata={"source": "employee_handbook.pdf"}
)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# def create_embeddings_and_store_qdrant(docs):
#     Qdrant.from_documents(
#         documents=docs,
#         embedding=embedding_model,  # ✅ USE embedding
#         url=QDRANT_URL,
#         api_key=QDRANT_API_KEY,
#         collection_name=QDRANT_COLLECTION
#     )

# def create_embeddings_and_store_qdrant(docs):
#     Qdrant.from_documents(
#         documents=docs,
#         embedding=embedding_model,
#         url=QDRANT_URL,
#         api_key=QDRANT_API_KEY,
#         collection_name=QDRANT_COLLECTION
#     )



# def create_embeddings_and_store_qdrant(docs):
#     for doc in docs:
#         print(doc.metadata)

#     # Create a Qdrant client
#     client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

#     # Create collection if it doesn't exist
#     if not client.collection_exists(collection_name=QDRANT_COLLECTION):
#         client.create_collection(
#             collection_name=QDRANT_COLLECTION,
#             vectors_config=VectorParams(
#                 size=384,  # 👈 Change this based on your embedding model output size
#                 distance=Distance.COSINE
#             )
#         )    
    
#     # Create index for metadata fields (only needed once per collection)
#     try:
#         client.create_payload_index(
#             collection_name=QDRANT_COLLECTION,
#             field_name="pdf_name",
#             field_schema=PayloadSchemaType.KEYWORD
#         )
#     except Exception as e:
#         if "already exists" not in str(e):
#             raise

#     # Store documents with embeddings
#     Qdrant.from_documents(
#         documents=docs,
#         embedding=embedding_model,
#         url=QDRANT_URL,
#         api_key=QDRANT_API_KEY,
#         collection_name=QDRANT_COLLECTION
#     )

def create_embeddings_and_store_qdrant(docs):
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name=QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=384,  # 👈 Change this based on your embedding model output size
                distance=Distance.COSINE
            )
        )    

    # ✅ Ensure payload index exists
    try:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="pdf_name",
            field_schema=PayloadSchemaType.KEYWORD
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise

    # ✅ Generate vectors and manually insert
    vectors = embedding_model.embed_documents([doc.page_content for doc in docs])

    points = []
    for doc, vector in zip(docs, vectors):
        # Flatten metadata
        payload = {"page_content": doc.page_content}
        payload.update(doc.metadata)  # <-- this flattens pdf_name

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload
        ))

    # ✅ Upload points
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points
    )




def qdrant_similarity_search(query, k=3):
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
   
    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embeddings=embedding_model
    )

    # Metadata filter based on PDF name
    pdf_name = "Business-Etiquette-ebook.pdf"
    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="pdf_name",
                match={"value": pdf_name}  # ✅ Correct usage (dict form)
            )
        ]
    )

    retrieved_chunks = vectorstore.similarity_search(
        query=query,
        k=k,
        filter=qdrant_filter
    )

    return retrieved_chunks
