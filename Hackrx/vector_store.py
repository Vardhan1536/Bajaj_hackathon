# vector_store.py (Updated for Pinecone Serverless)
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()


# --- Initialize Models and Database ---

# The new way to initialize Pinecone. It automatically uses the PINECONE_API_KEY.
# No 'environment' variable is needed for serverless.
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Using a high-quality sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# The dimension of the embeddings from this model
EMBEDDING_DIMENSION = 384


def create_pinecone_index(index_name: str):
    """Creates a Pinecone serverless index if it doesn't already exist."""
    # Check if the index already exists
    if index_name not in pc.list_indexes().names():
        # If not, create the index
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric='cosine',  # Cosine similarity is great for semantic search
            # This specifies the cloud provider and region for the serverless index
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")


def embed_and_store(chunks_with_ids: list[dict], index_name: str, document_id: str):
    """
    Generates embeddings for text chunks and stores them in Pinecone.

    Args:
        chunks_with_ids: A list of dictionaries, where each dict has 'id' and 'text'.
        index_name: The name of the Pinecone index.
        document_id: The ID of the parent document, for metadata.
    """
    index = pc.Index(index_name)
    batch_size = 128

    for i in range(0, len(chunks_with_ids), batch_size):
        batch = chunks_with_ids[i:i + batch_size]
        
        # Extract the text for embedding
        batch_chunks_text = [item['text'] for item in batch]
        embeddings = embedding_model.encode(batch_chunks_text).tolist()

        # Extract the pre-generated IDs
        ids = [item['id'] for item in batch]
        
        # Create metadata. Storing the document_id is crucial for filtering queries.
        # Storing the original text is still useful for direct retrieval from Pinecone.
        metadata = [
            {'text': item['text'], 'document_id': document_id} 
            for item in batch
        ]

        # Upsert using the shared IDs
        index.upsert(vectors=list(zip(ids, embeddings, metadata)))

    print(f"Successfully stored {len(chunks_with_ids)} chunks in index '{index_name}'.")


def query_vector_store(query: str, index_name: str, top_k: int = 5) -> str:
    """Queries the vector store to find the most relevant context."""
    # Get a handle to the specific index
    index = pc.Index(index_name)

    # Generate embedding for the query
    query_embedding = embedding_model.encode(query).tolist()

    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Combine the text from the results to form the context
    context = " ".join([match['metadata']['text'] for match in results['matches']])
    return context

def delete_pinecone_index(index_name: str):
    """Deletes a Pinecone index."""
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
        print(f"Cleaned up and deleted index '{index_name}'.")