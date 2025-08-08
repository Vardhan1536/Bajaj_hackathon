# main.py (Final Stateful Version)
import os
import time
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List
from uuid import uuid4
from sqlalchemy.orm import Session # <-- IMPORTANT: Add this import

# --- Import custom modules ---
from database import get_db, Document, AnswerLog, create_db_and_tables # <-- Add create_db_and_tables
from processing import load_document_from_url, chunk_text
from vector_store import create_pinecone_index, embed_and_store, query_vector_store
# We no longer need delete_pinecone_index for the main logic
from llm_handler import get_answer_from_llm

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
API_TOKEN = os.getenv("API_TOKEN")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Stateful LLM Document Processing System",
    description="An efficient, stateful API for processing queries against documents.",
    version="1.1.0"
)

# --- Database Startup Event ---
# This will run once when the app starts to ensure tables are created.
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# --- Security (No changes here) ---
bearer_scheme = HTTPBearer()
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing authentication token")
    return credentials

# --- Pydantic Models (No changes here) ---
class HackRxRequest(BaseModel):
    documents: str = Field(..., example="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=...")
    questions: List[str] = Field(..., example=["What is the grace period?"])

class HackRxResponse(BaseModel):
    answers: List[str]

# --- API ENDPOINT (REPLACED WITH STATEFUL LOGIC) ---
@app.post(
    "/api/v1/hackrx/run",
    response_model=HackRxResponse,
    dependencies=[Depends(get_current_user)],
    tags=["Core Logic"]
)
async def process_documents_stateful(request: HackRxRequest, db: Session = Depends(get_db)):
    """
    This stateful endpoint performs the RAG pipeline with caching:
    1. Checks if the document is already processed by querying PostgreSQL.
    2. If not, it processes, embeds, and stores the document permanently.
    3. Answers questions based on the document's vector index.
    4. Logs all Q&A interactions for auditing.
    """
    document_url = request.documents
    questions = request.questions

    try:
        # 1. Check if document is already in our database (caching logic)
        existing_document = db.query(Document).filter(Document.document_url == document_url).first()
        
        if existing_document:
            print(f"Document found in cache. Using existing Pinecone index: {existing_document.pinecone_index_name}")
            index_name = existing_document.pinecone_index_name
            document_id = existing_document.id
        else:
            print("New document detected. Starting full processing pipeline...")
            # Create a PERMANENT, unique index name for the new document
            index_name = f"doc-permanent-{str(uuid4())[:12]}"
            
            # 2a. Setup Pinecone Index & Process Document
            create_pinecone_index(index_name)
            document_text = load_document_from_url(document_url)
            if not document_text:
                raise HTTPException(status_code=400, detail="Failed to load or process document.")
            
            chunks = chunk_text(document_text)

            # 2b. Prepare Chunks and Store in Pinecone
            chunks_with_ids = [{'id': str(uuid4()), 'text': chunk} for chunk in chunks]
            embed_and_store(
                chunks_with_ids=chunks_with_ids, 
                index_name=index_name,
                document_id=document_url # Use URL as a general identifier in metadata
            )

            # 2c. Save the new document record to PostgreSQL
            new_document = Document(
                document_url=document_url,
                pinecone_index_name=index_name,
                status="processed"
            )
            db.add(new_document)
            db.commit()
            db.refresh(new_document)
            document_id = new_document.id
            print(f"New document saved to database with ID: {document_id}")

        # 3. Process Each Question (Retrieval-Augmented Generation)
        answers = []
        for question in questions:
            context = query_vector_store(question, index_name, top_k=5)
            answer = get_answer_from_llm(context, question)
            answers.append(answer)

            # 4. Log the question and answer to our audit table
            log_entry = AnswerLog(
                document_id=document_id,
                question=question,
                answer=answer,
                context_used=context
            )
            db.add(log_entry)
        
        db.commit() # Commit all the new log entries at once
        print(f"Logged {len(answers)} Q&A pairs to the database for document ID: {document_id}")
            
        return HackRxResponse(answers=answers)

    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback() # Rollback DB changes if an error occurs mid-transaction
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# --- To run the app locally for testing (No changes here) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)