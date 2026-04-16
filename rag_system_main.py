"""
Intelligent Multi-Source RAG System for Document Intelligence
A production-grade Retrieval-Augmented Generation system that handles PDFs, text files, 
and structured data sources with semantic search and accurate fact-based responses.

Author: Ajay Vinayak Y
Date: 2024
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

import numpy as np
from dotenv import load_dotenv

# LLM and Embedding imports
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Vector Store imports
from langchain.vectorstores import Pinecone
import pinecone

# Document Processing imports
from langchain.document_loaders import PDFPlumberLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# FastAPI and Web
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class DocumentMetadata:
    """Metadata for processed documents"""
    document_id: str
    source: str
    file_type: str
    chunk_count: int
    total_chars: int
    processing_time: float
    timestamp: str


class QueryRequest(BaseModel):
    """Request model for queries"""
    query: str
    top_k: int = 5
    include_sources: bool = True


class QueryResponse(BaseModel):
    """Response model for queries"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    retrieval_count: int


class UploadResponse(BaseModel):
    """Response model for file uploads"""
    status: str
    document_id: str
    chunks_created: int
    message: str


# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """Handles document loading and text splitting across multiple formats"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info(f"DocumentProcessor initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and parse PDF files"""
        try:
            logger.info(f"Loading PDF: {file_path}")
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} pages from PDF")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise
    
    def load_text(self, file_path: str) -> List[Document]:
        """Load and parse text files"""
        try:
            logger.info(f"Loading text file: {file_path}")
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            logger.info(f"Successfully loaded text document")
            return documents
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            raise
    
    def load_csv(self, file_path: str) -> List[Document]:
        """Load and parse CSV files"""
        try:
            logger.info(f"Loading CSV: {file_path}")
            loader = CSVLoader(file_path)
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} rows from CSV")
            return documents
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {str(e)}")
            raise
    
    def load_json(self, file_path: str) -> List[Document]:
        """Load and parse JSON files"""
        try:
            logger.info(f"Loading JSON: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    content = json.dumps(item, indent=2)
                    doc = Document(
                        page_content=content,
                        metadata={"source": file_path, "index": idx, "type": "json"}
                    )
                    documents.append(doc)
            else:
                content = json.dumps(data, indent=2)
                doc = Document(page_content=content, metadata={"source": file_path, "type": "json"})
                documents.append(doc)
            
            logger.info(f"Successfully loaded {len(documents)} items from JSON")
            return documents
        except Exception as e:
            logger.error(f"Error loading JSON {file_path}: {str(e)}")
            raise
    
    def process_document(self, file_path: str) -> Tuple[List[Document], str]:
        """
        Main processing function that determines file type and loads accordingly
        Returns: (list of Document chunks, file_type)
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            raw_docs = self.load_pdf(file_path)
            file_type = 'pdf'
        elif file_ext == '.txt':
            raw_docs = self.load_text(file_path)
            file_type = 'text'
        elif file_ext == '.csv':
            raw_docs = self.load_csv(file_path)
            file_type = 'csv'
        elif file_ext == '.json':
            raw_docs = self.load_json(file_path)
            file_type = 'json'
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(raw_docs)
        
        # Enrich metadata
        for chunk in chunks:
            if 'file_type' not in chunk.metadata:
                chunk.metadata['file_type'] = file_type
            if 'document_id' not in chunk.metadata:
                chunk.metadata['document_id'] = hashlib.md5(
                    file_path.encode()
                ).hexdigest()[:16]
        
        logger.info(f"Split document into {len(chunks)} chunks")
        return chunks, file_type


# ============================================================================
# RAG SYSTEM
# ============================================================================

class RAGSystem:
    """Main Retrieval-Augmented Generation System"""
    
    def __init__(self):
        """Initialize RAG system with embeddings and LLM"""
        try:
            # Initialize embeddings
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            logger.info("✓ OpenAI Embeddings initialized")
            
            # Initialize Pinecone
            pinecone_key = os.getenv("PINECONE_API_KEY")
            pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
            pinecone_index = os.getenv("PINECONE_INDEX_NAME", "rag-system")
            
            if not pinecone_key or not pinecone_env:
                raise ValueError("Pinecone credentials not set in environment variables")
            
            pinecone.init(api_key=pinecone_key, environment=pinecone_env)
            logger.info("✓ Pinecone initialized")
            
            # Initialize or get existing index
            self.index_name = pinecone_index
            self._initialize_pinecone_index()
            
            # Initialize vector store
            self.vector_store = Pinecone(
                index=pinecone.Index(self.index_name),
                embedding_function=self.embeddings.embed_query,
                text_key="text"
            )
            logger.info("✓ Vector store initialized")
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000
            )
            logger.info("✓ ChatGPT LLM initialized")
            
            # Initialize document processor
            self.processor = DocumentProcessor()
            
            # Metadata storage
            self.document_metadata: Dict[str, DocumentMetadata] = {}
            
            logger.info("✓ RAG System fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG System: {str(e)}")
            raise
    
    def _initialize_pinecone_index(self):
        """Create Pinecone index if it doesn't exist"""
        try:
            # Check if index exists
            existing_indexes = pinecone.list_indexes()
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embeddings dimension
                    metric="cosine"
                )
                logger.info(f"✓ Index {self.index_name} created successfully")
            else:
                logger.info(f"✓ Index {self.index_name} already exists")
        except Exception as e:
            logger.warning(f"Could not initialize index: {str(e)}")
    
    def add_documents(self, file_path: str) -> Dict[str, Any]:
        """
        Process and add documents to the vector store
        
        Args:
            file_path: Path to document file
        
        Returns:
            Dictionary with upload metadata
        """
        try:
            start_time = datetime.now()
            
            # Process document
            chunks, file_type = self.processor.process_document(file_path)
            
            if not chunks:
                raise ValueError(f"No content extracted from {file_path}")
            
            # Add to vector store
            doc_id = hashlib.md5(file_path.encode()).hexdigest()[:16]
            
            # Store metadata for each chunk
            metadatas = [
                {
                    "source": file_path,
                    "document_id": doc_id,
                    "chunk_index": idx,
                    "file_type": file_type
                }
                for idx in range(len(chunks))
            ]
            
            # Add to vector store
            self.vector_store.add_documents(chunks, metadatas=metadatas)
            
            # Store document metadata
            total_chars = sum(len(chunk.page_content) for chunk in chunks)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metadata = DocumentMetadata(
                document_id=doc_id,
                source=file_path,
                file_type=file_type,
                chunk_count=len(chunks),
                total_chars=total_chars,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
            self.document_metadata[doc_id] = metadata
            
            logger.info(
                f"✓ Document added: {file_path} "
                f"({len(chunks)} chunks, {total_chars} chars, {processing_time:.2f}s)"
            )
            
            return {
                "status": "success",
                "document_id": doc_id,
                "chunks_created": len(chunks),
                "file_type": file_type,
                "total_chars": total_chars,
                "processing_time": processing_time
            }
        
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG system for answers backed by source documents
        
        Args:
            query: User's question/query
            top_k: Number of relevant documents to retrieve
        
        Returns:
            Dictionary with answer, sources, and confidence score
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Retrieve relevant documents
            retrieved_docs = self.vector_store.similarity_search_with_score(
                query, k=top_k
            )
            
            if not retrieved_docs:
                return {
                    "answer": "I could not find relevant information in the knowledge base.",
                    "sources": [],
                    "confidence": 0.0,
                    "retrieval_count": 0
                }
            
            # Extract context from retrieved documents
            context = "\n\n".join([
                f"Source {idx+1}: {doc.page_content}"
                for idx, (doc, score) in enumerate(retrieved_docs)
            ])
            
            # Create prompt with context
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""Use the following pieces of context to answer the question. 
If you don't know the answer, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Provide a detailed, accurate answer based ONLY on the context provided:"""
            )
            
            # Generate answer
            response = self.llm.predict(
                text=prompt.format(context=context, question=query)
            )
            
            # Prepare sources
            sources = [
                {
                    "content": doc.page_content[:500],  # First 500 chars
                    "source_file": doc.metadata.get("source", "Unknown"),
                    "relevance_score": float(score),
                    "file_type": doc.metadata.get("file_type", "Unknown")
                }
                for doc, score in retrieved_docs
            ]
            
            # Calculate average confidence (inverse of average distance)
            avg_score = np.mean([score for _, score in retrieved_docs])
            confidence = max(0, 1 - avg_score)
            
            logger.info(f"✓ Query processed successfully. Found {len(sources)} sources.")
            
            return {
                "answer": response.strip(),
                "sources": sources,
                "confidence": float(confidence),
                "retrieval_count": len(sources)
            }
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_documents": len(self.document_metadata),
            "total_chunks": sum(m.chunk_count for m in self.document_metadata.values()),
            "documents": [
                {
                    "id": doc_id,
                    "source": meta.source,
                    "type": meta.file_type,
                    "chunks": meta.chunk_count,
                    "chars": meta.total_chars,
                    "timestamp": meta.timestamp
                }
                for doc_id, meta in self.document_metadata.items()
            ]
        }


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Intelligent RAG System API",
    description="Retrieval-Augmented Generation system for multi-source document intelligence",
    version="1.0.0"
)

# Initialize RAG system
try:
    rag_system = RAGSystem()
    logger.info("✓ RAG System initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG System: {e}")
    rag_system = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if rag_system else "error",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF, TXT, CSV, JSON)
    
    Supported formats:
    - PDF (.pdf)
    - Text (.txt)
    - CSV (.csv)
    - JSON (.json)
    """
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG System not initialized")
    
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Process document
        result = rag_system.add_documents(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return UploadResponse(
            status="success",
            document_id=result["document_id"],
            chunks_created=result["chunks_created"],
            message=f"Document processed successfully. Created {result['chunks_created']} chunks."
        )
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system for answers backed by documents
    
    The system will search through all uploaded documents and provide
    answers grounded in the actual content, along with source references.
    """
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG System not initialized")
    
    try:
        result = rag_system.query(request.query, top_k=request.top_k)
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Query failed: {str(e)}")


@app.get("/stats")
async def get_system_stats():
    """Get system statistics and metadata"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG System not initialized")
    
    return rag_system.get_stats()


@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG System not initialized")
    
    stats = rag_system.get_stats()
    return stats["documents"]


if __name__ == "__main__":
    # Start FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
