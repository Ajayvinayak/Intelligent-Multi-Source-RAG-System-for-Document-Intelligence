"""
RAG System - Client Interface & CLI Tool
Command-line interface for interacting with the RAG system without running the FastAPI server.
Useful for testing, batch processing, and local deployment.

Author: Ajay Vinayak Y
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import logging

from rag_system_main import RAGSystem, DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGClient:
    """Client interface for RAG system"""
    
    def __init__(self):
        """Initialize RAG client"""
        try:
            self.rag = RAGSystem()
            logger.info("✓ RAG System initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG System: {e}")
            sys.exit(1)
    
    def upload_single(self, file_path: str) -> None:
        """Upload a single document"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return
        
        try:
            result = self.rag.add_documents(file_path)
            self._print_result(result, "Upload Result")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
    
    def upload_batch(self, directory: str, pattern: str = "*") -> None:
        """Upload all documents from a directory"""
        path = Path(directory)
        
        if not path.is_dir():
            logger.error(f"Directory not found: {directory}")
            return
        
        files = list(path.glob(pattern))
        
        if not files:
            logger.warning(f"No files matching pattern '{pattern}' found in {directory}")
            return
        
        logger.info(f"Found {len(files)} files to upload")
        
        results = []
        for file_path in files:
            try:
                logger.info(f"Uploading: {file_path.name}")
                result = self.rag.add_documents(str(file_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to upload {file_path.name}: {e}")
        
        self._print_batch_results(results)
    
    def query(self, query: str, top_k: int = 5, show_sources: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            query: The question to ask
            top_k: Number of source documents to retrieve
            show_sources: Whether to display source documents
        
        Returns:
            Query result dictionary
        """
        try:
            logger.info(f"Processing query: {query}")
            result = self.rag.query(query, top_k=top_k)
            
            self._print_query_result(result, show_sources)
            return result
        
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return None
    
    def interactive_mode(self, top_k: int = 5):
        """Start interactive query mode"""
        print("\n" + "="*80)
        print("RAG System - Interactive Query Mode")
        print("Type 'exit' or 'quit' to exit")
        print("Type 'stats' to see system statistics")
        print("="*80 + "\n")
        
        while True:
            try:
                query = input("\n📝 Enter your query: ").strip()
                
                if query.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                if query.lower() == 'stats':
                    self.show_stats()
                    continue
                
                if not query:
                    print("Please enter a valid query")
                    continue
                
                self.query(query, top_k=top_k, show_sources=True)
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
    
    def show_stats(self) -> None:
        """Display system statistics"""
        stats = self.rag.get_stats()
        self._print_stats(stats)
    
    def test_mode(self) -> None:
        """Run predefined test queries"""
        test_queries = [
            "What is the main topic of the documents?",
            "Summarize the key findings",
            "What are the important data points?",
            "List all entities mentioned",
            "What conclusions are drawn?"
        ]
        
        logger.info("Running test queries...")
        results = []
        
        for query in test_queries:
            logger.info(f"\nTest Query: {query}")
            result = self.rag.query(query, top_k=3)
            results.append({
                "query": query,
                "answer": result["answer"],
                "confidence": result["confidence"],
                "retrieval_count": result["retrieval_count"]
            })
            print()
        
        self._save_test_results(results)
    
    # ========== Utility Methods ==========
    
    @staticmethod
    def _print_result(result: Dict[str, Any], title: str) -> None:
        """Pretty print a single result"""
        print(f"\n{'='*80}")
        print(f"✓ {title}")
        print('='*80)
        for key, value in result.items():
            print(f"{key:.<30} {value}")
    
    @staticmethod
    def _print_batch_results(results: List[Dict[str, Any]]) -> None:
        """Pretty print batch results"""
        print(f"\n{'='*80}")
        print("✓ Batch Upload Summary")
        print('='*80)
        
        total_chunks = sum(r.get('chunks_created', 0) for r in results)
        total_chars = sum(r.get('total_chars', 0) for r in results)
        
        print(f"Documents processed: {len(results)}")
        print(f"Total chunks created: {total_chunks}")
        print(f"Total characters: {total_chars:,}")
        
        print("\nDetails:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.get('document_id', 'Unknown')}")
            print(f"   Type: {result.get('file_type', 'Unknown')}")
            print(f"   Chunks: {result.get('chunks_created', 0)}")
            print(f"   Chars: {result.get('total_chars', 0):,}")
    
    @staticmethod
    def _print_query_result(result: Dict[str, Any], show_sources: bool = True) -> None:
        """Pretty print query result"""
        print(f"\n{'='*80}")
        print("📚 RAG Query Result")
        print('='*80)
        
        print(f"\n💡 Answer:\n{result['answer']}\n")
        
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"📖 Sources Retrieved: {result['retrieval_count']}")
        
        if show_sources and result['sources']:
            print(f"\n{'─'*80}")
            print("📌 Source Documents:")
            print('─'*80)
            
            for idx, source in enumerate(result['sources'], 1):
                print(f"\n{idx}. {source['source_file']}")
                print(f"   Type: {source['file_type']}")
                print(f"   Relevance: {source['relevance_score']:.3f}")
                print(f"   Content: {source['content'][:200]}...")
    
    @staticmethod
    def _print_stats(stats: Dict[str, Any]) -> None:
        """Pretty print system statistics"""
        print(f"\n{'='*80}")
        print("📊 RAG System Statistics")
        print('='*80)
        
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        
        if stats['documents']:
            print(f"\n{'─'*80}")
            print("Document List:")
            print('─'*80)
            
            for doc in stats['documents']:
                print(f"\n• {doc['source']}")
                print(f"  ID: {doc['id']}")
                print(f"  Type: {doc['type']}")
                print(f"  Chunks: {doc['chunks']}")
                print(f"  Size: {doc['chars']:,} characters")
                print(f"  Added: {doc['timestamp']}")
    
    @staticmethod
    def _save_test_results(results: List[Dict[str, Any]]) -> None:
        """Save test results to file"""
        output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ Test results saved to {output_file}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Intelligent RAG System - CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python rag_client.py --interactive
  
  # Upload single file
  python rag_client.py --upload document.pdf
  
  # Upload all PDFs from directory
  python rag_client.py --batch ./documents --pattern "*.pdf"
  
  # Query system
  python rag_client.py --query "What is the main topic?"
  
  # Run tests
  python rag_client.py --test
  
  # Show statistics
  python rag_client.py --stats
        """
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive query mode'
    )
    
    parser.add_argument(
        '--upload', '-u',
        type=str,
        metavar='FILE',
        help='Upload a single document'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=str,
        metavar='DIRECTORY',
        help='Upload all documents from directory'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*',
        help='File pattern for batch upload (default: *)'
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        metavar='QUERY',
        help='Query the RAG system'
    )
    
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=5,
        help='Number of source documents to retrieve (default: 5)'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run predefined test queries'
    )
    
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Show system statistics'
    )
    
    args = parser.parse_args()
    
    # Initialize client
    client = RAGClient()
    
    # Execute requested action
    if args.interactive:
        client.interactive_mode(top_k=args.top_k)
    
    elif args.upload:
        client.upload_single(args.upload)
    
    elif args.batch:
        client.upload_batch(args.batch, args.pattern)
    
    elif args.query:
        client.query(args.query, top_k=args.top_k)
    
    elif args.test:
        client.test_mode()
    
    elif args.stats:
        client.show_stats()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
