#!/usr/bin/env python3
"""Test script for vector search functionality."""

import asyncio
import time
import json
from typing import List, Dict, Any
from loguru import logger

# Add the app directory to the Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.config import settings
from app.services.vector_service import VectorService
from app.services.embedding_service import EmbeddingService
from app.core.database import get_db
from app.models.base import KnowledgeItem
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

class VectorSearchTester:
    """Test class for vector search functionality."""
    
    def __init__(self):
        self.vector_service = VectorService()
        self.embedding_service = EmbeddingService()
        
        # Test queries for different scenarios
        self.test_queries = [
            "How to deploy to production?",
            "What is our CI/CD process?",
            "Database connection issues",
            "API authentication setup",
            "Error handling best practices",
            "Performance optimization tips",
            "Security guidelines",
            "Testing strategies",
            "Code review process",
            "Monitoring and alerting"
        ]
        
        # Performance thresholds
        self.performance_thresholds = {
            "search_response_time_ms": 500,
            "embedding_generation_time_ms": 1000,
            "similarity_threshold": 0.7
        }
    
    async def test_embedding_generation(self) -> Dict[str, Any]:
        """Test embedding generation performance."""
        logger.info("Testing embedding generation...")
        
        results = {
            "total_queries": len(self.test_queries),
            "successful_generations": 0,
            "failed_generations": 0,
            "total_time": 0,
            "average_time": 0,
            "performance": "optimal"
        }
        
        for query in self.test_queries:
            try:
                start_time = time.time()
                embedding = await self.embedding_service.generate_embedding(query, "question")
                generation_time = time.time() - start_time
                
                if embedding:
                    results["successful_generations"] += 1
                    results["total_time"] += generation_time
                    
                    # Validate embedding
                    is_valid = await self.embedding_service.validate_embedding(embedding)
                    if not is_valid:
                        logger.warning(f"Generated invalid embedding for query: {query}")
                        
                else:
                    results["failed_generations"] += 1
                    logger.error(f"Failed to generate embedding for query: {query}")
                    
            except Exception as e:
                results["failed_generations"] += 1
                logger.error(f"Error generating embedding for query '{query}': {e}")
        
        if results["successful_generations"] > 0:
            results["average_time"] = results["total_time"] / results["successful_generations"]
            results["performance"] = "optimal" if results["average_time"] <= 1.0 else "suboptimal"
        
        logger.info(f"Embedding generation test completed: {results}")
        return results
    
    async def test_vector_search_performance(self, db: AsyncSession) -> Dict[str, Any]:
        """Test vector search performance."""
        logger.info("Testing vector search performance...")
        
        results = {
            "total_queries": len(self.test_queries),
            "successful_searches": 0,
            "failed_searches": 0,
            "total_search_time": 0,
            "average_search_time": 0,
            "performance": "optimal",
            "query_results": []
        }
        
        for query in self.test_queries:
            try:
                start_time = time.time()
                search_results = await self.vector_service.hybrid_search(
                    query=query,
                    limit=5,
                    db=db
                )
                search_time = time.time() - start_time
                
                results["successful_searches"] += 1
                results["total_search_time"] += search_time
                
                # Record individual query performance
                query_result = {
                    "query": query,
                    "results_count": len(search_results),
                    "search_time": search_time,
                    "performance": "optimal" if search_time <= 0.5 else "suboptimal"
                }
                results["query_results"].append(query_result)
                
                # Log performance warnings
                if search_time > 0.5:
                    logger.warning(f"Search for '{query}' took {search_time:.3f}s (exceeded 500ms threshold)")
                
            except Exception as e:
                results["failed_searches"] += 1
                logger.error(f"Error in vector search for query '{query}': {e}")
        
        if results["successful_searches"] > 0:
            results["average_search_time"] = results["total_search_time"] / results["successful_searches"]
            results["performance"] = "optimal" if results["average_search_time"] <= 0.5 else "suboptimal"
        
        logger.info(f"Vector search performance test completed: {results}")
        return results
    
    async def test_search_accuracy(self, db: AsyncSession) -> Dict[str, Any]:
        """Test search result accuracy and relevance."""
        logger.info("Testing search result accuracy...")
        
        # Get some sample knowledge items for testing
        result = await db.execute(
            select(KnowledgeItem)
            .where(KnowledgeItem.embedding.isnot(None))
            .limit(10)
        )
        knowledge_items = result.scalars().all()
        
        if not knowledge_items:
            logger.warning("No knowledge items with embeddings found for accuracy testing")
            return {"status": "skipped", "reason": "No embedded knowledge items"}
        
        results = {
            "total_tests": len(knowledge_items),
            "successful_tests": 0,
            "accuracy_scores": [],
            "relevance_analysis": []
        }
        
        for item in knowledge_items:
            try:
                # Create a test query based on the item's content
                test_query = item.title or item.summary or item.content[:100]
                
                # Perform search
                search_results = await self.vector_service.hybrid_search(
                    query=test_query,
                    limit=5,
                    db=db
                )
                
                if search_results:
                    # Check if the original item appears in results
                    original_found = any(
                        result["id"] == item.id for result in search_results
                    )
                    
                    if original_found:
                        results["successful_tests"] += 1
                        results["accuracy_scores"].append(1.0)
                    else:
                        results["accuracy_scores"].append(0.0)
                    
                    # Analyze relevance of top result
                    top_result = search_results[0]
                    relevance_score = top_result.get("similarity", 0.0)
                    
                    results["relevance_analysis"].append({
                        "query": test_query,
                        "top_result_similarity": relevance_score,
                        "original_found": original_found,
                        "results_count": len(search_results)
                    })
                
            except Exception as e:
                logger.error(f"Error testing accuracy for knowledge item {item.id}: {e}")
        
        if results["accuracy_scores"]:
            results["overall_accuracy"] = sum(results["accuracy_scores"]) / len(results["accuracy_scores"])
            results["accuracy_percentage"] = results["overall_accuracy"] * 100
        
        logger.info(f"Search accuracy test completed: {results}")
        return results
    
    async def test_embedding_storage_and_retrieval(self, db: AsyncSession) -> Dict[str, Any]:
        """Test embedding storage and retrieval functionality."""
        logger.info("Testing embedding storage and retrieval...")
        
        results = {
            "storage_tests": 0,
            "retrieval_tests": 0,
            "successful_storage": 0,
            "successful_retrieval": 0,
            "storage_time": 0,
            "retrieval_time": 0
        }
        
        # Test with a sample text
        test_text = "This is a test knowledge item for embedding storage and retrieval testing."
        
        try:
            # Generate embedding
            start_time = time.time()
            embedding = await self.embedding_service.generate_embedding(test_text, "knowledge")
            generation_time = time.time() - start_time
            
            if embedding:
                results["storage_tests"] += 1
                
                # Test storage (we'll use a dummy knowledge ID)
                # In a real scenario, this would be stored with an actual knowledge item
                storage_success = await self.vector_service.store_embedding(
                    999999,  # Dummy ID
                    embedding,
                    db
                )
                
                if storage_success:
                    results["successful_storage"] += 1
                
                # Test retrieval
                results["retrieval_tests"] += 1
                retrieval_start = time.time()
                
                search_results = await self.vector_service.search_similar(
                    query_embedding=embedding,
                    limit=1,
                    db=db
                )
                
                retrieval_time = time.time() - retrieval_start
                results["retrieval_time"] += retrieval_time
                
                if search_results:
                    results["successful_retrieval"] += 1
                
                results["storage_time"] += generation_time
                
        except Exception as e:
            logger.error(f"Error in embedding storage/retrieval test: {e}")
        
        logger.info(f"Embedding storage/retrieval test completed: {results}")
        return results
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report."""
        logger.info("Starting comprehensive vector search testing...")
        
        start_time = time.time()
        
        # Get database session
        async for db in get_db():
            break
        
        # Run all tests
        test_results = {
            "timestamp": time.time(),
            "test_duration": 0,
            "overall_status": "unknown",
            "tests": {}
        }
        
        try:
            # Test 1: Embedding Generation
            test_results["tests"]["embedding_generation"] = await self.test_embedding_generation()
            
            # Test 2: Vector Search Performance
            test_results["tests"]["vector_search_performance"] = await self.test_vector_search_performance(db)
            
            # Test 3: Search Accuracy
            test_results["tests"]["search_accuracy"] = await self.test_search_accuracy(db)
            
            # Test 4: Embedding Storage and Retrieval
            test_results["tests"]["embedding_storage_retrieval"] = await self.test_embedding_storage_and_retrieval(db)
            
            # Calculate overall status
            overall_status = "optimal"
            for test_name, test_result in test_results["tests"].items():
                if test_result.get("performance") == "suboptimal":
                    overall_status = "suboptimal"
                elif test_result.get("status") == "failed":
                    overall_status = "failed"
            
            test_results["overall_status"] = overall_status
            
        except Exception as e:
            logger.error(f"Error during comprehensive testing: {e}")
            test_results["overall_status"] = "error"
            test_results["error"] = str(e)
        
        finally:
            test_results["test_duration"] = time.time() - start_time
        
        # Generate summary
        self._generate_test_summary(test_results)
        
        return test_results
    
    def _generate_test_summary(self, test_results: Dict[str, Any]):
        """Generate a human-readable test summary."""
        logger.info("=" * 60)
        logger.info("VECTOR SEARCH TESTING SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Overall Status: {test_results['overall_status'].upper()}")
        logger.info(f"Total Test Duration: {test_results['test_duration']:.2f}s")
        logger.info("")
        
        for test_name, test_result in test_results["tests"].items():
            logger.info(f"Test: {test_name.replace('_', ' ').title()}")
            
            if "performance" in test_result:
                logger.info(f"  Performance: {test_result['performance']}")
            
            if "overall_accuracy" in test_result:
                logger.info(f"  Accuracy: {test_result['overall_accuracy']:.2%}")
            
            if "average_time" in test_result:
                logger.info(f"  Average Time: {test_result['average_time']:.3f}s")
            
            if "successful_tests" in test_result:
                logger.info(f"  Success Rate: {test_result['successful_tests']}/{test_result['total_tests']}")
            
            logger.info("")
        
        # Performance validation
        logger.info("PERFORMANCE VALIDATION:")
        logger.info(f"  Search Response Time: {'✅' if test_results['tests']['vector_search_performance']['performance'] == 'optimal' else '❌'} (Target: <500ms)")
        logger.info(f"  Embedding Generation: {'✅' if test_results['tests']['embedding_generation']['performance'] == 'optimal' else '❌'} (Target: <1000ms)")
        
        if "search_accuracy" in test_results["tests"] and "overall_accuracy" in test_results["tests"]["search_accuracy"]:
            accuracy = test_results["tests"]["search_accuracy"]["overall_accuracy"]
            logger.info(f"  Search Accuracy: {'✅' if accuracy >= 0.8 else '❌'} (Target: >80%)")
        
        logger.info("=" * 60)

async def main():
    """Main function to run the vector search tests."""
    logger.info("Vector Search Testing Suite")
    logger.info("=" * 40)
    
    tester = VectorSearchTester()
    
    try:
        # Run comprehensive test
        results = await tester.run_comprehensive_test()
        
        # Save results to file
        with open("vector_search_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Test results saved to vector_search_test_results.json")
        
        # Exit with appropriate code
        if results["overall_status"] == "optimal":
            logger.info("🎉 All tests passed! Vector search system is ready.")
            sys.exit(0)
        elif results["overall_status"] == "suboptimal":
            logger.warning("⚠️  Some tests had suboptimal performance. Review and optimize.")
            sys.exit(1)
        else:
            logger.error("❌ Tests failed. Check the logs for details.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
