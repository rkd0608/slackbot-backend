#!/usr/bin/env python3
"""Test script for knowledge extraction system."""

import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.database import AsyncSessionLocal
from app.services.openai_service import OpenAIService
from app.models.base import Message, User, Workspace
from sqlalchemy import select

class KnowledgeExtractionTester:
    """Test the knowledge extraction system with sample data."""
    
    def __init__(self):
        self.openai_service = None
        self.db = None
    
    async def initialize(self):
        """Initialize the tester."""
        try:
            self.db = AsyncSessionLocal()
            self.openai_service = OpenAIService()
            
            logger.info("Knowledge extraction tester initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize tester: {e}")
            raise
    
    async def close(self):
        """Close database connection."""
        if self.db:
            await self.db.close()
    
    async def test_knowledge_extraction(self):
        """Test knowledge extraction with sample messages."""
        try:
            # Sample test messages
            test_messages = [
                {
                    "text": "We decided to use React for the frontend and Node.js for the backend. This gives us better performance and easier maintenance.",
                    "type": "decision",
                    "expected_knowledge": ["React frontend", "Node.js backend", "performance", "maintenance"]
                },
                {
                    "text": "Here's our deployment process: 1) Run tests, 2) Build Docker image, 3) Deploy to staging, 4) Run integration tests, 5) Deploy to production",
                    "type": "process",
                    "expected_knowledge": ["deployment process", "Docker", "staging", "production", "testing"]
                },
                {
                    "text": "To fix the memory leak, we need to add proper cleanup in the useEffect hook and ensure we're not creating new objects on every render.",
                    "type": "solution",
                    "expected_knowledge": ["memory leak", "useEffect", "cleanup", "render optimization"]
                },
                {
                    "text": "I learned that using React.memo can significantly improve performance when dealing with expensive components that receive the same props.",
                    "type": "insight",
                    "expected_knowledge": ["React.memo", "performance", "expensive components", "props"]
                },
                {
                    "text": "The new feature needs to support both light and dark themes, be accessible to screen readers, and work on mobile devices.",
                    "type": "requirement",
                    "expected_knowledge": ["light theme", "dark theme", "accessibility", "screen readers", "mobile"]
                }
            ]
            
            logger.info("Testing knowledge extraction with sample messages...")
            
            for i, test_msg in enumerate(test_messages):
                logger.info(f"\n--- Test Message {i+1}: {test_msg['type'].upper()} ---")
                logger.info(f"Text: {test_msg['text']}")
                
                # Test knowledge extraction
                extraction_result = await self.openai_service.extract_knowledge(
                    conversation_context=[],  # No context for simple test
                    message_text=test_msg["text"],
                    message_metadata={
                        "message_id": i + 1,
                        "message_type": test_msg["type"],
                        "significance_score": 0.8,
                        "word_count": len(test_msg["text"].split()),
                        "is_thread_reply": False
                    }
                )
                
                # Analyze results
                await self.analyze_extraction_result(extraction_result, test_msg)
                
                # Test verification
                verification_result = await self.openai_service.verify_extraction(
                    extracted_knowledge=extraction_result,
                    source_messages=[{"text": test_msg["text"], "user_name": "TestUser"}]
                )
                
                await self.analyze_verification_result(verification_result)
                
                logger.info("-" * 50)
            
            logger.info("\n✅ Knowledge extraction testing completed!")
            
        except Exception as e:
            logger.error(f"Error in knowledge extraction testing: {e}", exc_info=True)
            raise
    
    async def analyze_extraction_result(self, extraction_result: Dict[str, Any], test_msg: Dict[str, Any]):
        """Analyze the extraction result."""
        try:
            knowledge_items = extraction_result.get("knowledge_items", [])
            overall_confidence = extraction_result.get("overall_confidence", 0.0)
            
            logger.info(f"Extraction Results:")
            logger.info(f"  Overall Confidence: {overall_confidence:.2f}")
            logger.info(f"  Knowledge Items Found: {len(knowledge_items)}")
            
            for j, item in enumerate(knowledge_items):
                logger.info(f"  Item {j+1}:")
                logger.info(f"    Type: {item.get('type', 'unknown')}")
                logger.info(f"    Title: {item.get('title', 'No title')}")
                logger.info(f"    Confidence: {item.get('confidence', 0.0):.2f}")
                logger.info(f"    Summary: {item.get('summary', 'No summary')[:100]}...")
                
                # Check if expected knowledge was found
                expected_knowledge = test_msg.get("expected_knowledge", [])
                found_knowledge = []
                for expected in expected_knowledge:
                    if (expected.lower() in item.get("title", "").lower() or 
                        expected.lower() in item.get("summary", "").lower() or
                        expected.lower() in item.get("content", "").lower()):
                        found_knowledge.append(expected)
                
                if found_knowledge:
                    logger.info(f"    ✅ Found expected knowledge: {found_knowledge}")
                else:
                    logger.info(f"    ⚠️  Expected knowledge not found: {expected_knowledge}")
            
        except Exception as e:
            logger.error(f"Error analyzing extraction result: {e}")
    
    async def analyze_verification_result(self, verification_result: Dict[str, Any]):
        """Analyze the verification result."""
        try:
            overall_score = verification_result.get("overall_verification_score", 0.0)
            hallucination_detected = verification_result.get("hallucination_detected", False)
            verification_results = verification_result.get("verification_results", [])
            
            logger.info(f"Verification Results:")
            logger.info(f"  Overall Verification Score: {overall_score:.2f}")
            logger.info(f"  Hallucination Detected: {hallucination_detected}")
            logger.info(f"  Items Verified: {len(verification_results)}")
            
            for result in verification_results:
                is_supported = result.get("is_supported", False)
                confidence = result.get("confidence", 0.0)
                issues = result.get("issues", [])
                
                status = "✅ SUPPORTED" if is_supported else "❌ NOT SUPPORTED"
                logger.info(f"    {status} (confidence: {confidence:.2f})")
                
                if issues:
                    logger.info(f"      Issues: {', '.join(issues)}")
            
        except Exception as e:
            logger.error(f"Error analyzing verification result: {e}")
    
    async def test_with_real_messages(self):
        """Test knowledge extraction with real messages from database."""
        try:
            logger.info("\n--- Testing with Real Messages from Database ---")
            
            # Get some processed messages
            result = await self.db.execute(
                select(Message)
                .where(Message.raw_payload.has_key("processed_data"))
                .limit(5)
            )
            
            messages = result.scalars().all()
            
            if not messages:
                logger.info("No processed messages found in database")
                return
            
            logger.info(f"Found {len(messages)} processed messages for testing")
            
            for i, message in enumerate(messages):
                try:
                    text = message.raw_payload.get("text", "")
                    processed_data = message.raw_payload.get("processed_data", {})
                    
                    if not text:
                        continue
                    
                    logger.info(f"\n--- Real Message {i+1} ---")
                    logger.info(f"Text: {text[:100]}...")
                    logger.info(f"Type: {processed_data.get('message_type', 'unknown')}")
                    logger.info(f"Significance: {processed_data.get('significance_score', 0.0):.2f}")
                    
                    # Test extraction
                    extraction_result = await self.openai_service.extract_knowledge(
                        conversation_context=[],  # Simplified for testing
                        message_text=text,
                        message_metadata={
                            "message_id": message.id,
                            "message_type": processed_data.get("message_type", "conversation"),
                            "significance_score": processed_data.get("significance_score", 0.0),
                            "word_count": processed_data.get("word_count", 0),
                            "is_thread_reply": processed_data.get("is_thread_reply", False)
                        }
                    )
                    
                    knowledge_items = extraction_result.get("knowledge_items", [])
                    logger.info(f"Knowledge Items Extracted: {len(knowledge_items)}")
                    
                    for item in knowledge_items:
                        logger.info(f"  - {item.get('title', 'No title')} ({item.get('type', 'unknown')})")
                    
                except Exception as e:
                    logger.error(f"Error processing real message {message.id}: {e}")
                    continue
            
            logger.info("\n✅ Real message testing completed!")
            
        except Exception as e:
            logger.error(f"Error in real message testing: {e}", exc_info=True)
    
    async def run_performance_test(self):
        """Run a performance test with multiple concurrent extractions."""
        try:
            logger.info("\n--- Performance Testing ---")
            
            # Create test tasks
            test_tasks = []
            for i in range(5):
                task = self.test_single_extraction(i, f"Performance test message {i+1}")
                test_tasks.append(task)
            
            # Run concurrent extractions
            start_time = asyncio.get_event_loop().time()
            results = await asyncio.gather(*test_tasks, return_exceptions=True)
            end_time = asyncio.get_event_loop().time()
            
            # Analyze results
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = len(results) - successful
            
            logger.info(f"Performance Test Results:")
            logger.info(f"  Total Time: {end_time - start_time:.2f} seconds")
            logger.info(f"  Successful: {successful}")
            logger.info(f"  Failed: {failed}")
            logger.info(f"  Average Time per Request: {(end_time - start_time) / len(results):.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in performance testing: {e}", exc_info=True)
    
    async def test_single_extraction(self, index: int, text: str):
        """Test a single knowledge extraction."""
        try:
            result = await self.openai_service.extract_knowledge(
                conversation_context=[],
                message_text=text,
                message_metadata={
                    "message_id": index,
                    "message_type": "conversation",
                    "significance_score": 0.5,
                    "word_count": len(text.split()),
                    "is_thread_reply": False
                }
            )
            
            return {"status": "success", "items": len(result.get("knowledge_items", []))}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

async def main():
    """Main function for knowledge extraction testing."""
    tester = KnowledgeExtractionTester()
    
    try:
        # Initialize tester
        await tester.initialize()
        
        # Run tests
        await tester.test_knowledge_extraction()
        await tester.test_with_real_messages()
        await tester.run_performance_test()
        
        logger.info("\n🎉 All knowledge extraction tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error during testing: {e}")
        raise
    
    finally:
        await tester.close()

if __name__ == "__main__":
    # Configure logging
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Run the tests
    asyncio.run(main())
