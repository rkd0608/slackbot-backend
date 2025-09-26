#!/usr/bin/env python3
"""
Test script for the new conversation-level processing system.

This demonstrates the key architectural improvements:
1. Conversation boundary detection
2. State analysis (starting, in-progress, paused, completed, abandoned)
3. Conversation-level knowledge extraction (only from completed conversations)
4. Full context assembly with narrative understanding
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.database import get_session_factory
from app.services.enhanced_conversation_state_manager import EnhancedConversationStateManager as ConversationStateManager, ConversationState
from app.services.conversation_knowledge_extractor import ConversationKnowledgeExtractor
from app.models.base import Conversation, Message
from sqlalchemy import select, desc
from loguru import logger


async def test_conversation_state_analysis():
    """Test conversation state detection on real conversations."""
    print("\n🔍 TESTING CONVERSATION STATE ANALYSIS")
    print("=" * 60)
    
    session_factory = get_session_factory()
    async with session_factory() as db:
        # Get recent conversations
        query = select(Conversation).order_by(desc(Conversation.created_at)).limit(5)
        result = await db.execute(query)
        conversations = result.scalars().all()
        
        if not conversations:
            print("❌ No conversations found in database")
            return
        
        state_manager = ConversationStateManager()
        
        for conv in conversations:
            print(f"\n📊 Analyzing Conversation {conv.id}")
            print(f"   Channel: {conv.slack_channel_id}")
            print(f"   Created: {conv.created_at}")
            
            # Analyze conversation state
            boundary = await state_manager.analyze_conversation_state(conv.id, db)
            
            print(f"   State: {boundary.state.value}")
            print(f"   Confidence: {boundary.confidence:.2f}")
            print(f"   Ready for extraction: {await state_manager.should_extract_knowledge(boundary)}")
            
            if boundary.topic:
                print(f"   Topic: {boundary.topic}")
            
            # Get message count
            msg_query = select(Message).where(Message.conversation_id == conv.id)
            msg_result = await db.execute(msg_query)
            messages = msg_result.scalars().all()
            print(f"   Messages: {len(messages)}")
            
            if messages:
                print(f"   Last message: {messages[-1].content[:50]}...")


async def test_conversation_level_extraction():
    """Test knowledge extraction from completed conversations."""
    print("\n🧠 TESTING CONVERSATION-LEVEL KNOWLEDGE EXTRACTION")
    print("=" * 60)
    
    session_factory = get_session_factory()
    async with session_factory() as db:
        extractor = ConversationKnowledgeExtractor()
        
        # Find extraction-ready conversations
        ready_conversations = await extractor._find_extraction_ready_conversations(
            workspace_id=1, db=db, limit=3
        )
        
        print(f"Found {len(ready_conversations)} conversations ready for extraction")
        
        if not ready_conversations:
            print("❌ No conversations ready for extraction")
            print("   This is normal - conversations need to be COMPLETED first")
            return
        
        for conv_id in ready_conversations:
            print(f"\n📝 Extracting from Conversation {conv_id}")
            
            # Test extraction readiness analysis
            readiness = await extractor.analyze_extraction_readiness(conv_id, db)
            print(f"   Readiness Analysis:")
            print(f"   - State: {readiness.get('current_state')}")
            print(f"   - Ready: {readiness.get('is_ready')}")
            print(f"   - Blocking factors: {readiness.get('blocking_factors', [])}")
            
            if readiness.get('is_ready'):
                # Extract knowledge
                knowledge_items = await extractor._extract_from_single_conversation(
                    conv_id, workspace_id=1, db=db
                )
                print(f"   ✅ Extracted {len(knowledge_items)} knowledge items")
                
                for item in knowledge_items:
                    print(f"      - {item.title} (confidence: {item.confidence_score:.2f})")
            else:
                print(f"   ⏳ Not ready for extraction")


async def test_conversation_boundary_detection():
    """Test conversation boundary detection across channels."""
    print("\n🎯 TESTING CONVERSATION BOUNDARY DETECTION")
    print("=" * 60)
    
    session_factory = get_session_factory()
    async with session_factory() as db:
        state_manager = ConversationStateManager()
        
        # Get unique channels
        query = select(Conversation.slack_channel_id).distinct()
        result = await db.execute(query)
        channels = [row[0] for row in result.fetchall()]
        
        print(f"Found {len(channels)} channels with conversations")
        
        for channel_id in channels[:3]:  # Test first 3 channels
            print(f"\n📺 Channel: {channel_id}")
            
            boundaries = await state_manager.detect_conversation_boundaries(
                channel_id, db, lookback_hours=24
            )
            
            print(f"   Detected {len(boundaries)} conversation boundaries")
            
            state_counts = {}
            for boundary in boundaries:
                state = boundary.state.value
                state_counts[state] = state_counts.get(state, 0) + 1
            
            for state, count in state_counts.items():
                print(f"   - {state}: {count}")


async def demonstrate_architecture_improvement():
    """Demonstrate the key architectural improvements."""
    print("\n🏗️  ARCHITECTURE IMPROVEMENT DEMONSTRATION")
    print("=" * 60)
    
    print("""
KEY IMPROVEMENTS IMPLEMENTED:

1. ✅ CONVERSATION BOUNDARY DETECTION
   - Detects when conversations start, continue, pause, complete, or are abandoned
   - Uses both rule-based patterns and AI analysis
   - Prevents premature knowledge extraction

2. ✅ CONVERSATION-LEVEL PROCESSING  
   - Waits for conversations to COMPLETE before extracting knowledge
   - Assembles full narrative context (problem → discussion → resolution)
   - Processes complete thoughts, not fragmented messages

3. ✅ STATE-AWARE EXTRACTION
   - Only extracts from conversations in COMPLETED state
   - Handles paused conversations ("step-by-step coming up")
   - Identifies abandoned discussions

4. ✅ INTELLIGENT TRIGGERING
   - Analyzes conversation state after each new message
   - Automatically triggers extraction when conversations complete
   - Scheduled batch processing for completed conversations

5. ✅ CONTEXT ASSEMBLY
   - Builds rich conversation context with participant mapping
   - Maintains chronological message flow
   - Includes temporal context and topic evolution

BEFORE (Message-Level):
❌ Processed individual messages as they arrived
❌ Extracted incomplete information 
❌ No understanding of conversation state
❌ Fragmented knowledge without context

AFTER (Conversation-Level):
✅ Waits for complete conversations
✅ Extracts knowledge with full narrative context
✅ Understands conversation boundaries and states
✅ Produces coherent, actionable knowledge
    """)


async def main():
    """Run all conversation processing tests."""
    print("🚀 CONVERSATION-LEVEL PROCESSING SYSTEM TEST")
    print("=" * 60)
    
    try:
        await demonstrate_architecture_improvement()
        await test_conversation_boundary_detection()
        await test_conversation_state_analysis() 
        await test_conversation_level_extraction()
        
        print("\n✅ CONVERSATION PROCESSING TESTS COMPLETED")
        print("=" * 60)
        print("""
NEXT STEPS:
1. Deploy the new system to see improved knowledge quality
2. Monitor conversation state detection accuracy
3. Adjust completion detection patterns based on real data
4. Implement threading context assembly (next priority)
        """)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Conversation processing test error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
