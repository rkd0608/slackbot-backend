#!/usr/bin/env python3
"""
Test script for Context Resolution System.

This demonstrates how the system now resolves contextual references:
- Pronouns ("it", "that", "this")
- Implicit references ("the process", "the issue") 
- Follow-up questions that require conversation context
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.database import get_session_factory
from app.services.context_resolver import ContextResolver
from app.models.base import Message, Conversation
from sqlalchemy import select, desc
from loguru import logger


async def test_reference_detection():
    """Test detection of contextual references in queries."""
    print("\n🔍 TESTING REFERENCE DETECTION")
    print("=" * 60)
    
    resolver = ContextResolver()
    
    test_queries = [
        "How many steps are provided for it?",
        "What is the process?",
        "Can you show me the configuration?",
        "How do I fix that issue?",
        "Where can I find those files?",
        "What was the decision about this?",
        "How long will it take?",
        "Who is responsible for the deployment?",
        "Normal query without references"
    ]
    
    for query in test_queries:
        references = resolver._detect_references(query)
        print(f"Query: '{query}'")
        print(f"  References detected: {references}")
        print()


async def test_context_resolution():
    """Test full context resolution with real conversation data."""
    print("\n🧠 TESTING CONTEXT RESOLUTION")
    print("=" * 60)
    
    session_factory = get_session_factory()
    async with session_factory() as db:
        resolver = ContextResolver()
        
        # Get a recent conversation to use for context
        conv_query = select(Conversation).order_by(desc(Conversation.created_at)).limit(1)
        conv_result = await db.execute(conv_query)
        conversation = conv_result.scalar_one_or_none()
        
        if not conversation:
            print("❌ No conversations found for context testing")
            return
        
        print(f"Using conversation {conversation.id} for context")
        
        # Test queries that need context resolution
        test_scenarios = [
            {
                "query": "How many steps are provided for it?",
                "description": "Pronoun resolution - 'it' should refer to recent topic"
            },
            {
                "query": "What is the process?",
                "description": "Implicit reference - 'the process' needs context"
            },
            {
                "query": "Can you explain that again?", 
                "description": "Demonstrative pronoun - 'that' needs resolution"
            },
            {
                "query": "How do I configure the system?",
                "description": "Implicit reference - 'the system' needs context"
            },
            {
                "query": "Who mentioned this earlier?",
                "description": "Temporal reference - 'this' + 'earlier'"
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n📝 Scenario: {scenario['description']}")
            print(f"Original query: '{scenario['query']}'")
            
            try:
                resolved_query, references = await resolver.resolve_query_context(
                    query=scenario['query'],
                    user_id="test_user",
                    channel_id=conversation.slack_channel_id,
                    workspace_id=conversation.workspace_id,
                    db=db
                )
                
                print(f"Resolved query: '{resolved_query}'")
                print(f"References resolved: {len(references)}")
                
                for ref in references:
                    print(f"  - '{ref.original_term}' → '{ref.resolved_entity}' (confidence: {ref.confidence:.2f})")
                
            except Exception as e:
                print(f"❌ Error: {e}")


async def test_conversation_context_building():
    """Test building conversation context from recent messages."""
    print("\n📚 TESTING CONVERSATION CONTEXT BUILDING")
    print("=" * 60)
    
    session_factory = get_session_factory()
    async with session_factory() as db:
        resolver = ContextResolver()
        
        # Get a conversation with messages
        conv_query = select(Conversation).order_by(desc(Conversation.created_at)).limit(1)
        conv_result = await db.execute(conv_query)
        conversation = conv_result.scalar_one_or_none()
        
        if not conversation:
            print("❌ No conversations found")
            return
        
        # Check if this conversation has messages
        msg_query = select(Message).where(Message.conversation_id == conversation.id).limit(1)
        msg_result = await db.execute(msg_query)
        has_messages = msg_result.scalar_one_or_none() is not None
        
        if not has_messages:
            print("❌ No messages found in conversation")
            return
        
        print(f"Building context for conversation {conversation.id}")
        
        try:
            context = await resolver._build_conversation_context(
                user_id="test_user",
                channel_id=conversation.slack_channel_id,
                workspace_id=conversation.workspace_id,
                db=db
            )
            
            print(f"✅ Context built successfully")
            print(f"Recent messages: {len(context.get('recent_messages', []))}")
            print(f"Recent entities: {context.get('recent_entities', {})}")
            print(f"Recent topics: {context.get('recent_topics', [])}")
            print(f"Main focus: {context.get('main_focus', 'None')}")
            print(f"User query history: {len(context.get('user_query_history', []))}")
            print(f"Knowledge items: {len(context.get('knowledge_items', []))}")
            
            if context.get('recent_messages'):
                print(f"\nSample recent messages:")
                for msg in context['recent_messages'][-3:]:
                    print(f"  {msg['user_id']}: {msg['content'][:50]}...")
            
        except Exception as e:
            print(f"❌ Error building context: {e}")
            logger.error(f"Context building error: {e}", exc_info=True)


async def demonstrate_before_after():
    """Demonstrate the improvement in query understanding."""
    print("\n🏗️  BEFORE vs AFTER DEMONSTRATION")
    print("=" * 60)
    
    print("""
PROBLEM SCENARIO:
User asks: "How many steps are provided for it?"

BEFORE (No Context Resolution):
❌ System searches for "how many steps are provided for it"
❌ No understanding of what "it" refers to
❌ Returns generic results about steps
❌ User gets frustrated with irrelevant answers

AFTER (With Context Resolution):
✅ System analyzes recent conversation context
✅ Identifies that "it" refers to "Kafka restart process" (from recent discussion)
✅ Resolves query to "How many steps are provided for Kafka restart process?"
✅ Searches with specific context
✅ Returns accurate, relevant information

TECHNICAL IMPROVEMENTS:
1. ✅ Pronoun Resolution: "it", "that", "this" → specific entities
2. ✅ Implicit Reference Resolution: "the process" → actual process name  
3. ✅ Conversation Context Assembly: Full narrative understanding
4. ✅ Recent Entity Tracking: Maintains focus on current topics
5. ✅ AI-Powered Resolution: Complex reference understanding
6. ✅ Confidence Scoring: Only resolves when confident
7. ✅ Query Validation: Ensures resolution preserves intent
    """)


async def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n⚠️  TESTING EDGE CASES")
    print("=" * 60)
    
    session_factory = get_session_factory()
    async with session_factory() as db:
        resolver = ContextResolver()
        
        edge_cases = [
            {
                "query": "What is the weather today?",
                "description": "Query without contextual references"
            },
            {
                "query": "it it it that that",
                "description": "Multiple ambiguous pronouns"
            },
            {
                "query": "",
                "description": "Empty query"
            },
            {
                "query": "How do I fix the the the process?",
                "description": "Malformed query with repeated articles"
            }
        ]
        
        for case in edge_cases:
            print(f"\n🧪 Edge case: {case['description']}")
            print(f"Query: '{case['query']}'")
            
            try:
                resolved_query, references = await resolver.resolve_query_context(
                    query=case['query'],
                    user_id="test_user", 
                    channel_id="test_channel",
                    workspace_id=1,
                    db=db
                )
                
                print(f"✅ Handled gracefully")
                print(f"Resolved: '{resolved_query}'")
                print(f"References: {len(references)}")
                
            except Exception as e:
                print(f"❌ Error: {e}")


async def main():
    """Run all context resolution tests."""
    print("🚀 CONTEXT RESOLUTION SYSTEM TEST")
    print("=" * 60)
    
    try:
        await demonstrate_before_after()
        await test_reference_detection()
        await test_conversation_context_building()
        await test_context_resolution()
        await test_edge_cases()
        
        print("\n✅ CONTEXT RESOLUTION TESTS COMPLETED")
        print("=" * 60)
        print("""
NEXT STEPS:
1. Deploy the context resolution system
2. Monitor resolution accuracy in production
3. Collect user feedback on improved query understanding
4. Implement process recognition improvements (next priority)
        """)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Context resolution test error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
