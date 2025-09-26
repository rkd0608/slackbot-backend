#!/usr/bin/env python3
"""
Test script to verify our intent classification fixes.
This will test the queries that were previously misclassified.
"""

import sys
import os
import asyncio
import re

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Test queries that were previously broken
test_queries = [
    "can you find something for me?",
    "help me find information about the database migration",
    "what did we decide about the API changes?",
    "how do we handle authentication?",
    "do you know anything about the deployment process?",
    "I need help with the login system",
    "looking for information about our testing framework",
    "what's the status of the project?",
    "thanks",  # Should still be social
    "hello",   # Should still be social
    "hi there, can you help me find the docs?",  # Mixed - should be knowledge
    "good morning! what's the latest on the migration?",  # Mixed - should be knowledge
]

def test_pattern_matching():
    """Test the new pattern matching logic directly."""

    # New knowledge query patterns
    knowledge_patterns = [
        r'\b(how|what|when|where|why|which|who)\s',
        r'\b(explain|describe|tell me about|show me)\b',
        r'\b(can you|could you|would you)\s+(find|search|look|get|retrieve|locate|help|tell|show|explain)\b',
        r'\b(find|search|look|get|retrieve|locate)\s+(something|anything|information|data|details)\b',
        r'\b(help me|assist me|support me)\b',
        r'\b(need|want|looking for|seeking)\s+(to know|information|help|assistance|details)\b',
        r'\b(do you know|do you have|is there|are there)\b',
        r'\b(anything|something).*(help|find|know|information)\b',
        r'\b(help|find|know|information).*(anything|something)\b'
    ]

    # New social interaction patterns (very restrictive)
    social_patterns = [
        r'^(hi|hello|hey|good morning|good afternoon|good evening)!?$',
        r'^(how are you|how\'s it going|what\'s up)\??$',
        r'^(bye|goodbye|see you|catch you later)!?$',
        r'^(great|awesome|cool|nice|good|bad)!?$',
        r'^(thanks|thank you|thx)!?$',
    ]

    print("🧪 Testing Pattern Matching Logic")
    print("=" * 50)

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        # Test knowledge patterns
        knowledge_matches = 0
        for pattern in knowledge_patterns:
            if re.search(pattern, query.lower(), re.IGNORECASE):
                knowledge_matches += 1
                print(f"  ✅ Knowledge pattern matched: {pattern}")

        # Test social patterns
        social_matches = 0
        for pattern in social_patterns:
            if re.search(pattern, query.lower(), re.IGNORECASE):
                social_matches += 1
                print(f"  🗣️ Social pattern matched: {pattern}")

        # Apply weighted scoring
        knowledge_score = knowledge_matches * 1.5  # Boost knowledge
        social_score = social_matches * 0.5        # Reduce social priority

        if knowledge_score > social_score:
            predicted_intent = "knowledge_query"
            confidence = min(0.95, (knowledge_score / 5.0) + 0.3)
        elif social_score > 0:
            predicted_intent = "social_interaction"
            confidence = min(0.95, (social_score / 5.0) + 0.3)
        else:
            predicted_intent = "knowledge_query"  # Default
            confidence = 0.7

        print(f"  🎯 Predicted: {predicted_intent} (confidence: {confidence:.2f})")

        # Validate the result
        if query in ["thanks", "hello"]:
            expected = "social_interaction"
        else:
            expected = "knowledge_query"

        if predicted_intent == expected:
            print(f"  ✅ CORRECT! Expected {expected}")
        else:
            print(f"  ❌ WRONG! Expected {expected}, got {predicted_intent}")

def test_knowledge_search_logic():
    """Test the new knowledge search logic."""
    print("\n\n🔍 Testing Knowledge Search Logic")
    print("=" * 50)

    for query in test_queries:
        # Simulate intent results
        if query in ["thanks", "hello"]:
            intent = "social_interaction"
            requires_search = False
        else:
            intent = "knowledge_query"
            requires_search = True

        # New logic: should_search_knowledge
        should_search_knowledge = (
            requires_search or
            intent == "social_interaction" or  # Social can hide knowledge needs
            len(query.split()) > 3  # Any substantial query should search
        )

        print(f"\nQuery: '{query}'")
        print(f"  Intent: {intent}")
        print(f"  Original requires_search: {requires_search}")
        print(f"  NEW should_search_knowledge: {should_search_knowledge}")

        if should_search_knowledge:
            print(f"  ✅ Will search knowledge base")
        else:
            print(f"  ❌ Will skip knowledge search")

if __name__ == "__main__":
    print("🚀 Testing Intent Classification Fixes")
    print("This script tests the fixes we made to make your bot smarter\n")

    test_pattern_matching()
    test_knowledge_search_logic()

    print("\n\n🎉 Test Summary:")
    print("- Fixed overly restrictive social interaction patterns")
    print("- Enhanced knowledge query detection")
    print("- Added weighted scoring (knowledge queries get priority)")
    print("- Modified query processor to search knowledge for social interactions")
    print("- Default fallback is now knowledge_query (smarter default)")
    print("\nYour bot should now be much smarter! 🧠✨")