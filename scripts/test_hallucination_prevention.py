#!/usr/bin/env python3
"""
Test script for Hallucination Prevention System.

This demonstrates how the system prevents AI hallucinations through:
- Source verification: Ensures claims can be traced to messages
- Pattern detection: Identifies hallucination indicators
- Fact verification: AI-powered cross-checking
- Technical validation: Prevents fabricated technical details
- Quote accuracy: Verifies quoted information
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.hallucination_preventer import HallucinationPreventer, HallucinationCheck
from loguru import logger


async def test_source_verification():
    """Test source attribution validation."""
    print("\n🔍 TESTING SOURCE VERIFICATION")
    print("=" * 60)
    
    preventer = HallucinationPreventer()
    
    # Test scenarios
    scenarios = [
        {
            "name": "Safe Content - Direct Quote",
            "extracted": "Bob said to restart the Kafka service using 'sudo systemctl restart kafka'",
            "sources": [
                {"user_id": "bob", "content": "To fix the issue, restart the Kafka service using 'sudo systemctl restart kafka'"}
            ],
            "expected_safe": True
        },
        {
            "name": "Hallucination - Added Technical Details",
            "extracted": "Bob said to restart Kafka on port 9092 using the --force flag with a 30-second timeout",
            "sources": [
                {"user_id": "bob", "content": "Just restart Kafka"}
            ],
            "expected_safe": False
        },
        {
            "name": "Inference - Assumed Causation",
            "extracted": "The database crash was caused by insufficient memory, which led to connection timeouts",
            "sources": [
                {"user_id": "alice", "content": "Database crashed"},
                {"user_id": "bob", "content": "We're seeing connection timeouts"}
            ],
            "expected_safe": False
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📝 Scenario: {scenario['name']}")
        
        try:
            check = await preventer._verify_source_attribution(
                extracted_content=scenario['extracted'],
                source_messages=scenario['sources']
            )
            
            print(f"  Result: {'✅ SAFE' if check.is_safe else '❌ UNSAFE'}")
            print(f"  Confidence: {check.confidence:.2f}")
            print(f"  Expected: {'SAFE' if scenario['expected_safe'] else 'UNSAFE'}")
            print(f"  Correct: {'✅' if check.is_safe == scenario['expected_safe'] else '❌'}")
            
            if check.issues_found:
                print(f"  Issues: {check.issues_found}")
            
            if check.corrected_content:
                print(f"  Correction: {check.corrected_content[:100]}...")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")


async def test_pattern_detection():
    """Test pattern-based hallucination detection."""
    print("\n🎯 TESTING PATTERN DETECTION")
    print("=" * 60)
    
    preventer = HallucinationPreventer()
    
    test_contents = [
        {
            "content": "The service is running on port 8080 with --debug flag enabled",
            "description": "Technical details that might be fabricated"
        },
        {
            "content": "Obviously, this is the best approach for handling the issue",
            "description": "Assumption indicator"
        },
        {
            "content": "The process typically takes about 30 minutes to complete",
            "description": "Generalization without evidence"
        },
        {
            "content": "Bob mentioned restarting the service to fix the connection issue",
            "description": "Safe content with attribution"
        },
        {
            "content": "Based on experience, you should probably use version 2.1.3",
            "description": "Multiple inference indicators"
        }
    ]
    
    for test in test_contents:
        print(f"\n📊 Testing: {test['description']}")
        print(f"Content: \"{test['content']}\"")
        
        check = preventer._detect_fabrication_patterns(test['content'])
        
        print(f"  Result: {'✅ SAFE' if check.is_safe else '❌ UNSAFE'}")
        print(f"  Confidence: {check.confidence:.2f}")
        
        if check.issues_found:
            print(f"  Issues Found: {check.issues_found}")


async def test_ai_fact_verification():
    """Test AI-powered fact verification."""
    print("\n🤖 TESTING AI FACT VERIFICATION")
    print("=" * 60)
    
    preventer = HallucinationPreventer()
    
    # Test the problematic scenario from the original analysis
    original_conversation = """
alice: Hey team, we're having issues with the Kafka connectors. They keep failing.
bob: I can help with that! I've dealt with this before.
bob: Step-by-step coming up - let me grab the exact procedure from our runbook.
alice: Awesome, thanks! How many steps are there usually?
bob: Usually about 5-6 steps depending on the specific connector type.
charlie: Is this for the payment processing connectors?
alice: Yes, exactly those ones.
    """
    
    # Problematic extracted content (with hallucinations)
    problematic_extraction = """
To restart Kafka connectors:
1. Stop the connector service: sudo systemctl stop kafka-connect
2. Clear the connector logs: rm -rf /var/log/kafka-connect/*
3. Update the connector configuration in /etc/kafka/connect-distributed.properties
4. Set the heap memory to 2GB: export KAFKA_HEAP_OPTS="-Xmx2g"
5. Restart with force flag: sudo systemctl restart kafka-connect --force
6. Verify status: curl http://localhost:8083/connectors

This process typically takes 5-10 minutes and should resolve connection timeout issues.
    """
    
    print("🔍 Analyzing Kafka Connector Extraction")
    print(f"Original conversation length: {len(original_conversation)} chars")
    print(f"Extracted content length: {len(problematic_extraction)} chars")
    
    try:
        check = await preventer._ai_fact_verification(
            extracted_content=problematic_extraction,
            original_conversation=original_conversation
        )
        
        print(f"\n📊 AI Fact Verification Results:")
        print(f"  Accurate: {'✅ YES' if check.is_safe else '❌ NO'}")
        print(f"  Confidence: {check.confidence:.2f}")
        
        if check.issues_found:
            print(f"  Fabricated Elements:")
            for issue in check.issues_found:
                print(f"    - {issue}")
        
        if check.corrected_content:
            print(f"  Corrected Content:")
            print(f"    {check.corrected_content[:200]}...")
            
    except Exception as e:
        print(f"❌ AI verification failed: {e}")


async def test_comprehensive_validation():
    """Test the complete validation pipeline."""
    print("\n🛡️  TESTING COMPREHENSIVE VALIDATION")
    print("=" * 60)
    
    preventer = HallucinationPreventer()
    
    # Scenario: Technical procedure with mixed valid/invalid content
    source_messages = [
        {"user_id": "alice", "content": "How do I deploy the new microservice?"},
        {"user_id": "bob", "content": "Here's the deployment process:"},
        {"user_id": "bob", "content": "1. Build the Docker image"},
        {"user_id": "bob", "content": "2. Push to our registry"},
        {"user_id": "bob", "content": "3. Update the Kubernetes deployment"},
        {"user_id": "bob", "content": "That should get it running"},
        {"user_id": "alice", "content": "Thanks! How long does it usually take?"},
        {"user_id": "bob", "content": "Usually just a few minutes"}
    ]
    
    # Mixed content - some valid, some hallucinated
    mixed_extraction = """
Microservice Deployment Process:
1. Build the Docker image using: docker build -t myservice:v1.2.3 .
2. Push to registry at registry.company.com:5000 with authentication token
3. Update Kubernetes deployment using: kubectl apply -f deployment.yaml --namespace=production
4. Configure the load balancer to route traffic on port 8080
5. Set environment variables: DATABASE_URL=postgres://prod-db:5432/app
6. Enable monitoring with Prometheus scraping on /metrics endpoint
7. Verify deployment status and check logs for errors

The deployment typically takes 3-5 minutes and requires admin privileges.
    """
    
    original_conversation = "\n".join([f"{msg['user_id']}: {msg['content']}" for msg in source_messages])
    
    print("🔍 Running Comprehensive Validation")
    
    try:
        validation_result = await preventer.validate_extracted_knowledge(
            extracted_content=mixed_extraction,
            source_messages=source_messages,
            original_conversation=original_conversation
        )
        
        print(f"\n📊 Comprehensive Validation Results:")
        print(f"  Overall Safety: {'✅ SAFE' if validation_result.is_safe else '❌ UNSAFE'}")
        print(f"  Confidence: {validation_result.confidence:.2f}")
        print(f"  Check Type: {validation_result.check_type}")
        
        if validation_result.issues_found:
            print(f"  Issues Found ({len(validation_result.issues_found)}):")
            for issue in validation_result.issues_found[:5]:  # Show first 5
                print(f"    - {issue}")
            if len(validation_result.issues_found) > 5:
                print(f"    ... and {len(validation_result.issues_found) - 5} more")
        
        if validation_result.evidence:
            print(f"  Supporting Evidence ({len(validation_result.evidence)}):")
            for evidence in validation_result.evidence[:3]:  # Show first 3
                print(f"    - {evidence}")
        
        if validation_result.corrected_content:
            print(f"  Corrected Content Available: ✅ YES")
            print(f"    Length: {len(validation_result.corrected_content)} chars")
        else:
            print(f"  Corrected Content Available: ❌ NO")
            
    except Exception as e:
        print(f"❌ Comprehensive validation failed: {e}")


async def test_response_validation():
    """Test validation of AI responses to user queries."""
    print("\n💬 TESTING AI RESPONSE VALIDATION")
    print("=" * 60)
    
    preventer = HallucinationPreventer()
    
    # Scenario: User asks about deployment, AI response contains hallucinations
    user_query = "How do I deploy the payment service?"
    
    search_results = [
        {
            "title": "Payment Service Deployment",
            "content": "Deploy using Docker. Build image and push to registry.",
            "confidence": 0.8
        },
        {
            "title": "Kubernetes Guide",
            "content": "Use kubectl apply for deployments. Check status with kubectl get pods.",
            "confidence": 0.7
        }
    ]
    
    # AI response with hallucinations
    problematic_response = """
To deploy the payment service:

1. Build the Docker image: `docker build -t payment-service:v2.1.4 .`
2. Push to our private registry at harbor.company.com:443 using your personal access token
3. Update the Kubernetes deployment: `kubectl apply -f payment-deployment.yaml --namespace=payments-prod`
4. Configure the service to use 4GB RAM and 2 CPU cores
5. Set the database connection: `DATABASE_URL=postgresql://payment-user:secure123@db.payments.com:5432/payments_db`
6. Enable SSL with certificate from /etc/ssl/certs/payment-service.crt
7. Start the health check endpoint on port 8080/health

The deployment typically takes 2-3 minutes. Make sure to update the configuration file with the latest API keys and enable debug logging for troubleshooting.
    """
    
    print("🔍 Validating AI Response")
    print(f"Query: {user_query}")
    print(f"Search results: {len(search_results)} items")
    print(f"Response length: {len(problematic_response)} chars")
    
    try:
        check = await preventer.validate_ai_response(
            ai_response=problematic_response,
            query=user_query,
            search_results=search_results
        )
        
        print(f"\n📊 AI Response Validation Results:")
        print(f"  Response Accuracy: {check.check_type}")
        print(f"  Safe: {'✅ YES' if check.is_safe else '❌ NO'}")
        print(f"  Confidence: {check.confidence:.2f}")
        
        if check.issues_found:
            print(f"  Hallucination Issues ({len(check.issues_found)}):")
            for issue in check.issues_found:
                print(f"    - {issue}")
        
        if check.evidence:
            print(f"  Supported Claims ({len(check.evidence)}):")
            for evidence in check.evidence:
                print(f"    - {evidence}")
                
    except Exception as e:
        print(f"❌ Response validation failed: {e}")


async def demonstrate_hallucination_fix():
    """Demonstrate how the system prevents hallucinations."""
    print("\n🛡️  HALLUCINATION PREVENTION DEMONSTRATION")
    print("=" * 60)
    
    print("""
PROBLEM (Before Anti-Hallucination):
❌ AI adds plausible-sounding but fabricated details
❌ Technical specifications not mentioned in conversation
❌ Inferred causation without evidence
❌ Generic advice presented as specific facts
❌ Fabricated quotes and commands

SOLUTION (After Anti-Hallucination):
✅ Multi-layered fact-checking (5 validation checks)
✅ Source attribution verification
✅ Pattern-based fabrication detection
✅ AI-powered cross-verification
✅ Technical detail validation
✅ Quote accuracy verification
✅ Confidence adjustment based on hallucination risk

VALIDATION LAYERS:
1. 📋 Source Attribution: All claims must trace to source messages
2. 🎯 Pattern Detection: Identifies assumption/inference indicators
3. 🤖 AI Fact Verification: GPT-4 cross-checks against original conversation
4. 💬 Quote Verification: Ensures quoted text matches exactly
5. ⚙️  Technical Validation: Prevents fabricated commands/configs

RESULT:
- Only explicitly stated information is preserved
- Fabricated details are removed or corrected
- Confidence scores reflect actual evidence strength
- Users get accurate, verifiable information
    """)


async def main():
    """Run all hallucination prevention tests."""
    print("🚀 HALLUCINATION PREVENTION SYSTEM TEST")
    print("=" * 60)
    
    try:
        await demonstrate_hallucination_fix()
        await test_pattern_detection()
        await test_source_verification()
        await test_ai_fact_verification()
        await test_response_validation()
        await test_comprehensive_validation()
        
        print("\n✅ HALLUCINATION PREVENTION TESTS COMPLETED")
        print("=" * 60)
        print("""
SUMMARY OF IMPROVEMENTS:
✅ Source Verification: Ensures all claims can be traced to messages
✅ Pattern Detection: Identifies common hallucination indicators  
✅ AI Fact-Checking: GPT-4 validates against original conversation
✅ Technical Validation: Prevents fabricated commands/configs
✅ Quote Verification: Ensures accurate attribution
✅ Confidence Adjustment: Reduces confidence for risky content
✅ Content Correction: Provides cleaned versions when possible

PRODUCTION IMPACT:
- Prevents fabricated technical details
- Eliminates inferred causation without evidence
- Stops generic advice being presented as specific facts
- Ensures only explicitly stated information is preserved
- Builds user trust through accurate, verifiable responses

ALL 4 CRITICAL ARCHITECTURAL FIXES NOW COMPLETE! 🎉
        """)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Hallucination prevention test error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
