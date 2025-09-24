#!/usr/bin/env python3
"""
CLI tool for running accuracy tests and validating production readiness.
Usage: python scripts/test_accuracy.py [command] [options]
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.accuracy_service import AccuracyService, AccuracyTestType
from app.services.feedback_service import FeedbackService
from app.core.database import get_session_factory
from app.models.base import Workspace
from sqlalchemy import select

async def run_single_test(test_type: str, workspace_id: int):
    """Run a single accuracy test."""
    try:
        test_enum = AccuracyTestType(test_type)
        accuracy_service = AccuracyService()
        
        async_session = get_session_factory()
        async with async_session() as db:
            result = await accuracy_service.run_accuracy_test(test_enum, workspace_id, db)
            
            print(f"\n🧪 {test_type.upper()} TEST RESULTS")
            print("=" * 50)
            print(f"Accuracy Score: {result.accuracy_score:.2%}")
            print(f"Precision: {result.precision:.2%}")
            print(f"Recall: {result.recall:.2%}")
            print(f"Hallucination Rate: {result.hallucination_rate:.2%}")
            print(f"Confidence Calibration: {result.confidence_calibration:.2%}")
            print(f"Test Cases: {result.test_cases_passed}/{result.test_cases_total}")
            
            if result.details:
                print(f"Details: {json.dumps(result.details, indent=2)}")
            
            # Determine if test passes
            accuracy_service = AccuracyService()
            passes = (
                result.accuracy_score >= accuracy_service.min_accuracy and
                result.hallucination_rate <= accuracy_service.max_hallucination_rate
            )
            
            status = "✅ PASS" if passes else "❌ FAIL"
            print(f"\nStatus: {status}")
            
            return passes
            
    except ValueError as e:
        print(f"❌ Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

async def validate_production_readiness(workspace_id: int):
    """Run comprehensive production readiness validation."""
    try:
        accuracy_service = AccuracyService()
        
        async_session = get_session_factory()
        async with async_session() as db:
            # Get workspace info
            result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
            workspace = result.scalar_one_or_none()
            
            if not workspace:
                print(f"❌ Workspace {workspace_id} not found")
                return False
            
            print(f"\n🚀 PRODUCTION READINESS VALIDATION")
            print(f"Workspace: {workspace.name} (ID: {workspace_id})")
            print("=" * 60)
            
            # Run comprehensive validation
            validation_result = await accuracy_service.validate_production_readiness(workspace_id, db)
            
            # Display results
            overall_score = validation_result.get("overall_readiness_score", 0.0)
            is_ready = validation_result.get("is_production_ready", False)
            
            print(f"Overall Readiness Score: {overall_score:.2%}")
            print(f"Production Ready: {'✅ YES' if is_ready else '❌ NO'}")
            
            # Show individual test results
            test_results = validation_result.get("test_results", {})
            print(f"\n📊 INDIVIDUAL TEST RESULTS:")
            for test_type, result in test_results.items():
                if hasattr(result, 'accuracy_score'):
                    score = result.accuracy_score
                    status = "✅" if score >= 0.85 else "❌"
                    print(f"  {status} {test_type}: {score:.2%}")
                else:
                    print(f"  ❓ {test_type}: Error or incomplete")
            
            # Show recommendations
            recommendations = validation_result.get("recommendations", [])
            if recommendations:
                print(f"\n💡 RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
            
            return is_ready
            
    except Exception as e:
        print(f"❌ Error in production validation: {e}")
        return False

async def generate_ground_truth(workspace_id: int, count: int = 50):
    """Generate ground truth dataset."""
    try:
        accuracy_service = AccuracyService()
        
        async_session = get_session_factory()
        async with async_session() as db:
            print(f"\n📊 GENERATING GROUND TRUTH DATASET")
            print(f"Workspace ID: {workspace_id}")
            print("=" * 50)
            
            ground_truth_cases = await accuracy_service.generate_ground_truth_dataset(workspace_id, db)
            
            print(f"Generated: {len(ground_truth_cases)} ground truth cases")
            
            if ground_truth_cases:
                print(f"\n📝 SAMPLE CASES:")
                for i, case in enumerate(ground_truth_cases[:3], 1):
                    gt = case.get("ground_truth", {})
                    print(f"  {i}. Decision: {gt.get('decision', 'N/A')}")
                    print(f"     Confidence: {gt.get('confidence', 0)}/10")
            
            return len(ground_truth_cases) > 0
            
    except Exception as e:
        print(f"❌ Error generating ground truth: {e}")
        return False

async def generate_synthetic_tests(count: int = 100):
    """Generate synthetic test cases."""
    try:
        accuracy_service = AccuracyService()
        
        print(f"\n🧪 GENERATING SYNTHETIC TEST CASES")
        print(f"Count: {count}")
        print("=" * 50)
        
        synthetic_cases = await accuracy_service.generate_synthetic_test_cases(count)
        
        print(f"Generated: {len(synthetic_cases)} synthetic test cases")
        
        if synthetic_cases:
            print(f"\n📝 SAMPLE CASES:")
            for i, case in enumerate(synthetic_cases[:2], 1):
                gt = case.get("ground_truth", {})
                print(f"  {i}. Type: {case.get('type', 'N/A')}")
                print(f"     Decision: {gt.get('decision', 'N/A')}")
        
        return len(synthetic_cases) > 0
        
    except Exception as e:
        print(f"❌ Error generating synthetic tests: {e}")
        return False

async def get_feedback_insights(workspace_id: int):
    """Get feedback insights for a workspace."""
    try:
        feedback_service = FeedbackService()
        
        async_session = get_session_factory()
        async with async_session() as db:
            print(f"\n📈 FEEDBACK INSIGHTS")
            print(f"Workspace ID: {workspace_id}")
            print("=" * 50)
            
            insights = await feedback_service.get_feedback_insights(workspace_id, db)
            
            # Display statistics
            stats = insights.get("feedback_statistics", {})
            print(f"Total Feedback: {stats.get('total_feedback', 0)}")
            print(f"Average Rating: {stats.get('average_rating', 0):.2f}/5")
            print(f"Satisfaction Rate: {stats.get('satisfaction_rate', 0):.2%}")
            
            # Display feedback by type
            feedback_by_type = stats.get("feedback_by_type", {})
            if feedback_by_type:
                print(f"\n📊 FEEDBACK BY TYPE:")
                for ftype, count in feedback_by_type.items():
                    print(f"  {ftype}: {count}")
            
            # Display problematic items
            problematic = insights.get("problematic_knowledge_items", [])
            if problematic:
                print(f"\n⚠️  PROBLEMATIC KNOWLEDGE ITEMS:")
                for item in problematic[:5]:  # Show top 5
                    print(f"  - {item.get('title', 'N/A')} (Score: {item.get('quality_score', 0):.2f})")
            
            # Display recommendations
            recommendations = insights.get("improvement_recommendations", [])
            if recommendations:
                print(f"\n💡 RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error getting feedback insights: {e}")
        return False

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="AI Accuracy Testing CLI")
    parser.add_argument("command", choices=[
        "test", "validate", "ground-truth", "synthetic", "feedback"
    ], help="Command to run")
    
    parser.add_argument("--workspace-id", type=int, default=1, 
                       help="Workspace ID to test (default: 1)")
    parser.add_argument("--test-type", choices=[t.value for t in AccuracyTestType],
                       help="Specific test type to run")
    parser.add_argument("--count", type=int, default=100,
                       help="Number of items to generate (default: 100)")
    
    args = parser.parse_args()
    
    if args.command == "test":
        if not args.test_type:
            print("❌ --test-type is required for 'test' command")
            sys.exit(1)
        success = asyncio.run(run_single_test(args.test_type, args.workspace_id))
    elif args.command == "validate":
        success = asyncio.run(validate_production_readiness(args.workspace_id))
    elif args.command == "ground-truth":
        success = asyncio.run(generate_ground_truth(args.workspace_id, args.count))
    elif args.command == "synthetic":
        success = asyncio.run(generate_synthetic_tests(args.count))
    elif args.command == "feedback":
        success = asyncio.run(get_feedback_insights(args.workspace_id))
    else:
        print(f"❌ Unknown command: {args.command}")
        sys.exit(1)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
