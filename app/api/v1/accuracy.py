"""
API endpoints for AI accuracy testing and validation.
Provides automated testing capabilities for production readiness.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

from ...core.database import get_db
from ...services.accuracy_service import AccuracyService, AccuracyTestType
from ...models.base import Workspace

router = APIRouter()

@router.post("/test/{test_type}")
async def run_accuracy_test(
    test_type: str,
    workspace_id: int = Query(..., description="Workspace ID to test"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Run a specific accuracy test type."""
    try:
        # Validate test type
        try:
            test_enum = AccuracyTestType(test_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid test type. Must be one of: {[t.value for t in AccuracyTestType]}"
            )
        
        # Validate workspace exists
        workspace = await db.get(Workspace, workspace_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Run the test
        accuracy_service = AccuracyService()
        result = await accuracy_service.run_accuracy_test(test_enum, workspace_id, db)
        
        return {
            "test_type": result.test_type.value,
            "accuracy_score": result.accuracy_score,
            "precision": result.precision,
            "recall": result.recall,
            "hallucination_rate": result.hallucination_rate,
            "confidence_calibration": result.confidence_calibration,
            "test_cases_passed": result.test_cases_passed,
            "test_cases_total": result.test_cases_total,
            "details": result.details,
            "tested_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running accuracy test: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/validate-production-readiness")
async def validate_production_readiness(
    workspace_id: int = Query(..., description="Workspace ID to validate"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Comprehensive production readiness validation."""
    try:
        # Validate workspace exists
        workspace = await db.get(Workspace, workspace_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Run comprehensive validation
        accuracy_service = AccuracyService()
        validation_result = await accuracy_service.validate_production_readiness(workspace_id, db)
        
        return validation_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating production readiness: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/generate-ground-truth")
async def generate_ground_truth_dataset(
    workspace_id: int = Query(..., description="Workspace ID to generate ground truth for"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Generate ground truth dataset from existing conversations."""
    try:
        # Validate workspace exists
        workspace = await db.get(Workspace, workspace_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        # Generate ground truth dataset
        accuracy_service = AccuracyService()
        ground_truth_cases = await accuracy_service.generate_ground_truth_dataset(workspace_id, db)
        
        return {
            "workspace_id": workspace_id,
            "ground_truth_cases_generated": len(ground_truth_cases),
            "cases": ground_truth_cases[:5],  # Return first 5 as examples
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating ground truth dataset: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/generate-synthetic-tests")
async def generate_synthetic_test_cases(
    count: int = Query(100, description="Number of synthetic test cases to generate"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Generate synthetic test cases with known correct answers."""
    try:
        if count > 500:
            raise HTTPException(status_code=400, detail="Maximum 500 synthetic cases per request")
        
        # Generate synthetic test cases
        accuracy_service = AccuracyService()
        synthetic_cases = await accuracy_service.generate_synthetic_test_cases(count)
        
        return {
            "synthetic_cases_generated": len(synthetic_cases),
            "requested_count": count,
            "cases": synthetic_cases[:3],  # Return first 3 as examples
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating synthetic test cases: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/test-types")
async def get_available_test_types() -> Dict[str, Any]:
    """Get list of available accuracy test types."""
    return {
        "test_types": [
            {
                "name": test_type.value,
                "description": _get_test_description(test_type)
            }
            for test_type in AccuracyTestType
        ]
    }

@router.get("/production-criteria")
async def get_production_criteria() -> Dict[str, Any]:
    """Get production readiness criteria and thresholds."""
    accuracy_service = AccuracyService()
    return {
        "minimum_accuracy": accuracy_service.min_accuracy,
        "maximum_hallucination_rate": accuracy_service.max_hallucination_rate,
        "minimum_confidence_calibration": accuracy_service.min_confidence_calibration,
        "criteria": [
            "85%+ accuracy on ground truth dataset",
            "< 5% hallucination rate",
            "80%+ confidence calibration",
            "Consistent results across test runs",
            "Proper error handling and fallbacks"
        ]
    }

def _get_test_description(test_type: AccuracyTestType) -> str:
    """Get human-readable description for test type."""
    descriptions = {
        AccuracyTestType.GROUND_TRUTH: "Test against known correct answers from real conversations",
        AccuracyTestType.SYNTHETIC: "Test against AI-generated conversations with known outcomes", 
        AccuracyTestType.HALLUCINATION: "Test AI's tendency to fabricate information not in source",
        AccuracyTestType.CONSISTENCY: "Test consistency of AI responses across multiple runs",
        AccuracyTestType.REGRESSION: "Test for accuracy degradation over time"
    }
    return descriptions.get(test_type, "Unknown test type")
