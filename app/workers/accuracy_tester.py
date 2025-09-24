"""
Automated accuracy testing worker for continuous quality monitoring.
Runs daily accuracy tests and alerts on quality degradation.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta
from loguru import logger

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .celery_app import celery_app
from ..services.accuracy_service import AccuracyService, AccuracyTestType
from ..services.slack_service import SlackService
from ..models.base import Workspace

def get_async_session():
    """Create a new async session for each task."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from ..core.config import settings
    
    engine = create_async_engine(settings.database_url)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return AsyncSessionLocal

@celery_app.task
def run_daily_accuracy_tests():
    """Run daily accuracy tests for all workspaces."""
    return asyncio.run(run_daily_accuracy_tests_async())

async def run_daily_accuracy_tests_async() -> Dict[str, Any]:
    """Async implementation of daily accuracy testing."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            logger.info("Starting daily accuracy tests...")
            
            # Get all active workspaces
            result = await db.execute(select(Workspace))
            workspaces = result.fetchall()
            
            accuracy_service = AccuracyService()
            test_results = {}
            alerts = []
            
            for workspace in workspaces:
                workspace_id = workspace.id
                workspace_name = workspace.name
                
                logger.info(f"Testing accuracy for workspace: {workspace_name}")
                
                try:
                    # Run comprehensive accuracy validation
                    validation_result = await accuracy_service.validate_production_readiness(workspace_id, db)
                    test_results[workspace_id] = validation_result
                    
                    # Check if accuracy has degraded
                    if not validation_result.get("is_production_ready", False):
                        alerts.append({
                            "workspace_id": workspace_id,
                            "workspace_name": workspace_name,
                            "readiness_score": validation_result.get("overall_readiness_score", 0.0),
                            "issues": validation_result.get("recommendations", [])
                        })
                        
                        logger.warning(f"Accuracy degraded for workspace {workspace_name}: {validation_result.get('overall_readiness_score', 0.0)}")
                    
                except Exception as e:
                    logger.error(f"Error testing workspace {workspace_name}: {e}")
                    alerts.append({
                        "workspace_id": workspace_id,
                        "workspace_name": workspace_name,
                        "error": str(e)
                    })
            
            # Send alerts if needed
            if alerts:
                await send_accuracy_alerts(alerts, db)
            
            logger.info(f"Completed daily accuracy tests for {len(workspaces)} workspaces")
            
            return {
                "status": "completed",
                "workspaces_tested": len(workspaces),
                "alerts_generated": len(alerts),
                "test_results": test_results,
                "tested_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in daily accuracy tests: {e}")
            return {
                "status": "error",
                "error": str(e),
                "tested_at": datetime.utcnow().isoformat()
            }

@celery_app.task
def run_accuracy_regression_test(workspace_id: int):
    """Run accuracy regression test for a specific workspace."""
    return asyncio.run(run_accuracy_regression_test_async(workspace_id))

async def run_accuracy_regression_test_async(workspace_id: int) -> Dict[str, Any]:
    """Async implementation of accuracy regression testing."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            logger.info(f"Running regression test for workspace {workspace_id}")
            
            accuracy_service = AccuracyService()
            
            # Run regression test
            result = await accuracy_service.run_accuracy_test(
                AccuracyTestType.REGRESSION, 
                workspace_id, 
                db
            )
            
            # Check if accuracy has regressed significantly
            if result.accuracy_score < accuracy_service.min_accuracy:
                await send_regression_alert(workspace_id, result, db)
            
            logger.info(f"Regression test completed for workspace {workspace_id}: {result.accuracy_score:.2f}")
            
            return {
                "status": "completed",
                "workspace_id": workspace_id,
                "accuracy_score": result.accuracy_score,
                "test_cases_passed": result.test_cases_passed,
                "test_cases_total": result.test_cases_total,
                "tested_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in regression test for workspace {workspace_id}: {e}")
            return {
                "status": "error",
                "workspace_id": workspace_id,
                "error": str(e),
                "tested_at": datetime.utcnow().isoformat()
            }

@celery_app.task
def run_hallucination_detection_test(workspace_id: int):
    """Run hallucination detection test for a specific workspace."""
    return asyncio.run(run_hallucination_detection_test_async(workspace_id))

async def run_hallucination_detection_test_async(workspace_id: int) -> Dict[str, Any]:
    """Async implementation of hallucination detection testing."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            logger.info(f"Running hallucination detection test for workspace {workspace_id}")
            
            accuracy_service = AccuracyService()
            
            # Run hallucination test
            result = await accuracy_service.run_accuracy_test(
                AccuracyTestType.HALLUCINATION, 
                workspace_id, 
                db
            )
            
            # Check if hallucination rate is too high
            if result.hallucination_rate > accuracy_service.max_hallucination_rate:
                await send_hallucination_alert(workspace_id, result, db)
            
            logger.info(f"Hallucination test completed for workspace {workspace_id}: {result.hallucination_rate:.2f}")
            
            return {
                "status": "completed",
                "workspace_id": workspace_id,
                "hallucination_rate": result.hallucination_rate,
                "test_cases_passed": result.test_cases_passed,
                "test_cases_total": result.test_cases_total,
                "tested_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in hallucination test for workspace {workspace_id}: {e}")
            return {
                "status": "error",
                "workspace_id": workspace_id,
                "error": str(e),
                "tested_at": datetime.utcnow().isoformat()
            }

async def send_accuracy_alerts(alerts: List[Dict[str, Any]], db: AsyncSession) -> None:
    """Send accuracy degradation alerts."""
    try:
        # For now, just log the alerts
        # In production, you might send emails, Slack notifications, etc.
        
        logger.warning(f"ACCURACY ALERT: {len(alerts)} workspaces have accuracy issues:")
        
        for alert in alerts:
            if "error" in alert:
                logger.error(f"  - {alert['workspace_name']}: {alert['error']}")
            else:
                logger.warning(f"  - {alert['workspace_name']}: Readiness score {alert['readiness_score']:.2f}")
                for issue in alert.get("issues", []):
                    logger.warning(f"    * {issue}")
        
        # TODO: Implement actual alerting (email, Slack, etc.)
        
    except Exception as e:
        logger.error(f"Error sending accuracy alerts: {e}")

async def send_regression_alert(workspace_id: int, result: Any, db: AsyncSession) -> None:
    """Send regression alert for a specific workspace."""
    try:
        # Get workspace info
        workspace_result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
        workspace = workspace_result.scalar_one_or_none()
        
        workspace_name = workspace.name if workspace else f"Workspace {workspace_id}"
        
        logger.error(f"REGRESSION ALERT: {workspace_name} accuracy dropped to {result.accuracy_score:.2f}")
        
        # TODO: Implement actual alerting
        
    except Exception as e:
        logger.error(f"Error sending regression alert: {e}")

async def send_hallucination_alert(workspace_id: int, result: Any, db: AsyncSession) -> None:
    """Send hallucination alert for a specific workspace."""
    try:
        # Get workspace info
        workspace_result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
        workspace = workspace_result.scalar_one_or_none()
        
        workspace_name = workspace.name if workspace else f"Workspace {workspace_id}"
        
        logger.error(f"HALLUCINATION ALERT: {workspace_name} hallucination rate at {result.hallucination_rate:.2f}")
        
        # TODO: Implement actual alerting
        
    except Exception as e:
        logger.error(f"Error sending hallucination alert: {e}")

@celery_app.task
def generate_accuracy_report(workspace_id: int, days_back: int = 30):
    """Generate comprehensive accuracy report for a workspace."""
    return asyncio.run(generate_accuracy_report_async(workspace_id, days_back))

async def generate_accuracy_report_async(workspace_id: int, days_back: int = 30) -> Dict[str, Any]:
    """Generate comprehensive accuracy report."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            logger.info(f"Generating accuracy report for workspace {workspace_id}")
            
            accuracy_service = AccuracyService()
            
            # Run all test types
            test_results = {}
            for test_type in AccuracyTestType:
                try:
                    result = await accuracy_service.run_accuracy_test(test_type, workspace_id, db)
                    test_results[test_type.value] = {
                        "accuracy_score": result.accuracy_score,
                        "precision": result.precision,
                        "recall": result.recall,
                        "hallucination_rate": result.hallucination_rate,
                        "test_cases_passed": result.test_cases_passed,
                        "test_cases_total": result.test_cases_total
                    }
                except Exception as e:
                    logger.error(f"Error running {test_type.value} test: {e}")
                    test_results[test_type.value] = {"error": str(e)}
            
            # Get overall readiness
            validation_result = await accuracy_service.validate_production_readiness(workspace_id, db)
            
            # Generate recommendations
            recommendations = validation_result.get("recommendations", [])
            
            report = {
                "workspace_id": workspace_id,
                "report_period_days": days_back,
                "test_results": test_results,
                "overall_readiness": validation_result.get("overall_readiness_score", 0.0),
                "is_production_ready": validation_result.get("is_production_ready", False),
                "recommendations": recommendations,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Accuracy report generated for workspace {workspace_id}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating accuracy report: {e}")
            return {
                "workspace_id": workspace_id,
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat()
            }
