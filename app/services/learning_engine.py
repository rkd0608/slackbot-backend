"""Basic learning engine for continuous improvement of intent classification."""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc, update

from ..models.intent_classification import IntentClassificationHistory, ResponseEffectiveness, UserCommunicationProfile


@dataclass
class LearningMetrics:
    """Learning metrics for system improvement."""
    total_interactions: int
    classification_accuracy: float
    user_satisfaction_avg: float
    response_effectiveness_avg: float
    learning_confidence: float
    improvement_trend: str
    top_performing_intents: List[str]
    areas_for_improvement: List[str]


@dataclass
class FeedbackData:
    """Feedback data for learning."""
    classification_id: int
    user_id: str
    workspace_id: int
    original_intent: str
    confidence_score: float
    user_rating: Optional[float]
    was_helpful: Optional[bool]
    correction_applied: bool
    improvement_suggestions: Optional[str]
    timestamp: datetime


class LearningEngine:
    """Basic learning engine for continuous improvement."""
    
    def __init__(self):
        self.logger = logger
        self.learning_rate = 0.1
        self.min_samples_for_learning = 10
    
    async def process_feedback(
        self,
        db: AsyncSession,
        feedback_data: FeedbackData
    ) -> bool:
        """Process user feedback for learning."""
        try:
            # Update classification history with feedback
            await self._update_classification_with_feedback(db, feedback_data)
            
            # Update user profile based on feedback
            await self._update_user_profile_from_feedback(db, feedback_data)
            
            # Update learning metrics
            await self._update_learning_metrics(db, feedback_data)
            
            # Check if we need to trigger model updates
            await self._check_model_update_triggers(db, feedback_data.workspace_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing feedback: {e}")
            return False
    
    async def _update_classification_with_feedback(
        self,
        db: AsyncSession,
        feedback_data: FeedbackData
    ) -> None:
        """Update classification history with user feedback."""
        try:
            # Update the classification record
            await db.execute(
                update(IntentClassificationHistory)
                .where(IntentClassificationHistory.id == feedback_data.classification_id)
                .values(
                    user_satisfaction=feedback_data.user_rating,
                    was_correct=feedback_data.was_helpful,
                    correction_applied=feedback_data.correction_applied,
                    learning_notes=feedback_data.improvement_suggestions
                )
            )
            
            await db.commit()
            
        except Exception as e:
            self.logger.error(f"Error updating classification with feedback: {e}")
            await db.rollback()
    
    async def _update_user_profile_from_feedback(
        self,
        db: AsyncSession,
        feedback_data: FeedbackData
    ) -> None:
        """Update user profile based on feedback."""
        try:
            # Get user profile
            result = await db.execute(
                select(UserCommunicationProfile)
                .where(
                    and_(
                        UserCommunicationProfile.user_id == feedback_data.user_id,
                        UserCommunicationProfile.workspace_id == feedback_data.workspace_id
                    )
                )
            )
            profile = result.scalar_one_or_none()
            
            if not profile:
                return
            
            # Update learning confidence based on feedback
            if feedback_data.was_helpful is not None:
                if feedback_data.was_helpful:
                    # Positive feedback increases confidence
                    profile.learning_confidence = min(1.0, profile.learning_confidence + 0.05)
                else:
                    # Negative feedback decreases confidence
                    profile.learning_confidence = max(0.0, profile.learning_confidence - 0.1)
            
            # Update interaction count
            profile.interaction_count += 1
            
            # Update last updated timestamp
            profile.last_updated = datetime.utcnow()
            
            await db.commit()
            
        except Exception as e:
            self.logger.error(f"Error updating user profile from feedback: {e}")
            await db.rollback()
    
    async def _update_learning_metrics(
        self,
        db: AsyncSession,
        feedback_data: FeedbackData
    ) -> None:
        """Update learning metrics based on feedback."""
        try:
            # This would typically update some kind of metrics table
            # For now, we'll just log the metrics update
            self.logger.info(f"Updated learning metrics for user {feedback_data.user_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating learning metrics: {e}")
    
    async def _check_model_update_triggers(
        self,
        db: AsyncSession,
        workspace_id: int
    ) -> None:
        """Check if we need to trigger model updates based on feedback patterns."""
        try:
            # Get recent feedback data
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            result = await db.execute(
                select(IntentClassificationHistory)
                .where(
                    and_(
                        IntentClassificationHistory.workspace_id == workspace_id,
                        IntentClassificationHistory.created_at >= cutoff_date,
                        IntentClassificationHistory.was_correct.isnot(None)
                    )
                )
            )
            recent_feedback = result.scalars().all()
            
            if len(recent_feedback) < self.min_samples_for_learning:
                return
            
            # Calculate accuracy
            correct_count = sum(1 for feedback in recent_feedback if feedback.was_correct)
            accuracy = correct_count / len(recent_feedback)
            
            # If accuracy is below threshold, trigger learning
            if accuracy < 0.7:  # 70% accuracy threshold
                self.logger.warning(f"Low accuracy detected: {accuracy:.2f}. Triggering learning update.")
                await self._trigger_learning_update(db, workspace_id)
            
        except Exception as e:
            self.logger.error(f"Error checking model update triggers: {e}")
    
    async def _trigger_learning_update(
        self,
        db: AsyncSession,
        workspace_id: int
    ) -> None:
        """Trigger a learning update for the workspace."""
        try:
            # This would typically trigger some kind of model retraining
            # For now, we'll just log the trigger
            self.logger.info(f"Triggered learning update for workspace {workspace_id}")
            
        except Exception as e:
            self.logger.error(f"Error triggering learning update: {e}")
    
    async def get_learning_metrics(
        self,
        db: AsyncSession,
        workspace_id: int,
        days_back: int = 30
    ) -> LearningMetrics:
        """Get learning metrics for the workspace."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Get total interactions
            result = await db.execute(
                select(func.count(IntentClassificationHistory.id))
                .where(
                    and_(
                        IntentClassificationHistory.workspace_id == workspace_id,
                        IntentClassificationHistory.created_at >= cutoff_date
                    )
                )
            )
            total_interactions = result.scalar() or 0
            
            # Get classification accuracy
            result = await db.execute(
                select(IntentClassificationHistory)
                .where(
                    and_(
                        IntentClassificationHistory.workspace_id == workspace_id,
                        IntentClassificationHistory.created_at >= cutoff_date,
                        IntentClassificationHistory.was_correct.isnot(None)
                    )
                )
            )
            feedback_data = result.scalars().all()
            
            if feedback_data:
                correct_count = sum(1 for feedback in feedback_data if feedback.was_correct)
                classification_accuracy = correct_count / len(feedback_data)
            else:
                classification_accuracy = 0.0
            
            # Get user satisfaction average
            result = await db.execute(
                select(func.avg(IntentClassificationHistory.user_satisfaction))
                .where(
                    and_(
                        IntentClassificationHistory.workspace_id == workspace_id,
                        IntentClassificationHistory.created_at >= cutoff_date,
                        IntentClassificationHistory.user_satisfaction.isnot(None)
                    )
                )
            )
            user_satisfaction_avg = result.scalar() or 0.0
            
            # Get response effectiveness average
            result = await db.execute(
                select(func.avg(ResponseEffectiveness.user_rating))
                .where(
                    and_(
                        ResponseEffectiveness.workspace_id == workspace_id,
                        ResponseEffectiveness.created_at >= cutoff_date,
                        ResponseEffectiveness.user_rating.isnot(None)
                    )
                )
            )
            response_effectiveness_avg = result.scalar() or 0.0
            
            # Calculate learning confidence
            learning_confidence = min(1.0, total_interactions / 100.0)
            
            # Determine improvement trend
            improvement_trend = await self._calculate_improvement_trend(db, workspace_id, cutoff_date)
            
            # Get top performing intents
            top_performing_intents = await self._get_top_performing_intents(db, workspace_id, cutoff_date)
            
            # Get areas for improvement
            areas_for_improvement = await self._get_areas_for_improvement(db, workspace_id, cutoff_date)
            
            return LearningMetrics(
                total_interactions=total_interactions,
                classification_accuracy=classification_accuracy,
                user_satisfaction_avg=user_satisfaction_avg,
                response_effectiveness_avg=response_effectiveness_avg,
                learning_confidence=learning_confidence,
                improvement_trend=improvement_trend,
                top_performing_intents=top_performing_intents,
                areas_for_improvement=areas_for_improvement
            )
            
        except Exception as e:
            self.logger.error(f"Error getting learning metrics: {e}")
            return LearningMetrics(
                total_interactions=0,
                classification_accuracy=0.0,
                user_satisfaction_avg=0.0,
                response_effectiveness_avg=0.0,
                learning_confidence=0.0,
                improvement_trend="unknown",
                top_performing_intents=[],
                areas_for_improvement=[]
            )
    
    async def _calculate_improvement_trend(
        self,
        db: AsyncSession,
        workspace_id: int,
        cutoff_date: datetime
    ) -> str:
        """Calculate improvement trend over time."""
        try:
            # Split the time period into two halves
            mid_date = cutoff_date + (datetime.utcnow() - cutoff_date) / 2
            
            # Get accuracy for first half
            result = await db.execute(
                select(IntentClassificationHistory)
                .where(
                    and_(
                        IntentClassificationHistory.workspace_id == workspace_id,
                        IntentClassificationHistory.created_at >= cutoff_date,
                        IntentClassificationHistory.created_at < mid_date,
                        IntentClassificationHistory.was_correct.isnot(None)
                    )
                )
            )
            early_feedback = result.scalars().all()
            
            # Get accuracy for second half
            result = await db.execute(
                select(IntentClassificationHistory)
                .where(
                    and_(
                        IntentClassificationHistory.workspace_id == workspace_id,
                        IntentClassificationHistory.created_at >= mid_date,
                        IntentClassificationHistory.was_correct.isnot(None)
                    )
                )
            )
            recent_feedback = result.scalars().all()
            
            if not early_feedback or not recent_feedback:
                return "insufficient_data"
            
            early_accuracy = sum(1 for feedback in early_feedback if feedback.was_correct) / len(early_feedback)
            recent_accuracy = sum(1 for feedback in recent_feedback if feedback.was_correct) / len(recent_feedback)
            
            if recent_accuracy > early_accuracy + 0.05:
                return "improving"
            elif recent_accuracy < early_accuracy - 0.05:
                return "declining"
            else:
                return "stable"
                
        except Exception as e:
            self.logger.error(f"Error calculating improvement trend: {e}")
            return "unknown"
    
    async def _get_top_performing_intents(
        self,
        db: AsyncSession,
        workspace_id: int,
        cutoff_date: datetime
    ) -> List[str]:
        """Get top performing intents based on accuracy."""
        try:
            result = await db.execute(
                select(
                    IntentClassificationHistory.classified_intent,
                    func.count(IntentClassificationHistory.id).label('total'),
                    func.sum(func.case([(IntentClassificationHistory.was_correct == True, 1)], else_=0)).label('correct')
                )
                .where(
                    and_(
                        IntentClassificationHistory.workspace_id == workspace_id,
                        IntentClassificationHistory.created_at >= cutoff_date,
                        IntentClassificationHistory.was_correct.isnot(None)
                    )
                )
                .group_by(IntentClassificationHistory.classified_intent)
                .having(func.count(IntentClassificationHistory.id) >= 5)  # At least 5 samples
                .order_by(func.sum(func.case([(IntentClassificationHistory.was_correct == True, 1)], else_=0)) / func.count(IntentClassificationHistory.id).desc())
                .limit(5)
            )
            
            results = result.all()
            return [row.classified_intent for row in results]
            
        except Exception as e:
            self.logger.error(f"Error getting top performing intents: {e}")
            return []
    
    async def _get_areas_for_improvement(
        self,
        db: AsyncSession,
        workspace_id: int,
        cutoff_date: datetime
    ) -> List[str]:
        """Get areas that need improvement based on low accuracy."""
        try:
            result = await db.execute(
                select(
                    IntentClassificationHistory.classified_intent,
                    func.count(IntentClassificationHistory.id).label('total'),
                    func.sum(func.case([(IntentClassificationHistory.was_correct == True, 1)], else_=0)).label('correct')
                )
                .where(
                    and_(
                        IntentClassificationHistory.workspace_id == workspace_id,
                        IntentClassificationHistory.created_at >= cutoff_date,
                        IntentClassificationHistory.was_correct.isnot(None)
                    )
                )
                .group_by(IntentClassificationHistory.classified_intent)
                .having(func.count(IntentClassificationHistory.id) >= 3)  # At least 3 samples
                .order_by(func.sum(func.case([(IntentClassificationHistory.was_correct == True, 1)], else_=0)) / func.count(IntentClassificationHistory.id).asc())
                .limit(3)
            )
            
            results = result.all()
            improvement_areas = []
            
            for row in results:
                accuracy = row.correct / row.total if row.total > 0 else 0
                if accuracy < 0.6:  # Less than 60% accuracy
                    improvement_areas.append(f"{row.classified_intent} (accuracy: {accuracy:.1%})")
            
            return improvement_areas
            
        except Exception as e:
            self.logger.error(f"Error getting areas for improvement: {e}")
            return []
    
    async def record_classification(
        self,
        db: AsyncSession,
        workspace_id: int,
        user_id: str,
        channel_id: str,
        query_id: Optional[int],
        original_message: str,
        classified_intent: str,
        confidence_score: float,
        classification_method: str,
        conversation_context: Dict[str, Any],
        user_context: Dict[str, Any],
        channel_context: Dict[str, Any]
    ) -> int:
        """Record a classification for learning purposes."""
        try:
            classification_record = IntentClassificationHistory(
                workspace_id=workspace_id,
                user_id=user_id,
                channel_id=channel_id,
                query_id=query_id,
                original_message=original_message,
                classified_intent=classified_intent,
                confidence_score=confidence_score,
                classification_method=classification_method,
                conversation_context=conversation_context,
                user_context=user_context,
                channel_context=channel_context,
                response_generated=False,
                follow_up_required=False
            )
            
            db.add(classification_record)
            await db.commit()
            
            return classification_record.id
            
        except Exception as e:
            self.logger.error(f"Error recording classification: {e}")
            await db.rollback()
            return 0
    
    async def record_response_effectiveness(
        self,
        db: AsyncSession,
        workspace_id: int,
        user_id: str,
        query_id: int,
        response_type: str,
        response_style: str,
        response_length: int,
        intent_confidence: float,
        conversation_stage: str,
        time_of_day: Optional[str] = None
    ) -> bool:
        """Record response effectiveness for learning."""
        try:
            effectiveness_record = ResponseEffectiveness(
                workspace_id=workspace_id,
                user_id=user_id,
                query_id=query_id,
                response_type=response_type,
                response_style=response_style,
                response_length=response_length,
                intent_confidence=intent_confidence,
                conversation_stage=conversation_stage,
                time_of_day=time_of_day
            )
            
            db.add(effectiveness_record)
            await db.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording response effectiveness: {e}")
            await db.rollback()
            return False
