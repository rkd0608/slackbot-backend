"""
Smart feedback collection and learning service.
Implements detailed feedback collection for continuous AI improvement.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc, update

from .openai_service import OpenAIService
from ..models.base import Query, QueryFeedback, KnowledgeItem, User, Workspace, KnowledgeQuality

class FeedbackType(Enum):
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"
    MISSING_INFO = "missing_info"
    INCORRECT_INFO = "incorrect_info"
    REPORT_ISSUE = "report_issue"

class FeedbackService:
    """Service for collecting and processing user feedback to improve AI accuracy."""
    
    def __init__(self):
        self.openai_service = OpenAIService()
        
        # Feedback thresholds for triggering actions
        self.negative_feedback_threshold = 0.3  # 30% negative feedback triggers review
        self.low_confidence_threshold = 0.6     # Below 60% confidence needs attention
        self.min_feedback_count = 5             # Minimum feedback before taking action
    
    async def record_feedback(
        self,
        query_id: int,
        user_id: int,
        workspace_id: int,
        feedback_type: str,
        feedback_data: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Record detailed user feedback for a query response."""
        try:
            # Validate feedback type
            try:
                feedback_enum = FeedbackType(feedback_type)
            except ValueError:
                raise ValueError(f"Invalid feedback type: {feedback_type}")
            
            # Create feedback record
            feedback = QueryFeedback(
                query_id=query_id,
                user_id=user_id,
                workspace_id=workspace_id,
                feedback_type=feedback_type,
                rating=feedback_data.get("rating"),
                is_helpful=feedback_data.get("is_helpful"),
                feedback_text=feedback_data.get("feedback_text"),
                issue_category=feedback_data.get("issue_category"),
                suggested_correction=feedback_data.get("suggested_correction"),
                interaction_metadata=feedback_data.get("metadata", {})
            )
            
            db.add(feedback)
            await db.flush()
            
            # Update knowledge quality scores
            await self._update_knowledge_quality(query_id, feedback_type, feedback_data, db)
            
            # Check if this feedback triggers any automated actions
            actions_triggered = await self._check_feedback_triggers(query_id, workspace_id, db)
            
            await db.commit()
            
            logger.info(f"Recorded feedback for query {query_id}: {feedback_type}")
            
            return {
                "feedback_id": feedback.id,
                "actions_triggered": actions_triggered,
                "message": "Feedback recorded successfully"
            }
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error recording feedback: {e}")
            raise
    
    async def collect_detailed_feedback(
        self,
        query_id: int,
        user_id: int,
        workspace_id: int,
        feedback_responses: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Process detailed feedback from modal or form submission."""
        try:
            # Extract specific feedback components
            feedback_components = {
                "accuracy_rating": feedback_responses.get("accuracy_rating"),
                "completeness_rating": feedback_responses.get("completeness_rating"),
                "relevance_rating": feedback_responses.get("relevance_rating"),
                "missing_information": feedback_responses.get("missing_information"),
                "incorrect_information": feedback_responses.get("incorrect_information"),
                "suggested_improvements": feedback_responses.get("suggested_improvements"),
                "overall_satisfaction": feedback_responses.get("overall_satisfaction")
            }
            
            # Record each component as separate feedback
            feedback_ids = []
            for component, value in feedback_components.items():
                if value is not None:
                    feedback_data = {
                        "rating": value if isinstance(value, (int, float)) else None,
                        "feedback_text": value if isinstance(value, str) else None,
                        "metadata": {"component": component}
                    }
                    
                    result = await self.record_feedback(
                        query_id, user_id, workspace_id, 
                        "accuracy" if "rating" in component else "missing_info",
                        feedback_data, db
                    )
                    feedback_ids.append(result["feedback_id"])
            
            # Analyze feedback for patterns
            feedback_analysis = await self._analyze_feedback_patterns(query_id, db)
            
            return {
                "feedback_ids": feedback_ids,
                "analysis": feedback_analysis,
                "message": "Detailed feedback processed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error collecting detailed feedback: {e}")
            raise
    
    async def get_feedback_insights(self, workspace_id: int, db: AsyncSession) -> Dict[str, Any]:
        """Get insights and analytics from collected feedback."""
        try:
            # Get feedback statistics
            feedback_stats = await self._get_feedback_statistics(workspace_id, db)
            
            # Identify problematic knowledge items
            problematic_items = await self._identify_problematic_knowledge(workspace_id, db)
            
            # Get improvement recommendations
            recommendations = await self._generate_improvement_recommendations(workspace_id, db)
            
            # Calculate overall satisfaction trends
            satisfaction_trends = await self._calculate_satisfaction_trends(workspace_id, db)
            
            return {
                "feedback_statistics": feedback_stats,
                "problematic_knowledge_items": problematic_items,
                "improvement_recommendations": recommendations,
                "satisfaction_trends": satisfaction_trends,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback insights: {e}")
            raise
    
    async def trigger_knowledge_review(self, knowledge_item_id: int, reason: str, db: AsyncSession) -> Dict[str, Any]:
        """Trigger automated review of a knowledge item based on feedback."""
        try:
            # Get the knowledge item
            result = await db.execute(
                select(KnowledgeItem).where(KnowledgeItem.id == knowledge_item_id)
            )
            knowledge_item = result.scalar_one_or_none()
            
            if not knowledge_item:
                raise ValueError(f"Knowledge item {knowledge_item_id} not found")
            
            # Use AI to review the knowledge item quality
            review_result = await self._ai_review_knowledge_item(knowledge_item, reason)
            
            # Update knowledge quality record
            await self._update_knowledge_quality_record(
                knowledge_item_id, 
                review_result,
                db
            )
            
            logger.info(f"Triggered review for knowledge item {knowledge_item_id}: {reason}")
            
            return {
                "knowledge_item_id": knowledge_item_id,
                "review_result": review_result,
                "action_taken": "Knowledge item flagged for review" if review_result["needs_attention"] else "Knowledge item validated",
                "reviewed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error triggering knowledge review: {e}")
            raise
    
    async def _update_knowledge_quality(
        self, 
        query_id: int, 
        feedback_type: str, 
        feedback_data: Dict[str, Any], 
        db: AsyncSession
    ) -> None:
        """Update knowledge quality scores based on feedback."""
        try:
            # Get the query and associated knowledge items
            result = await db.execute(
                select(Query).where(Query.id == query_id)
            )
            query = result.scalar_one_or_none()
            
            if not query or not query.response:
                return
            
            # Extract knowledge item IDs from query response
            search_results = query.response.get("search_results", [])
            knowledge_item_ids = [item.get("knowledge_item_id") for item in search_results if item.get("knowledge_item_id")]
            
            # Update quality scores for each knowledge item
            for knowledge_item_id in knowledge_item_ids:
                await self._update_single_knowledge_quality(
                    knowledge_item_id, 
                    feedback_type, 
                    feedback_data, 
                    db
                )
            
        except Exception as e:
            logger.error(f"Error updating knowledge quality: {e}")
    
    async def _update_single_knowledge_quality(
        self, 
        knowledge_item_id: int, 
        feedback_type: str, 
        feedback_data: Dict[str, Any], 
        db: AsyncSession
    ) -> None:
        """Update quality score for a single knowledge item."""
        try:
            # Get or create knowledge quality record
            result = await db.execute(
                select(KnowledgeQuality).where(
                    KnowledgeQuality.knowledge_item_id == knowledge_item_id
                )
            )
            quality_record = result.scalar_one_or_none()
            
            if not quality_record:
                # Get workspace_id from knowledge item
                ki_result = await db.execute(
                    select(KnowledgeItem.workspace_id).where(KnowledgeItem.id == knowledge_item_id)
                )
                workspace_id = ki_result.scalar_one()
                
                quality_record = KnowledgeQuality(
                    knowledge_item_id=knowledge_item_id,
                    workspace_id=workspace_id,
                    positive_feedback_count=0,
                    negative_feedback_count=0,
                    total_usage_count=1,
                    quality_score=0.5,
                    confidence_adjustment=0.0
                )
                db.add(quality_record)
            
            # Update counts based on feedback type
            if feedback_type in ["helpful", "accuracy"] and feedback_data.get("is_helpful", False):
                quality_record.positive_feedback_count += 1
            elif feedback_type in ["not_helpful", "incorrect_info", "report_issue"]:
                quality_record.negative_feedback_count += 1
            
            quality_record.total_usage_count += 1
            quality_record.last_feedback_at = datetime.utcnow()
            
            # Recalculate quality score
            total_feedback = quality_record.positive_feedback_count + quality_record.negative_feedback_count
            if total_feedback > 0:
                quality_record.quality_score = quality_record.positive_feedback_count / total_feedback
                
                # Flag for review if quality is low and has enough feedback
                if quality_record.quality_score < 0.4 and total_feedback >= self.min_feedback_count:
                    quality_record.needs_review = True
            
            quality_record.last_quality_update = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating single knowledge quality: {e}")
    
    async def _check_feedback_triggers(self, query_id: int, workspace_id: int, db: AsyncSession) -> List[str]:
        """Check if feedback triggers any automated actions."""
        try:
            actions_triggered = []
            
            # Get recent feedback for this query
            result = await db.execute(
                select(QueryFeedback).where(
                    and_(
                        QueryFeedback.query_id == query_id,
                        QueryFeedback.created_at >= datetime.utcnow() - timedelta(days=1)
                    )
                )
            )
            recent_feedback = result.fetchall()
            
            if not recent_feedback:
                return actions_triggered
            
            # Check for high negative feedback rate
            negative_feedback = sum(1 for f in recent_feedback if f.feedback_type in ["not_helpful", "incorrect_info"])
            negative_rate = negative_feedback / len(recent_feedback)
            
            if negative_rate > self.negative_feedback_threshold and len(recent_feedback) >= self.min_feedback_count:
                actions_triggered.append("flagged_for_review")
                # TODO: Actually flag the query/knowledge for review
            
            return actions_triggered
            
        except Exception as e:
            logger.error(f"Error checking feedback triggers: {e}")
            return []
    
    async def _analyze_feedback_patterns(self, query_id: int, db: AsyncSession) -> Dict[str, Any]:
        """Analyze feedback patterns for insights."""
        try:
            # Get all feedback for this query
            result = await db.execute(
                select(QueryFeedback).where(QueryFeedback.query_id == query_id)
            )
            feedback_list = result.fetchall()
            
            if not feedback_list:
                return {"pattern": "insufficient_data"}
            
            # Analyze patterns
            feedback_types = [f.feedback_type for f in feedback_list]
            ratings = [f.rating for f in feedback_list if f.rating is not None]
            
            analysis = {
                "total_feedback": len(feedback_list),
                "feedback_types": {ftype: feedback_types.count(ftype) for ftype in set(feedback_types)},
                "average_rating": sum(ratings) / len(ratings) if ratings else None,
                "pattern": "positive" if sum(1 for f in feedback_list if f.is_helpful) > len(feedback_list) / 2 else "needs_improvement"
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing feedback patterns: {e}")
            return {"pattern": "analysis_error"}
    
    async def _get_feedback_statistics(self, workspace_id: int, db: AsyncSession) -> Dict[str, Any]:
        """Get comprehensive feedback statistics."""
        try:
            # Get feedback counts by type
            result = await db.execute(
                select(
                    QueryFeedback.feedback_type,
                    func.count(QueryFeedback.id).label('count')
                ).where(
                    QueryFeedback.workspace_id == workspace_id
                ).group_by(QueryFeedback.feedback_type)
            )
            
            feedback_by_type = {row.feedback_type: row.count for row in result.fetchall()}
            
            # Get average ratings
            result = await db.execute(
                select(func.avg(QueryFeedback.rating)).where(
                    and_(
                        QueryFeedback.workspace_id == workspace_id,
                        QueryFeedback.rating.isnot(None)
                    )
                )
            )
            average_rating = result.scalar() or 0.0
            
            # Get satisfaction rate
            result = await db.execute(
                select(
                    func.count(QueryFeedback.id).filter(QueryFeedback.is_helpful == True).label('helpful'),
                    func.count(QueryFeedback.id).label('total')
                ).where(
                    and_(
                        QueryFeedback.workspace_id == workspace_id,
                        QueryFeedback.is_helpful.isnot(None)
                    )
                )
            )
            satisfaction_data = result.first()
            satisfaction_rate = (satisfaction_data.helpful / satisfaction_data.total) if satisfaction_data.total > 0 else 0.0
            
            return {
                "feedback_by_type": feedback_by_type,
                "average_rating": float(average_rating),
                "satisfaction_rate": satisfaction_rate,
                "total_feedback": sum(feedback_by_type.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback statistics: {e}")
            return {}
    
    async def _identify_problematic_knowledge(self, workspace_id: int, db: AsyncSession) -> List[Dict[str, Any]]:
        """Identify knowledge items with poor feedback."""
        try:
            result = await db.execute(
                select(KnowledgeQuality, KnowledgeItem.title).join(
                    KnowledgeItem, KnowledgeQuality.knowledge_item_id == KnowledgeItem.id
                ).where(
                    and_(
                        KnowledgeQuality.workspace_id == workspace_id,
                        KnowledgeQuality.quality_score < 0.6,
                        KnowledgeQuality.total_usage_count >= self.min_feedback_count
                    )
                ).order_by(KnowledgeQuality.quality_score.asc())
            )
            
            problematic_items = []
            for quality, title in result.fetchall():
                problematic_items.append({
                    "knowledge_item_id": quality.knowledge_item_id,
                    "title": title,
                    "quality_score": quality.quality_score,
                    "positive_feedback": quality.positive_feedback_count,
                    "negative_feedback": quality.negative_feedback_count,
                    "total_usage": quality.total_usage_count,
                    "needs_review": quality.needs_review
                })
            
            return problematic_items
            
        except Exception as e:
            logger.error(f"Error identifying problematic knowledge: {e}")
            return []
    
    async def _generate_improvement_recommendations(self, workspace_id: int, db: AsyncSession) -> List[str]:
        """Generate specific recommendations based on feedback patterns."""
        try:
            recommendations = []
            
            # Get feedback statistics
            stats = await self._get_feedback_statistics(workspace_id, db)
            
            if stats.get("satisfaction_rate", 0) < 0.7:
                recommendations.append("Overall satisfaction is low - review knowledge extraction prompts")
            
            if stats.get("average_rating", 0) < 3.0:
                recommendations.append("Average ratings are low - improve response quality and relevance")
            
            feedback_by_type = stats.get("feedback_by_type", {})
            if feedback_by_type.get("incorrect_info", 0) > feedback_by_type.get("helpful", 0) * 0.2:
                recommendations.append("High rate of incorrect information - strengthen source validation")
            
            if feedback_by_type.get("missing_info", 0) > feedback_by_type.get("helpful", 0) * 0.3:
                recommendations.append("Users frequently report missing information - expand knowledge coverage")
            
            if not recommendations:
                recommendations.append("Feedback patterns look healthy - continue monitoring")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating improvement recommendations: {e}")
            return ["Error generating recommendations"]
    
    async def _calculate_satisfaction_trends(self, workspace_id: int, db: AsyncSession) -> Dict[str, Any]:
        """Calculate satisfaction trends over time."""
        try:
            # Get satisfaction by day for the last 30 days
            result = await db.execute(
                select(
                    func.date(QueryFeedback.created_at).label('date'),
                    func.avg(QueryFeedback.rating).label('avg_rating'),
                    func.count(QueryFeedback.id).filter(QueryFeedback.is_helpful == True).label('helpful'),
                    func.count(QueryFeedback.id).label('total')
                ).where(
                    and_(
                        QueryFeedback.workspace_id == workspace_id,
                        QueryFeedback.created_at >= datetime.utcnow() - timedelta(days=30)
                    )
                ).group_by(func.date(QueryFeedback.created_at)).order_by('date')
            )
            
            trends = []
            for row in result.fetchall():
                satisfaction_rate = (row.helpful / row.total) if row.total > 0 else 0.0
                trends.append({
                    "date": row.date.isoformat() if row.date else None,
                    "average_rating": float(row.avg_rating) if row.avg_rating else 0.0,
                    "satisfaction_rate": satisfaction_rate,
                    "total_feedback": row.total
                })
            
            return {
                "daily_trends": trends,
                "trend_direction": "improving" if len(trends) > 1 and trends[-1]["satisfaction_rate"] > trends[0]["satisfaction_rate"] else "stable"
            }
            
        except Exception as e:
            logger.error(f"Error calculating satisfaction trends: {e}")
            return {}
    
    async def _ai_review_knowledge_item(self, knowledge_item: KnowledgeItem, reason: str) -> Dict[str, Any]:
        """Use AI to review a knowledge item's quality."""
        try:
            system_prompt = """You are reviewing a knowledge item for quality issues based on user feedback.
            
            Analyze the knowledge item and determine:
            1. Is the information accurate and well-sourced?
            2. Is it complete and comprehensive?
            3. Is it relevant and useful?
            4. Are there any obvious errors or inconsistencies?
            5. Does it need updating or revision?
            
            Return JSON:
            {
                "needs_attention": boolean,
                "quality_score": 0.0-1.0,
                "issues_found": ["issue1", "issue2"],
                "recommendations": ["rec1", "rec2"],
                "confidence": 0.0-1.0
            }"""
            
            user_message = f"""Knowledge Item to Review:
            Title: {knowledge_item.title}
            Content: {knowledge_item.content}
            Summary: {knowledge_item.summary}
            
            Review Reason: {reason}
            
            Please analyze this knowledge item for quality issues."""
            
            response = await self.openai_service._make_request(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response['choices'][0]['message']['content']
            review_result = json.loads(content)
            
            return review_result
            
        except Exception as e:
            logger.error(f"Error in AI knowledge review: {e}")
            return {
                "needs_attention": True,
                "quality_score": 0.5,
                "issues_found": ["Review failed - manual inspection needed"],
                "recommendations": ["Manual review required"],
                "confidence": 0.0
            }
    
    async def _update_knowledge_quality_record(
        self, 
        knowledge_item_id: int, 
        review_result: Dict[str, Any], 
        db: AsyncSession
    ) -> None:
        """Update knowledge quality record based on AI review."""
        try:
            result = await db.execute(
                select(KnowledgeQuality).where(
                    KnowledgeQuality.knowledge_item_id == knowledge_item_id
                )
            )
            quality_record = result.scalar_one_or_none()
            
            if quality_record:
                quality_record.needs_review = review_result.get("needs_attention", False)
                quality_record.is_flagged = review_result.get("needs_attention", False)
                
                # Adjust confidence based on review
                if review_result.get("quality_score"):
                    quality_record.confidence_adjustment = review_result["quality_score"] - 0.5
                
                quality_record.last_quality_update = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating knowledge quality record: {e}")
