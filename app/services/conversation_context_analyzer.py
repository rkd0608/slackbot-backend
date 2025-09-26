"""Conversation context analyzer for understanding conversation flow and context."""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc, or_

from ..models.base import Message, Conversation, User, Workspace
from ..models.intent_classification import ConversationContext, ChannelCulture, UserCommunicationProfile


@dataclass
class ConversationAnalysis:
    """Analysis result of conversation context."""
    thread_messages: List[Dict[str, Any]]
    recent_messages: List[Dict[str, Any]]
    participants: List[str]
    channel_context: Dict[str, Any]
    temporal_context: Dict[str, Any]
    bot_mentioned_recently: bool
    is_direct_response: bool
    conversation_stage: str
    active_topic: Optional[str]
    sentiment_score: float
    urgency_indicators: List[str]
    context_confidence: float


class ConversationContextAnalyzer:
    """Analyzes conversation context for better intent classification."""
    
    def __init__(self):
        self.logger = logger
    
    async def analyze_conversation_context(
        self,
        db: AsyncSession,
        workspace_id: int,
        channel_id: str,
        thread_ts: Optional[str] = None,
        user_id: Optional[str] = None,
        message_text: Optional[str] = None
    ) -> ConversationAnalysis:
        """Analyze the current conversation context."""
        try:
            # Get recent messages in the thread or channel
            thread_messages = await self._get_thread_messages(db, workspace_id, channel_id, thread_ts)
            recent_messages = await self._get_recent_messages(db, workspace_id, channel_id)
            
            # Analyze participants
            participants = await self._analyze_participants(db, thread_messages, recent_messages)
            
            # Get channel context
            channel_context = await self._get_channel_context(db, workspace_id, channel_id)
            
            # Analyze temporal context
            temporal_context = await self._analyze_temporal_context(recent_messages)
            
            # Check if bot was mentioned recently
            bot_mentioned_recently = await self._check_bot_mentions(recent_messages)
            
            # Check if this is a direct response to bot
            is_direct_response = await self._is_direct_response_to_bot(thread_messages, user_id)
            
            # Analyze conversation stage
            conversation_stage = await self._analyze_conversation_stage(thread_messages, recent_messages)
            
            # Extract active topic
            active_topic = await self._extract_active_topic(thread_messages, recent_messages)
            
            # Analyze sentiment
            sentiment_score = await self._analyze_sentiment(thread_messages, recent_messages)
            
            # Look for urgency indicators
            urgency_indicators = await self._detect_urgency_indicators(message_text, recent_messages)
            
            # Calculate context confidence
            context_confidence = await self._calculate_context_confidence(
                thread_messages, recent_messages, participants, channel_context
            )
            
            return ConversationAnalysis(
                thread_messages=thread_messages,
                recent_messages=recent_messages,
                participants=participants,
                channel_context=channel_context,
                temporal_context=temporal_context,
                bot_mentioned_recently=bot_mentioned_recently,
                is_direct_response=is_direct_response,
                conversation_stage=conversation_stage,
                active_topic=active_topic,
                sentiment_score=sentiment_score,
                urgency_indicators=urgency_indicators,
                context_confidence=context_confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing conversation context: {e}")
            # Return minimal context on error
            return ConversationAnalysis(
                thread_messages=[],
                recent_messages=[],
                participants=[],
                channel_context={},
                temporal_context={},
                bot_mentioned_recently=False,
                is_direct_response=False,
                conversation_stage="unknown",
                active_topic=None,
                sentiment_score=0.0,
                urgency_indicators=[],
                context_confidence=0.0
            )
    
    async def _get_thread_messages(
        self,
        db: AsyncSession,
        workspace_id: int,
        channel_id: str,
        thread_ts: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get messages from the current thread."""
        try:
            if not thread_ts:
                return []
            
            # Get conversation for this thread
            result = await db.execute(
                select(Conversation)
                .where(
                    and_(
                        Conversation.workspace_id == workspace_id,
                        Conversation.slack_channel_id == channel_id,
                        Conversation.thread_timestamp == thread_ts
                    )
                )
            )
            conversation = result.scalar_one_or_none()
            
            if not conversation:
                return []
            
            # Get messages in this conversation
            result = await db.execute(
                select(Message)
                .where(Message.conversation_id == conversation.id)
                .order_by(Message.created_at.desc())
                .limit(20)
            )
            messages = result.scalars().all()
            
            return [
                {
                    "id": msg.id,
                    "content": msg.content,
                    "user_id": msg.slack_user_id,
                    "timestamp": msg.created_at,
                    "metadata": msg.message_metadata
                }
                for msg in messages
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting thread messages: {e}")
            return []
    
    async def _get_recent_messages(
        self,
        db: AsyncSession,
        workspace_id: int,
        channel_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent messages in the channel."""
        try:
            # Get recent conversation
            result = await db.execute(
                select(Conversation)
                .where(
                    and_(
                        Conversation.workspace_id == workspace_id,
                        Conversation.slack_channel_id == channel_id
                    )
                )
                .order_by(Conversation.created_at.desc())
                .limit(1)
            )
            conversation = result.scalar_one_or_none()
            
            if not conversation:
                return []
            
            # Get recent messages
            result = await db.execute(
                select(Message)
                .where(Message.conversation_id == conversation.id)
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
            messages = result.scalars().all()
            
            return [
                {
                    "id": msg.id,
                    "content": msg.content,
                    "user_id": msg.slack_user_id,
                    "timestamp": msg.created_at,
                    "metadata": msg.message_metadata
                }
                for msg in messages
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting recent messages: {e}")
            return []
    
    async def _analyze_participants(
        self,
        db: AsyncSession,
        thread_messages: List[Dict[str, Any]],
        recent_messages: List[Dict[str, Any]]
    ) -> List[str]:
        """Analyze participants in the conversation."""
        try:
            participants = set()
            
            # Add participants from thread messages
            for msg in thread_messages:
                participants.add(msg["user_id"])
            
            # Add participants from recent messages
            for msg in recent_messages:
                participants.add(msg["user_id"])
            
            return list(participants)
            
        except Exception as e:
            self.logger.error(f"Error analyzing participants: {e}")
            return []
    
    async def _get_channel_context(
        self,
        db: AsyncSession,
        workspace_id: int,
        channel_id: str
    ) -> Dict[str, Any]:
        """Get channel-specific context and culture."""
        try:
            # Get channel culture
            result = await db.execute(
                select(ChannelCulture)
                .where(
                    and_(
                        ChannelCulture.workspace_id == workspace_id,
                        ChannelCulture.channel_id == channel_id
                    )
                )
            )
            channel_culture = result.scalar_one_or_none()
            
            if channel_culture:
                return {
                    "formality_level": channel_culture.formality_level,
                    "topic_focus": channel_culture.topic_focus,
                    "response_expectations": channel_culture.response_expectations,
                    "common_topics": channel_culture.common_topics or [],
                    "active_participants": channel_culture.active_participants or [],
                    "channel_purpose": channel_culture.channel_purpose,
                    "communication_norms": channel_culture.communication_norms or {}
                }
            else:
                # Return default context
                return {
                    "formality_level": 0.5,
                    "topic_focus": "general",
                    "response_expectations": "balanced",
                    "common_topics": [],
                    "active_participants": [],
                    "channel_purpose": "general discussion",
                    "communication_norms": {}
                }
                
        except Exception as e:
            self.logger.error(f"Error getting channel context: {e}")
            return {}
    
    async def _analyze_temporal_context(
        self,
        recent_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze temporal aspects of the conversation."""
        try:
            now = datetime.utcnow()
            
            if not recent_messages:
                return {
                    "time_of_day": "unknown",
                    "day_of_week": "unknown",
                    "hours_since_last_message": 999,
                    "message_frequency": 0.0
                }
            
            # Get the most recent message
            latest_message = max(recent_messages, key=lambda x: x["timestamp"])
            hours_since_last = (now - latest_message["timestamp"]).total_seconds() / 3600
            
            # Analyze time of day
            hour = latest_message["timestamp"].hour
            if 6 <= hour < 12:
                time_of_day = "morning"
            elif 12 <= hour < 17:
                time_of_day = "afternoon"
            elif 17 <= hour < 22:
                time_of_day = "evening"
            else:
                time_of_day = "night"
            
            # Analyze day of week
            day_of_week = latest_message["timestamp"].strftime("%A").lower()
            
            # Calculate message frequency (messages per hour in last 24 hours)
            recent_24h = [msg for msg in recent_messages 
                         if (now - msg["timestamp"]).total_seconds() < 24 * 3600]
            message_frequency = len(recent_24h) / 24.0
            
            return {
                "time_of_day": time_of_day,
                "day_of_week": day_of_week,
                "hours_since_last_message": hours_since_last,
                "message_frequency": message_frequency
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal context: {e}")
            return {}
    
    async def _check_bot_mentions(self, recent_messages: List[Dict[str, Any]]) -> bool:
        """Check if bot was mentioned in recent messages."""
        try:
            bot_mention_patterns = ["@reno", "@bot", "reno", "bot"]
            
            for msg in recent_messages:
                content_lower = msg["content"].lower()
                for pattern in bot_mention_patterns:
                    if pattern in content_lower:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking bot mentions: {e}")
            return False
    
    async def _is_direct_response_to_bot(
        self,
        thread_messages: List[Dict[str, Any]],
        user_id: Optional[str]
    ) -> bool:
        """Check if this is a direct response to the bot."""
        try:
            if not thread_messages or not user_id:
                return False
            
            # Look for recent bot messages in the thread
            for msg in thread_messages[:5]:  # Check last 5 messages
                if msg["user_id"] == "bot" or "bot" in msg["user_id"].lower():
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking direct response: {e}")
            return False
    
    async def _analyze_conversation_stage(
        self,
        thread_messages: List[Dict[str, Any]],
        recent_messages: List[Dict[str, Any]]
    ) -> str:
        """Analyze what stage the conversation is in."""
        try:
            total_messages = len(thread_messages) + len(recent_messages)
            
            if total_messages == 0:
                return "starting"
            elif total_messages < 3:
                return "developing"
            elif total_messages < 10:
                return "active"
            else:
                return "concluding"
                
        except Exception as e:
            self.logger.error(f"Error analyzing conversation stage: {e}")
            return "unknown"
    
    async def _extract_active_topic(
        self,
        thread_messages: List[Dict[str, Any]],
        recent_messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Extract the main topic being discussed."""
        try:
            # Simple keyword extraction for now
            # In a more sophisticated system, this would use NLP
            all_messages = thread_messages + recent_messages
            
            if not all_messages:
                return None
            
            # Get the most recent message content
            recent_content = " ".join([msg["content"] for msg in all_messages[:5]])
            
            # Simple topic extraction (this could be enhanced with NLP)
            topic_keywords = {
                "project": ["project", "task", "work", "assignment"],
                "meeting": ["meeting", "call", "discussion", "sync"],
                "technical": ["code", "bug", "feature", "technical", "system"],
                "social": ["hello", "hi", "how are you", "thanks", "good morning"],
                "urgent": ["urgent", "asap", "emergency", "critical", "important"]
            }
            
            recent_content_lower = recent_content.lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in recent_content_lower for keyword in keywords):
                    return topic
            
            return "general"
            
        except Exception as e:
            self.logger.error(f"Error extracting active topic: {e}")
            return None
    
    async def _analyze_sentiment(
        self,
        thread_messages: List[Dict[str, Any]],
        recent_messages: List[Dict[str, Any]]
    ) -> float:
        """Analyze sentiment of the conversation."""
        try:
            # Simple sentiment analysis based on keywords
            # In a more sophisticated system, this would use a proper sentiment analysis model
            all_messages = thread_messages + recent_messages
            
            if not all_messages:
                return 0.0
            
            positive_words = ["good", "great", "excellent", "awesome", "thanks", "happy", "love", "amazing"]
            negative_words = ["bad", "terrible", "awful", "hate", "angry", "frustrated", "disappointed", "sad"]
            
            sentiment_score = 0.0
            total_words = 0
            
            for msg in all_messages[:10]:  # Analyze last 10 messages
                content_lower = msg["content"].lower()
                words = content_lower.split()
                total_words += len(words)
                
                for word in words:
                    if word in positive_words:
                        sentiment_score += 1
                    elif word in negative_words:
                        sentiment_score -= 1
            
            if total_words > 0:
                return sentiment_score / total_words
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    async def _detect_urgency_indicators(
        self,
        message_text: Optional[str],
        recent_messages: List[Dict[str, Any]]
    ) -> List[str]:
        """Detect urgency indicators in the message and conversation."""
        try:
            urgency_indicators = []
            
            if not message_text:
                return urgency_indicators
            
            text_lower = message_text.lower()
            
            # Check for urgency keywords
            urgency_keywords = {
                "urgent": ["urgent", "asap", "immediately", "right now"],
                "emergency": ["emergency", "critical", "crisis", "urgent"],
                "deadline": ["deadline", "due", "expires", "deadline"],
                "help": ["help", "stuck", "problem", "issue", "error"],
                "question": ["?", "question", "how", "what", "why", "when", "where"]
            }
            
            for indicator_type, keywords in urgency_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    urgency_indicators.append(indicator_type)
            
            # Check for multiple question marks or exclamation marks
            if "??" in text_lower or "!!" in text_lower:
                urgency_indicators.append("high_emphasis")
            
            return urgency_indicators
            
        except Exception as e:
            self.logger.error(f"Error detecting urgency indicators: {e}")
            return []
    
    async def _calculate_context_confidence(
        self,
        thread_messages: List[Dict[str, Any]],
        recent_messages: List[Dict[str, Any]],
        participants: List[str],
        channel_context: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the context analysis."""
        try:
            confidence = 0.0
            
            # Base confidence from message count
            total_messages = len(thread_messages) + len(recent_messages)
            if total_messages > 0:
                confidence += 0.3
            
            # Confidence from participant diversity
            if len(participants) > 1:
                confidence += 0.2
            
            # Confidence from channel context richness
            if channel_context.get("formality_level") is not None:
                confidence += 0.2
            
            if channel_context.get("topic_focus"):
                confidence += 0.1
            
            if channel_context.get("common_topics"):
                confidence += 0.1
            
            # Confidence from recent activity
            if recent_messages:
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating context confidence: {e}")
            return 0.0
