"""
Enhanced Conversation State Manager with sophisticated boundary detection.

This implements the core architectural improvement: conversation-level processing
instead of message-level processing.
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func
from loguru import logger

from ..models.base import Conversation, Message, User
from ..services.openai_service import OpenAIService


class ConversationState(Enum):
    """Conversation states in the processing lifecycle."""
    INITIATED = "initiated"           # First message on a topic
    DEVELOPING = "developing"         # Active back-and-forth discussion
    PAUSED = "paused"                # No activity but likely to continue
    RESOLVED = "resolved"             # Natural conclusion reached
    ABANDONED = "abandoned"           # No activity for extended period
    PROCESSING = "processing"         # Currently being processed for knowledge extraction


@dataclass
class ConversationBoundary:
    """Represents a detected conversation boundary with metadata."""
    conversation_id: int
    state: ConversationState
    confidence: float
    start_time: datetime
    end_time: Optional[datetime]
    participant_count: int
    message_count: int
    topic: Optional[str]
    resolution_indicators: List[str]
    metadata: Dict[str, Any]


@dataclass
class TopicShift:
    """Represents a detected topic shift within a conversation."""
    message_id: int
    shift_confidence: float
    old_topic: str
    new_topic: str
    shift_type: str  # 'gradual', 'abrupt', 'branch', 'return'


class EnhancedConversationStateManager:
    """
    Sophisticated conversation boundary detection and state management.
    
    This is the core improvement that enables conversation-level processing
    instead of fragmented message-level processing.
    """
    
    def __init__(self):
        self.openai_service = OpenAIService()
        
        # Configuration for boundary detection
        self.config = {
            'pause_threshold_minutes': 30,     # Consider paused after 30 min
            'abandon_threshold_hours': 2,      # Consider abandoned after 2 hours
            'min_messages_for_analysis': 3,    # Need at least 3 messages
            'topic_shift_threshold': 0.7,      # Confidence threshold for topic shifts
            'resolution_confidence_threshold': 0.8
        }
        
        # Resolution markers - phrases that indicate conversation completion
        self.resolution_markers = [
            "decided", "let's go with", "sounds good", "agreed", "done",
            "resolved", "fixed", "completed", "that works", "perfect",
            "thanks", "got it", "understood", "makes sense", "approved",
            "confirmed", "final decision", "we'll do", "plan is",
            "action items", "next steps", "moving forward", "closing this"
        ]
        
        # Continuation indicators - suggest conversation will continue
        self.continuation_indicators = [
            "will get back", "need to check", "let me think", "hold on",
            "coming up", "in a bit", "later today", "tomorrow", "next week",
            "pending", "waiting for", "need more info", "to be continued",
            "follow up", "circle back", "revisit", "discuss further"
        ]

    async def analyze_conversation_state(
        self, 
        conversation_id: int, 
        db: AsyncSession
    ) -> ConversationBoundary:
        """
        Analyze the current state of a conversation using multiple detection methods.
        
        This is the core method that determines conversation boundaries.
        """
        try:
            # Get conversation and its messages
            messages = await self._get_conversation_messages(conversation_id, db)
            
            if not messages:
                return self._create_empty_boundary(conversation_id)
            
            if len(messages) < self.config['min_messages_for_analysis']:
                return self._create_developing_boundary(conversation_id, messages)
            
            # Multi-stage analysis
            temporal_analysis = await self._analyze_temporal_patterns(messages)
            participant_analysis = await self._analyze_participant_patterns(messages)
            content_analysis = await self._analyze_content_patterns(messages)
            ai_analysis = await self._ai_analyze_conversation_state(messages)
            
            # Combine analyses to determine state
            state, confidence = await self._determine_conversation_state(
                temporal_analysis, participant_analysis, content_analysis, ai_analysis
            )
            
            # Extract topic if available
            topic = await self._extract_conversation_topic(messages)
            
            return ConversationBoundary(
                conversation_id=conversation_id,
                state=state,
                confidence=confidence,
                start_time=messages[0].created_at,
                end_time=messages[-1].created_at if state in [ConversationState.RESOLVED, ConversationState.ABANDONED] else None,
                participant_count=len(set(msg.slack_user_id for msg in messages)),
                message_count=len(messages),
                topic=topic,
                resolution_indicators=content_analysis.get('resolution_indicators', []),
                metadata={
                    'temporal_analysis': temporal_analysis,
                    'participant_analysis': participant_analysis,
                    'content_analysis': content_analysis,
                    'ai_analysis': ai_analysis,
                    'last_activity': messages[-1].created_at
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing conversation state {conversation_id}: {e}")
            return self._create_error_boundary(conversation_id)

    async def detect_conversation_boundaries(
        self, 
        channel_id: str, 
        db: AsyncSession,
        lookback_hours: int = 24
    ) -> List[ConversationBoundary]:
        """
        Detect conversation boundaries across a channel within a time window.
        
        This groups related messages into logical conversation units.
        """
        try:
            # Get conversations in the channel within time window
            cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
            
            query = select(Conversation).where(
                and_(
                    Conversation.slack_channel_id == channel_id,
                    Conversation.updated_at >= cutoff_time
                )
            ).order_by(Conversation.created_at)
            
            result = await db.execute(query)
            conversations = result.scalars().all()
            
            boundaries = []
            for conv in conversations:
                boundary = await self.analyze_conversation_state(conv.id, db)
                boundaries.append(boundary)
            
            # Detect topic continuity across conversations
            boundaries = await self._detect_cross_conversation_continuity(boundaries, db)
            
            return boundaries
            
        except Exception as e:
            logger.error(f"Error detecting conversation boundaries in {channel_id}: {e}")
            return []

    async def should_extract_knowledge(self, boundary: ConversationBoundary) -> bool:
        """
        Determine if a conversation is ready for knowledge extraction.
        
        Only extract from RESOLVED conversations with sufficient content.
        """
        if boundary.state != ConversationState.RESOLVED:
            return False
        
        if boundary.confidence < self.config['resolution_confidence_threshold']:
            return False
        
        if boundary.message_count < self.config['min_messages_for_analysis']:
            return False
        
        # Check if conversation has substantive content
        if not boundary.topic or len(boundary.resolution_indicators) == 0:
            return False
        
        return True

    async def detect_topic_shifts(
        self, 
        messages: List[Message]
    ) -> List[TopicShift]:
        """
        Detect when conversation topic shifts within a thread.
        
        This helps identify conversation boundaries within long threads.
        """
        if len(messages) < 5:  # Need sufficient messages to detect shifts
            return []
        
        try:
            # Analyze messages in sliding windows
            topic_shifts = []
            window_size = 3
            
            for i in range(len(messages) - window_size):
                window_before = messages[i:i + window_size]
                window_after = messages[i + window_size:i + (window_size * 2)]
                
                if len(window_after) < window_size:
                    continue
                
                # Use AI to detect topic shift
                shift = await self._ai_detect_topic_shift(window_before, window_after)
                
                if shift and shift.shift_confidence > self.config['topic_shift_threshold']:
                    topic_shifts.append(shift)
            
            return topic_shifts
            
        except Exception as e:
            logger.error(f"Error detecting topic shifts: {e}")
            return []

    # Private helper methods

    async def _get_conversation_messages(
        self, 
        conversation_id: int, 
        db: AsyncSession
    ) -> List[Message]:
        """Get all messages in a conversation ordered by timestamp."""
        query = select(Message).where(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at)
        
        result = await db.execute(query)
        return result.scalars().all()

    async def _analyze_temporal_patterns(self, messages: List[Message]) -> Dict[str, Any]:
        """Analyze temporal patterns in message timing."""
        if len(messages) < 2:
            return {'pattern': 'insufficient_data'}
        
        # Calculate time gaps between messages
        gaps = []
        for i in range(1, len(messages)):
            gap = (messages[i].created_at - messages[i-1].created_at).total_seconds() / 60
            gaps.append(gap)
        
        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        recent_gap = gaps[-1] if gaps else 0
        
        # Determine temporal pattern
        if recent_gap > self.config['abandon_threshold_hours'] * 60:
            pattern = 'abandoned'
        elif recent_gap > self.config['pause_threshold_minutes']:
            pattern = 'paused'
        elif avg_gap < 5:  # Very active discussion
            pattern = 'active'
        else:
            pattern = 'normal'
        
        return {
            'pattern': pattern,
            'avg_gap_minutes': avg_gap,
            'max_gap_minutes': max_gap,
            'recent_gap_minutes': recent_gap,
            'total_duration_hours': (messages[-1].created_at - messages[0].created_at).total_seconds() / 3600
        }

    async def _analyze_participant_patterns(self, messages: List[Message]) -> Dict[str, Any]:
        """Analyze participant engagement patterns."""
        participants = {}
        
        for msg in messages:
            user_id = msg.slack_user_id
            if user_id not in participants:
                participants[user_id] = {
                    'message_count': 0,
                    'first_message': msg.created_at,
                    'last_message': msg.created_at,
                    'engagement_score': 0
                }
            
            participants[user_id]['message_count'] += 1
            participants[user_id]['last_message'] = msg.created_at
            participants[user_id]['engagement_score'] = len(msg.content.split())
        
        # Analyze engagement patterns
        total_participants = len(participants)
        active_participants = sum(1 for p in participants.values() if p['message_count'] > 1)
        
        # Check for participant exit patterns
        recent_messages = messages[-3:] if len(messages) >= 3 else messages
        recent_participants = set(msg.slack_user_id for msg in recent_messages)
        
        return {
            'total_participants': total_participants,
            'active_participants': active_participants,
            'engagement_ratio': active_participants / total_participants if total_participants > 0 else 0,
            'recent_participants': list(recent_participants),
            'participant_exit_detected': len(recent_participants) < total_participants / 2,
            'participants_detail': participants
        }

    async def _analyze_content_patterns(self, messages: List[Message]) -> Dict[str, Any]:
        """Analyze content patterns for resolution and continuation indicators."""
        recent_messages = messages[-3:] if len(messages) >= 3 else messages
        recent_content = ' '.join(msg.content.lower() for msg in recent_messages)
        
        # Check for resolution markers
        resolution_indicators = [
            marker for marker in self.resolution_markers 
            if marker in recent_content
        ]
        
        # Check for continuation indicators
        continuation_indicators = [
            indicator for indicator in self.continuation_indicators
            if indicator in recent_content
        ]
        
        # Analyze question patterns
        question_count = sum(1 for msg in recent_messages if '?' in msg.content)
        
        return {
            'resolution_indicators': resolution_indicators,
            'continuation_indicators': continuation_indicators,
            'has_resolution_markers': len(resolution_indicators) > 0,
            'has_continuation_markers': len(continuation_indicators) > 0,
            'recent_question_count': question_count,
            'resolution_strength': len(resolution_indicators) / len(recent_messages) if recent_messages else 0
        }

    async def _ai_analyze_conversation_state(self, messages: List[Message]) -> Dict[str, Any]:
        """Use AI to analyze conversation state and completion."""
        try:
            # Format conversation for AI analysis
            conversation_text = self._format_messages_for_ai(messages)
            
            system_prompt = """You are an expert conversation analyst. Analyze this Slack conversation and determine its current state.

CONVERSATION STATES:
- INITIATED: Just starting, limited discussion
- DEVELOPING: Active back-and-forth discussion ongoing  
- PAUSED: Discussion paused but likely to continue (waiting for info, etc.)
- RESOLVED: Natural conclusion reached with decision/outcome
- ABANDONED: No recent activity, discussion died out

ANALYSIS CRITERIA:
1. Look for resolution markers: decisions made, agreements reached, "sounds good", "let's do it"
2. Check for continuation signals: "will get back", "need to check", "more later"
3. Analyze participant engagement patterns
4. Evaluate if the discussion reached a logical conclusion

Return JSON:
{
    "state": "INITIATED|DEVELOPING|PAUSED|RESOLVED|ABANDONED",
    "confidence": 0.0-1.0,
    "reasoning": "Specific reasoning for this state assessment",
    "key_indicators": ["list", "of", "key", "phrases", "or", "patterns"],
    "completion_assessment": {
        "has_clear_outcome": true|false,
        "decision_made": true|false,
        "next_steps_defined": true|false,
        "participants_satisfied": true|false
    }
}"""

            response = await self.openai_service._make_request(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this conversation:\n\n{conversation_text}"}
                ],
                temperature=0.1,
                max_tokens=400
            )
            
            import json
            result = json.loads(response['choices'][0]['message']['content'])
            return result
            
        except Exception as e:
            logger.error(f"Error in AI conversation state analysis: {e}")
            return {
                'state': 'DEVELOPING',
                'confidence': 0.5,
                'reasoning': f'AI analysis failed: {e}',
                'key_indicators': [],
                'completion_assessment': {
                    'has_clear_outcome': False,
                    'decision_made': False,
                    'next_steps_defined': False,
                    'participants_satisfied': False
                }
            }

    async def _determine_conversation_state(
        self,
        temporal: Dict[str, Any],
        participant: Dict[str, Any], 
        content: Dict[str, Any],
        ai: Dict[str, Any]
    ) -> Tuple[ConversationState, float]:
        """Combine multiple analyses to determine final conversation state."""
        
        # Start with AI assessment
        ai_state = ai.get('state', 'DEVELOPING')
        ai_confidence = ai.get('confidence', 0.5)
        
        # Apply rule-based adjustments
        confidence_adjustments = []
        final_state = ConversationState(ai_state.lower())
        
        # Temporal pattern adjustments
        if temporal['pattern'] == 'abandoned':
            final_state = ConversationState.ABANDONED
            confidence_adjustments.append(0.9)
        elif temporal['pattern'] == 'paused' and content['has_continuation_markers']:
            final_state = ConversationState.PAUSED
            confidence_adjustments.append(0.8)
        
        # Content pattern adjustments
        if content['has_resolution_markers'] and content['resolution_strength'] > 0.5:
            if final_state == ConversationState.DEVELOPING:
                final_state = ConversationState.RESOLVED
            confidence_adjustments.append(0.8)
        
        # Participant pattern adjustments
        if participant['participant_exit_detected'] and not content['has_continuation_markers']:
            if final_state in [ConversationState.DEVELOPING, ConversationState.PAUSED]:
                final_state = ConversationState.ABANDONED
                confidence_adjustments.append(0.7)
        
        # Calculate final confidence
        if confidence_adjustments:
            final_confidence = max(confidence_adjustments)
        else:
            final_confidence = ai_confidence
        
        # Ensure confidence is reasonable
        final_confidence = max(0.1, min(0.95, final_confidence))
        
        return final_state, final_confidence

    async def _extract_conversation_topic(self, messages: List[Message]) -> Optional[str]:
        """Extract the main topic/subject of the conversation."""
        if len(messages) < 2:
            return None
        
        try:
            conversation_text = self._format_messages_for_ai(messages[:10])  # First 10 messages
            
            system_prompt = """Extract the main topic/subject of this conversation in 3-5 words.
Focus on the core subject being discussed, not the specific details.

Examples:
- "Database migration planning"
- "API authentication issue" 
- "Code review feedback"
- "Feature specification discussion"

Return only the topic phrase, nothing else."""

            response = await self.openai_service._make_request(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation_text}
                ],
                temperature=0.1,
                max_tokens=20
            )
            
            topic = response['choices'][0]['message']['content'].strip()
            return topic if len(topic) > 3 else None
            
        except Exception as e:
            logger.error(f"Error extracting conversation topic: {e}")
            return None

    def _format_messages_for_ai(self, messages: List[Message]) -> str:
        """Format messages for AI analysis."""
        formatted = []
        
        for msg in messages:
            timestamp = msg.created_at.strftime("%H:%M")
            user_id = msg.slack_user_id[-4:]  # Last 4 chars for brevity
            content = msg.content[:200]  # Truncate long messages
            formatted.append(f"[{timestamp}] {user_id}: {content}")
        
        return '\n'.join(formatted)

    def _create_empty_boundary(self, conversation_id: int) -> ConversationBoundary:
        """Create boundary for conversation with no messages."""
        return ConversationBoundary(
            conversation_id=conversation_id,
            state=ConversationState.INITIATED,
            confidence=0.1,
            start_time=datetime.utcnow(),
            end_time=None,
            participant_count=0,
            message_count=0,
            topic=None,
            resolution_indicators=[],
            metadata={}
        )

    def _create_developing_boundary(self, conversation_id: int, messages: List[Message]) -> ConversationBoundary:
        """Create boundary for developing conversation with few messages."""
        return ConversationBoundary(
            conversation_id=conversation_id,
            state=ConversationState.DEVELOPING,
            confidence=0.6,
            start_time=messages[0].created_at,
            end_time=None,
            participant_count=len(set(msg.slack_user_id for msg in messages)),
            message_count=len(messages),
            topic=None,
            resolution_indicators=[],
            metadata={'reason': 'insufficient_messages_for_full_analysis'}
        )

    def _create_error_boundary(self, conversation_id: int) -> ConversationBoundary:
        """Create boundary when analysis fails."""
        return ConversationBoundary(
            conversation_id=conversation_id,
            state=ConversationState.DEVELOPING,
            confidence=0.1,
            start_time=datetime.utcnow(),
            end_time=None,
            participant_count=0,
            message_count=0,
            topic=None,
            resolution_indicators=[],
            metadata={'error': 'analysis_failed'}
        )

    async def _detect_cross_conversation_continuity(
        self, 
        boundaries: List[ConversationBoundary], 
        db: AsyncSession
    ) -> List[ConversationBoundary]:
        """Detect when conversations continue across different threads/channels."""
        # This is a placeholder for more sophisticated cross-conversation analysis
        # For now, return boundaries as-is
        return boundaries

    async def _ai_detect_topic_shift(
        self, 
        window_before: List[Message], 
        window_after: List[Message]
    ) -> Optional[TopicShift]:
        """Use AI to detect topic shifts between message windows."""
        try:
            before_text = self._format_messages_for_ai(window_before)
            after_text = self._format_messages_for_ai(window_after)
            
            system_prompt = """Analyze if there's a topic shift between these two conversation segments.

Return JSON:
{
    "has_shift": true|false,
    "confidence": 0.0-1.0,
    "old_topic": "brief description of first topic",
    "new_topic": "brief description of second topic", 
    "shift_type": "gradual|abrupt|branch|return"
}"""

            response = await self.openai_service._make_request(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"BEFORE:\n{before_text}\n\nAFTER:\n{after_text}"}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            import json
            result = json.loads(response['choices'][0]['message']['content'])
            
            if result.get('has_shift'):
                return TopicShift(
                    message_id=window_after[0].id,
                    shift_confidence=result.get('confidence', 0.5),
                    old_topic=result.get('old_topic', 'unknown'),
                    new_topic=result.get('new_topic', 'unknown'),
                    shift_type=result.get('shift_type', 'gradual')
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting topic shift: {e}")
            return None
