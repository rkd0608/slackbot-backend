"""Multi-layer intent classification system for Slack bot queries.

This module implements a hierarchical intent classification system with:
1. Rule-based filtering for obvious cases
2. Contextual analysis for conversation awareness  
3. AI-powered classification for ambiguous cases

The system is designed for high performance with early termination
and comprehensive intent taxonomy coverage.
"""

import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc

from .openai_service import OpenAIService
from .conversation_context_analyzer import ConversationContextAnalyzer, ConversationAnalysis
from .user_behavior_profiler import UserBehaviorProfiler, UserProfile
from .learning_engine import LearningEngine, FeedbackData
from ..models.base import Message, Conversation, User


@dataclass
class IntentClassificationResult:
    """Result of intent classification with confidence and metadata."""
    intent: str
    confidence: float
    classification_method: str
    contextual_metadata: Dict[str, Any]
    entities: List[str]
    temporal_scope: Optional[str]
    is_conversational_response: bool
    requires_knowledge_search: bool


@dataclass
class ConversationContext:
    """Context information for conversation-aware classification."""
    thread_messages: List[Dict[str, Any]]
    recent_messages: List[Dict[str, Any]]
    participants: List[str]
    channel_context: Dict[str, Any]
    temporal_context: Dict[str, Any]
    bot_mentioned_recently: bool
    is_direct_response: bool


class IntentClassifier:
    """
    Multi-layer intent classification system.
    
    Implements hierarchical classification with:
    1. Rule-based filtering (fast, handles 60-70% of cases)
    2. Contextual analysis (conversation awareness)
    3. AI-powered classification (ambiguous cases)
    """
    
    def __init__(self):
        self.openai_service = OpenAIService()
        self.context_analyzer = ConversationContextAnalyzer()
        self.user_profiler = UserBehaviorProfiler()
        self.learning_engine = LearningEngine()
        
        # Intent taxonomy matching requirements
        self.intent_categories = {
            'knowledge_query': {
                'description': 'Messages seeking specific information from team knowledge base',
                'patterns': [
                    # Direct question words - HIGHEST PRIORITY
                    r'\b(how|what|when|where|why|which|who)\s',
                    r'\b(explain|describe|tell me about|show me)\b',
                    r'\b(can you|could you|would you)\s+(find|search|look|get|retrieve|locate|help|tell|show|explain)\b',
                    r'\b(find|search|look|get|retrieve|locate)\s+(something|anything|information|data|details)\b',
                    r'\b(help me|assist me|support me)\b',
                    r'\b(need|want|looking for|seeking)\s+(to know|information|help|assistance|details)\b',
                    r'\b(do you know|do you have|is there|are there)\b',

                    # Process and status queries
                    r'\b(process|procedure|workflow|steps|method|approach)\b',
                    r'\b(decision|decided|chose|rationale|reason)\b',
                    r'\b(status|progress|update|current state|where are we)\b',
                    r'\b(timeline|schedule|deadline|when will|when did)\b',

                    # Problem-solving queries
                    r'\b(troubleshoot|fix|debug|error|problem|issue|bug)\b',
                    r'\b(resource|tool|service|library|document|documentation)\b',

                    # Conversational knowledge seeking
                    r'\b(any idea|any thoughts|suggestions|recommendations)\b',
                    r'\b(what about|how about|what if)\b',
                    r'\b(remember|recall|mentioned|discussed)\b',
                    r'\b(previous|earlier|before|last time)\b',

                    # Even vague help requests should be knowledge queries
                    r'\b(anything|something).*(help|find|know|information)\b',
                    r'\b(help|find|know|information).*(anything|something)\b'
                ],
                'confidence_threshold': 0.4  # LOWERED - be more generous with knowledge queries
            },
            'conversational_response': {
                'description': 'Messages responding to bot output rather than requesting new information',
                'patterns': [
                    r'\b(thanks?|thank you|thx)\b',
                    r'\b(ok|okay|got it|understood)\b',
                    r'\b(yes|no|correct|right|wrong)\b',
                    r'\b(agree|disagree|exactly|precisely)\b',
                    r'\b(more|less|additional|further)\b',
                    r'\b(clear|unclear|confusing|helpful)\b'
                ],
                'confidence_threshold': 0.6
            },
            'clarification_request': {
                'description': 'Messages asking for more detail about previous bot responses',
                'patterns': [
                    r'\b(can you|could you|would you)\s+(explain|clarify|elaborate|expand)\b',
                    r'\b(what do you mean|what does that mean)\b',
                    r'\b(more details?|more information|more specific)\b',
                    r'\b(how so|why so|what about)\b',
                    r'\b(example|examples|for instance)\b'
                ],
                'confidence_threshold': 0.7
            },
            'social_interaction': {
                'description': 'Casual conversational messages including greetings and thanks',
                'patterns': [
                    # VERY SPECIFIC patterns - only match pure social interactions
                    r'^(hi|hello|hey|good morning|good afternoon|good evening)!?$',
                    r'^(how are you|how\'s it going|what\'s up)\??$',
                    r'^(bye|goodbye|see you|catch you later)!?$',
                    r'^(great|awesome|cool|nice|good|bad)!?$',
                    r'^(thanks|thank you|thx)!?$',
                    # Simple greetings with mentions
                    r'^@\w+\s+(hi|hello|hey)!?$',
                    r'^@\w+\s+(hi|hello|hey)\s+\w+!?$',
                    # Emoji-only messages
                    r'^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+$'
                ],
                'confidence_threshold': 0.95  # VERY HIGH - only pure social interactions
            },
            'context_request': {
                'description': 'Messages asking about recent conversation context or summaries',
                'patterns': [
                    r'\b(what did we discuss|what was said|what happened)\b',
                    r'\b(summary|summarize|recap|overview)\b',
                    r'\b(previous|earlier|before|last time)\b',
                    r'\b(current discussion|this thread|this conversation)\b'
                ],
                'confidence_threshold': 0.7
            },
            'ignore': {
                'description': 'Messages not intended for bot processing',
                'patterns': [
                    r'^\s*$',  # Empty messages
                    r'^[^\w\s]*$',  # Only punctuation/symbols
                    r'\b(private|confidential|internal)\b',
                    r'^@\w+\s*$'  # Just a mention without content
                ],
                'confidence_threshold': 0.9
            }
        }
        
        # Response timing patterns for conversational analysis
        self.response_timing_patterns = {
            'immediate_response': 300,  # 5 minutes
            'follow_up_response': 3600,  # 1 hour
            'delayed_response': 86400  # 24 hours
        }
        
        # Question word patterns for knowledge queries
        self.question_words = {
            'what', 'how', 'when', 'where', 'why', 'which', 'who',
            'can', 'could', 'would', 'should', 'is', 'are', 'was', 'were'
        }
        
        # Temporal indicators
        self.temporal_indicators = {
            'today': 1,
            'this morning': 0.5,
            'this afternoon': 0.5,
            'yesterday': 1,
            'last hour': 1/24,
            'recent': 3,
            'recently': 3,
            'just now': 1/24,
            'earlier': 1,
            'last week': 7,
            'this week': 7,
            'last month': 30
        }

    async def classify_intent(
        self, 
        message_text: str,
        user_id: str,
        channel_id: str,
        workspace_id: int,
        thread_ts: Optional[str] = None,
        db: Optional[AsyncSession] = None
    ) -> IntentClassificationResult:
        """
        Main intent classification entry point with enhanced context awareness.
        
        Implements hierarchical classification with:
        1. User behavior profiling
        2. Conversation context analysis
        3. Rule-based classification
        4. Contextual classification
        5. AI-powered classification
        """
        try:
            # Phase 0: Update user profile with new message
            user_profile = None
            if db:
                user_profile = await self.user_profiler.update_user_profile(
                    db, user_id, workspace_id, message_text
                )
            
            # Phase 1: Analyze conversation context
            conversation_analysis = None
            if db:
                conversation_analysis = await self.context_analyzer.analyze_conversation_context(
                    db, workspace_id, channel_id, thread_ts, user_id, message_text
                )
            
            # Phase 2: Rule-based classification (enhanced with context)
            rule_result = await self._rule_based_classification(
                message_text, user_profile, conversation_analysis
            )
            if rule_result.confidence >= 0.7:  # High confidence threshold for rule-based
                logger.info(f"Rule-based classification: {rule_result.intent} (confidence: {rule_result.confidence:.2f})")
                # Record classification for learning
                if db:
                    await self._record_classification(
                        db, workspace_id, user_id, channel_id, None, message_text,
                        rule_result, conversation_analysis, user_profile
                    )
                return rule_result
            
            # Phase 3: Contextual classification (enhanced with user profile)
            if db and conversation_analysis:
                context_result = await self._contextual_classification(
                    message_text, conversation_analysis, user_profile
                )
                if context_result.confidence >= 0.6:  # Medium confidence threshold for contextual
                    logger.info(f"Contextual classification: {context_result.intent} (confidence: {context_result.confidence:.2f})")
                    # Record classification for learning
                    await self._record_classification(
                        db, workspace_id, user_id, channel_id, None, message_text,
                        context_result, conversation_analysis, user_profile
                    )
                    return context_result
            
            # Phase 4: AI-powered classification (enhanced with context and user profile)
            ai_result = await self._ai_classification(
                message_text, conversation_analysis, user_profile
            )
            logger.info(f"AI classification: {ai_result.intent} (confidence: {ai_result.confidence:.2f})")
            
            # Record classification for learning
            if db:
                await self._record_classification(
                    db, workspace_id, user_id, channel_id, None, message_text,
                    ai_result, conversation_analysis, user_profile
                )
            
            return ai_result
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            # Fallback to basic classification
            return IntentClassificationResult(
                intent="knowledge_query",
                confidence=0.5,
                classification_method="fallback",
                contextual_metadata={},
                entities=[],
                temporal_scope=None,
                is_conversational_response=False,
                requires_knowledge_search=True
            )

    async def _rule_based_classification(
        self, 
        message_text: str, 
        user_profile: Optional[UserProfile] = None,
        conversation_analysis: Optional[ConversationAnalysis] = None
    ) -> IntentClassificationResult:
        """Phase 1: Fast rule-based classification for obvious cases."""
        text_lower = message_text.lower().strip()
        
        # Check for empty or meaningless messages
        if not text_lower or len(text_lower) < 2:
            return IntentClassificationResult(
                intent="ignore",
                confidence=0.95,
                classification_method="rule_based",
                contextual_metadata={"reason": "empty_message"},
                entities=[],
                temporal_scope=None,
                is_conversational_response=False,
                requires_knowledge_search=False
            )
        
        # Check each intent category
        intent_scores = {}
        matched_patterns = {}
        
        for intent, config in self.intent_categories.items():
            score = 0
            patterns_matched = []
            
            for pattern in config['patterns']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score += 1
                    patterns_matched.append(pattern)
            
            if score > 0:
                intent_scores[intent] = score
                matched_patterns[intent] = patterns_matched
        
        # REMOVED: The artificial social interaction boost was breaking knowledge queries
        # Let pattern matching handle classification naturally
        
        # Determine best intent with weighted scoring
        if intent_scores:
            # Apply weighted scoring - knowledge queries get priority
            weighted_scores = {}
            for intent, score in intent_scores.items():
                if intent == 'knowledge_query':
                    weighted_scores[intent] = score * 1.5  # Boost knowledge queries
                elif intent == 'social_interaction':
                    weighted_scores[intent] = score * 0.5  # Reduce social interaction priority
                else:
                    weighted_scores[intent] = score

            best_intent = max(weighted_scores.items(), key=lambda x: x[1])[0]
            confidence = min(0.95, (weighted_scores[best_intent] / 5.0) + 0.3)  # Better confidence calculation
            
            # Extract entities and temporal scope
            entities = self._extract_entities(message_text)
            temporal_scope = self._detect_temporal_scope(message_text)
            
            # Determine if this is a conversational response
            is_conversational = best_intent == 'conversational_response'
            
            # Determine if knowledge search is required
            requires_search = best_intent in ['knowledge_query', 'clarification_request', 'context_request']
            
            return IntentClassificationResult(
                intent=best_intent,
                confidence=confidence,
                classification_method="rule_based",
                contextual_metadata={
                    "matched_patterns": matched_patterns.get(best_intent, []),
                    "pattern_count": intent_scores[best_intent]
                },
                entities=entities,
                temporal_scope=temporal_scope,
                is_conversational_response=is_conversational,
                requires_knowledge_search=requires_search
            )
        
        # If no patterns matched, it's likely a knowledge query that we didn't catch
        
        # Default to knowledge query if no patterns match - BE SMART!
        return IntentClassificationResult(
            intent="knowledge_query",
            confidence=0.7,  # HIGHER CONFIDENCE - if user is talking to bot, they probably want help
            classification_method="rule_based",
            contextual_metadata={"reason": "default_to_knowledge_query"},
            entities=self._extract_entities(message_text),
            temporal_scope=self._detect_temporal_scope(message_text),
            is_conversational_response=False,
            requires_knowledge_search=True
        )

    def _is_likely_social_interaction(
        self, 
        text_lower: str, 
        conversation_analysis: ConversationAnalysis, 
        user_profile: UserProfile
    ) -> bool:
        """Enhanced social interaction detection using context and user profile."""
        try:
            # Check for common social patterns
            social_patterns = [
                r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
                r'\b(yo|sup|wassup|what\'s up)\b',
                r'\b(how are you|how\'s it going|how do you do)\b',
                r'\b(thanks?|thank you|thx|ty)\b',
                r'\b(bye|goodbye|see you|catch you later)\b',
                r'\b(great|awesome|cool|nice|good|bad|terrible)\b',
                r'\b(lol|haha|hehe|lmao|rofl)\b'
            ]
            
            # Check if message matches social patterns
            for pattern in social_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return True
            
            # Check if user typically uses social language
            if user_profile and user_profile.common_greetings:
                for greeting in user_profile.common_greetings:
                    if greeting.lower() in text_lower:
                        return True
            
            # Check if conversation context suggests social interaction
            if conversation_analysis:
                # If conversation is in early stage and message is short
                if (conversation_analysis.conversation_stage in ["starting", "developing"] and 
                    len(text_lower.split()) <= 5):
                    return True
                
                # If sentiment is positive and message is casual
                if (conversation_analysis.sentiment_score > 0.3 and 
                    len(text_lower.split()) <= 10):
                    return True
            
            # Check for very short, casual messages
            if len(text_lower.split()) <= 3 and not any(char in text_lower for char in ['?', '!']):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in social interaction detection: {e}")
            return False

    async def _analyze_conversation_context(
        self,
        message_text: str,
        user_id: str,
        channel_id: str,
        workspace_id: int,
        thread_ts: Optional[str],
        db: AsyncSession
    ) -> ConversationContext:
        """Phase 2: Analyze conversation context for better classification."""
        try:
            # Get recent messages in the channel
            recent_messages_query = select(Message).join(Conversation).where(
                and_(
                    Conversation.workspace_id == workspace_id,
                    Conversation.slack_channel_id == channel_id,
                    Message.created_at >= datetime.utcnow() - timedelta(hours=24)
                )
            ).order_by(desc(Message.created_at)).limit(20)
            
            recent_messages_result = await db.execute(recent_messages_query)
            recent_messages = recent_messages_result.scalars().all()
            
            # Get thread messages if in a thread
            thread_messages = []
            if thread_ts:
                thread_query = select(Message).join(Conversation).where(
                    and_(
                        Conversation.workspace_id == workspace_id,
                        Conversation.slack_channel_id == channel_id,
                        Message.thread_ts == thread_ts
                    )
                ).order_by(Message.created_at)
                
                thread_result = await db.execute(thread_query)
                thread_messages = thread_result.scalars().all()
            
            # Analyze participants
            participants = list(set([msg.slack_user_id for msg in recent_messages if msg.slack_user_id]))
            
            # Check if bot was mentioned recently
            bot_mentioned_recently = any(
                'reno' in msg.content.lower() or '@reno' in msg.content.lower()
                for msg in recent_messages[:5]
            )
            
            # Check if this is a direct response to bot
            is_direct_response = self._is_direct_response_to_bot(message_text, recent_messages)
            
            # Build context
            context = ConversationContext(
                thread_messages=[self._message_to_dict(msg) for msg in thread_messages],
                recent_messages=[self._message_to_dict(msg) for msg in recent_messages],
                participants=participants,
                channel_context={"channel_id": channel_id, "workspace_id": workspace_id},
                temporal_context={"message_time": datetime.utcnow().isoformat()},
                bot_mentioned_recently=bot_mentioned_recently,
                is_direct_response=is_direct_response
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Error analyzing conversation context: {e}")
            return ConversationContext(
                thread_messages=[],
                recent_messages=[],
                participants=[],
                channel_context={},
                temporal_context={},
                bot_mentioned_recently=False,
                is_direct_response=False
            )

    async def _contextual_classification(
        self, 
        message_text: str, 
        conversation_analysis: ConversationAnalysis,
        user_profile: Optional[UserProfile] = None
    ) -> IntentClassificationResult:
        """Phase 2: Context-aware classification using conversation history."""
        try:
            # Analyze if this is a conversational response
            if conversation_analysis.is_direct_response:
                # Check for acknowledgment patterns
                if self._is_acknowledgment(message_text):
                    return IntentClassificationResult(
                        intent="conversational_response",
                        confidence=0.8,
                        classification_method="contextual",
                        contextual_metadata={"reason": "direct_response_acknowledgment"},
                        entities=[],
                        temporal_scope=None,
                        is_conversational_response=True,
                        requires_knowledge_search=False
                    )
                
                # Check for clarification requests
                if self._is_clarification_request(message_text):
                    return IntentClassificationResult(
                        intent="clarification_request",
                        confidence=0.8,
                        classification_method="contextual",
                        contextual_metadata={"reason": "direct_response_clarification"},
                        entities=self._extract_entities(message_text),
                        temporal_scope=self._detect_temporal_scope(message_text),
                        is_conversational_response=True,
                        requires_knowledge_search=True
                    )
            
            # Check for context requests
            if self._is_context_request(message_text, conversation_analysis):
                return IntentClassificationResult(
                    intent="context_request",
                    confidence=0.8,
                    classification_method="contextual",
                    contextual_metadata={"reason": "context_request_detected"},
                    entities=self._extract_entities(message_text),
                    temporal_scope=self._detect_temporal_scope(message_text),
                    is_conversational_response=False,
                    requires_knowledge_search=True
                )
            
            # Check for social interaction
            if self._is_social_interaction(message_text, context):
                return IntentClassificationResult(
                    intent="social_interaction",
                    confidence=0.7,
                    classification_method="contextual",
                    contextual_metadata={"reason": "social_interaction_detected"},
                    entities=[],
                    temporal_scope=None,
                    is_conversational_response=False,
                    requires_knowledge_search=False
                )
            
            # Default to knowledge query with context
            return IntentClassificationResult(
                intent="knowledge_query",
                confidence=0.6,
                classification_method="contextual",
                contextual_metadata={"context_analyzed": True},
                entities=self._extract_entities(message_text),
                temporal_scope=self._detect_temporal_scope(message_text),
                is_conversational_response=False,
                requires_knowledge_search=True
            )
            
        except Exception as e:
            logger.error(f"Error in contextual classification: {e}")
            # Fallback to rule-based
            return await self._rule_based_classification(message_text)

    async def _ai_classification(
        self, 
        message_text: str,
        conversation_analysis: Optional[ConversationAnalysis] = None,
        user_profile: Optional[UserProfile] = None
    ) -> IntentClassificationResult:
        """Phase 3: AI-powered classification for ambiguous cases."""
        try:
            # Create enhanced classification prompt with context
            prompt = self._create_enhanced_classification_prompt(
                message_text, conversation_analysis, user_profile
            )
            
            # Call OpenAI service
            response = await self.openai_service.generate_completion(
                prompt=prompt,
                max_completion_tokens=200,
                temperature=1.0
            )
            
            # Parse AI response
            result = self._parse_ai_classification_response(response, message_text)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AI classification: {e}")
            # Fallback to rule-based
            return await self._rule_based_classification(message_text)

    def _extract_entities(self, message_text: str) -> List[str]:
        """Extract entities from message text."""
        entities = []
        
        # Extract user mentions
        user_mentions = re.findall(r'<@[^>]+>', message_text)
        entities.extend(user_mentions)
        
        # Extract channel mentions
        channel_mentions = re.findall(r'<#[^>]+>', message_text)
        entities.extend(channel_mentions)
        
        # Extract URLs
        urls = re.findall(r'https?://[^\s]+', message_text)
        entities.extend(urls)
        
        # Extract potential technical terms (capitalized words)
        tech_terms = re.findall(r'\b[A-Z][a-zA-Z]*(?:[A-Z][a-zA-Z]*)*\b', message_text)
        entities.extend(tech_terms)
        
        return entities

    def _detect_temporal_scope(self, message_text: str) -> Optional[str]:
        """Detect temporal scope in the message."""
        text_lower = message_text.lower()
        
        for indicator in self.temporal_indicators.keys():
            if indicator in text_lower:
                return indicator
        
        return None

    def _is_direct_response_to_bot(self, message_text: str, recent_messages: List[Dict[str, Any]]) -> bool:
        """Check if message is a direct response to bot output."""
        # Check if bot was mentioned in recent messages
        for msg in recent_messages[:3]:  # Check last 3 messages
            if 'reno' in msg.get('content', '').lower():
                # Check timing (within 5 minutes)
                msg_time = msg.get('created_at')
                if msg_time:
                    time_diff = (datetime.utcnow() - msg_time).total_seconds()
                    if time_diff < 300:  # 5 minutes
                        return True
        
        return False

    def _is_acknowledgment(self, message_text: str) -> bool:
        """Check if message is an acknowledgment."""
        text_lower = message_text.lower().strip()
        acknowledgment_patterns = [
            r'^(thanks?|thank you|thx)\b',
            r'^(ok|okay|got it|understood)\b',
            r'^(yes|no|correct|right|wrong)\b',
            r'^(agree|disagree|exactly|precisely)\b'
        ]
        
        for pattern in acknowledgment_patterns:
            if re.match(pattern, text_lower):
                return True
        
        return False

    def _is_clarification_request(self, message_text: str) -> bool:
        """Check if message is a clarification request."""
        text_lower = message_text.lower()
        clarification_patterns = [
            r'\b(can you|could you|would you)\s+(explain|clarify|elaborate|expand)\b',
            r'\b(what do you mean|what does that mean)\b',
            r'\b(more details?|more information|more specific)\b'
        ]
        
        for pattern in clarification_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False

    def _is_context_request(self, message_text: str, context: ConversationContext) -> bool:
        """Check if message is requesting conversation context."""
        text_lower = message_text.lower()
        context_patterns = [
            r'\b(what did we discuss|what was said|what happened)\b',
            r'\b(summary|summarize|recap|overview)\b',
            r'\b(previous|earlier|before|last time)\b'
        ]
        
        for pattern in context_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False

    def _is_social_interaction(self, message_text: str, context: Optional[ConversationContext] = None) -> bool:
        """Check if message is social interaction."""
        text_lower = message_text.lower().strip()
        social_patterns = [
            r'^(hi|hello|hey|good morning|good afternoon|good evening)\b',
            r'^(how are you|how\'s it going|what\'s up)\b',
            r'^(bye|goodbye|see you|catch you later)\b'
        ]
        
        for pattern in social_patterns:
            if re.match(pattern, text_lower):
                return True
        
        return False

    def _message_to_dict(self, message: Message) -> Dict[str, Any]:
        """Convert Message object to dictionary."""
        return {
            'id': message.id,
            'content': message.content,
            'slack_user_id': message.slack_user_id,
            'created_at': message.created_at,
            'thread_ts': message.thread_ts,
            'message_metadata': message.message_metadata
        }

    def _create_classification_prompt(self, message_text: str) -> str:
        """Create prompt for AI classification."""
        return f"""
Classify the following Slack message into one of these intent categories:

1. knowledge_query - Seeking specific information from team knowledge base
2. conversational_response - Responding to bot output rather than requesting new information  
3. clarification_request - Asking for more detail about previous bot responses
4. social_interaction - Casual conversational messages (greetings, thanks, etc.)
5. context_request - Asking about recent conversation context or summaries
6. ignore - Messages not intended for bot processing

Message: "{message_text}"

Respond with only the intent category name and a confidence score (0.0-1.0) in this format:
intent: [category]
confidence: [score]
reason: [brief explanation]
"""

    def _parse_ai_classification_response(self, response: str, message_text: str) -> IntentClassificationResult:
        """Parse AI classification response."""
        try:
            lines = response.strip().split('\n')
            intent = "knowledge_query"
            confidence = 0.5
            reason = "ai_classification"
            
            for line in lines:
                if line.startswith('intent:'):
                    intent = line.split(':', 1)[1].strip()
                elif line.startswith('confidence:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        confidence = 0.5
                elif line.startswith('reason:'):
                    reason = line.split(':', 1)[1].strip()
            
            # Validate intent
            if intent not in self.intent_categories:
                intent = "knowledge_query"
                confidence = 0.3
            
            return IntentClassificationResult(
                intent=intent,
                confidence=confidence,
                classification_method="ai_powered",
                contextual_metadata={"ai_reason": reason},
                entities=self._extract_entities(message_text),
                temporal_scope=self._detect_temporal_scope(message_text),
                is_conversational_response=intent == "conversational_response",
                requires_knowledge_search=intent in ["knowledge_query", "clarification_request", "context_request"]
            )
            
        except Exception as e:
            logger.error(f"Error parsing AI classification response: {e}")
            return IntentClassificationResult(
                intent="knowledge_query",
                confidence=0.3,
                classification_method="ai_fallback",
                contextual_metadata={"error": str(e)},
                entities=self._extract_entities(message_text),
                temporal_scope=self._detect_temporal_scope(message_text),
                is_conversational_response=False,
                requires_knowledge_search=True
            )

    def _create_enhanced_classification_prompt(
        self,
        message_text: str,
        conversation_analysis: Optional[ConversationAnalysis] = None,
        user_profile: Optional[UserProfile] = None
    ) -> str:
        """Create enhanced classification prompt with context and user profile."""
        base_prompt = f"""
Classify the following message into one of these intent categories:

1. knowledge_query - User is asking for specific information, processes, or technical details
2. conversational_response - User is responding to previous bot output (thanks, ok, got it, etc.)
3. clarification_request - User is asking for more details about previous responses
4. social_interaction - Casual greetings, small talk, social conversation
5. context_request - User is asking about conversation history or context
6. ignore - Spam, irrelevant, or meaningless messages

Message: "{message_text}"
"""

        # Add context information
        if conversation_analysis:
            base_prompt += f"""

Context Information:
- Conversation stage: {conversation_analysis.conversation_stage}
- Active topic: {conversation_analysis.active_topic or 'None'}
- Sentiment: {conversation_analysis.sentiment_score:.2f}
- Bot mentioned recently: {conversation_analysis.bot_mentioned_recently}
- Is direct response: {conversation_analysis.is_direct_response}
- Urgency indicators: {', '.join(conversation_analysis.urgency_indicators) or 'None'}
"""

        # Add user profile information
        if user_profile:
            base_prompt += f"""

User Profile:
- Formality level: {user_profile.formality_level:.2f}
- Verbosity preference: {user_profile.verbosity_preference:.2f}
- Common greetings: {', '.join(user_profile.common_greetings) or 'None'}
- Communication culture: {user_profile.communication_culture}
- Preferred response style: {user_profile.preferred_response_style}
"""

        base_prompt += """

Respond with only the intent category name and a confidence score (0.0-1.0) in this format:
intent: [category_name]
confidence: [score]
reasoning: [brief explanation]
"""

        return base_prompt

    async def _record_classification(
        self,
        db: AsyncSession,
        workspace_id: int,
        user_id: str,
        channel_id: str,
        query_id: Optional[int],
        message_text: str,
        result: IntentClassificationResult,
        conversation_analysis: Optional[ConversationAnalysis] = None,
        user_profile: Optional[UserProfile] = None
    ) -> None:
        """Record classification for learning purposes."""
        try:
            # Prepare context data
            conversation_context = {}
            if conversation_analysis:
                conversation_context = {
                    "conversation_stage": conversation_analysis.conversation_stage,
                    "active_topic": conversation_analysis.active_topic,
                    "sentiment_score": conversation_analysis.sentiment_score,
                    "bot_mentioned_recently": conversation_analysis.bot_mentioned_recently,
                    "is_direct_response": conversation_analysis.is_direct_response,
                    "urgency_indicators": conversation_analysis.urgency_indicators,
                    "context_confidence": conversation_analysis.context_confidence
                }
            
            user_context = {}
            if user_profile:
                user_context = {
                    "formality_level": user_profile.formality_level,
                    "verbosity_preference": user_profile.verbosity_preference,
                    "emoji_usage_frequency": user_profile.emoji_usage_frequency,
                    "question_asking_frequency": user_profile.question_asking_frequency,
                    "preferred_response_style": user_profile.preferred_response_style,
                    "communication_culture": user_profile.communication_culture,
                    "learning_confidence": user_profile.learning_confidence,
                    "interaction_count": user_profile.interaction_count
                }
            
            channel_context = {}
            if conversation_analysis:
                channel_context = conversation_analysis.channel_context
            
            # Record the classification
            await self.learning_engine.record_classification(
                db=db,
                workspace_id=workspace_id,
                user_id=user_id,
                channel_id=channel_id,
                query_id=query_id,
                original_message=message_text,
                classified_intent=result.intent,
                confidence_score=result.confidence,
                classification_method=result.classification_method,
                conversation_context=conversation_context,
                user_context=user_context,
                channel_context=channel_context
            )
            
        except Exception as e:
            logger.error(f"Error recording classification: {e}")
