"""
Enhanced Query Processor with multi-modal search and response synthesis.

This implements the sophisticated response generation system that provides
specific, sourced, and actionable responses.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func, text
from loguru import logger

from ..core.database import get_session_factory
from ..models.base import Conversation, Message, KnowledgeItem, User, Workspace, Query
from ..services.openai_service import OpenAIService
from ..services.vector_service import VectorService
from ..workers.celery_app import celery_app


@dataclass
class QueryContext:
    """Rich context about the user's query."""
    query_text: str
    query_type: str  # 'temporal', 'semantic', 'participant', 'hybrid'
    intent: str  # 'status', 'process', 'decision', 'timeline'
    temporal_scope: Optional[str]  # 'today', 'recent', 'yesterday', etc.
    mentioned_participants: List[str]
    mentioned_concepts: List[str]
    specificity_level: str  # 'overview', 'detailed', 'implementation'


@dataclass
class SearchResult:
    """Enhanced search result with source attribution."""
    knowledge_item: Optional[KnowledgeItem]
    conversation: Optional[Conversation]
    messages: Optional[List[Message]]
    relevance_score: float
    source_type: str  # 'knowledge', 'conversation', 'message'
    attribution: Dict[str, Any]
    context_snippet: str


@dataclass
class ResponseSynthesis:
    """Synthesized response with full attribution."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    response_type: str
    related_information: List[str]
    next_steps: List[str]
    verification_links: List[str]


class EnhancedQueryProcessor:
    """
    Sophisticated query processing with multi-modal search and response synthesis.
    
    This replaces simple search with a comprehensive system that understands
    query intent and provides specific, actionable responses.
    """
    
    def __init__(self):
        self.openai_service = OpenAIService()
        self.vector_service = VectorService()
        
        # Query processing configuration
        self.config = {
            'max_search_results': 10,
            'min_confidence_threshold': 0.6,
            'temporal_query_hours': 48,
            'max_response_length': 2000,
            'source_attribution_required': True
        }
        
        # Intent patterns for query classification
        self.intent_patterns = {
            'status': ['status', 'what happened', 'current state', 'progress', 'update'],
            'process': ['how to', 'steps', 'procedure', 'workflow', 'process'],
            'decision': ['why', 'decided', 'chose', 'decision', 'rationale'],
            'timeline': ['when', 'timeline', 'schedule', 'deadline', 'date'],
            'troubleshooting': ['error', 'problem', 'issue', 'broken', 'fix', 'debug'],
            'resource': ['tool', 'service', 'library', 'recommend', 'use']
        }
        
        # Temporal indicators for time-based queries
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
            'this week': 7,
            'last week': 7
        }

    async def process_enhanced_query(
        self,
        query_id: Optional[int],
        workspace_id: int,
        user_id: int,
        channel_id: str,
        query_text: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Process query with enhanced understanding and multi-modal search.
        
        This is the main entry point for the enhanced query processing system.
        """
        try:
            logger.info(f"Processing enhanced query: {query_text}")
            
            # 1. Query Understanding and Decomposition
            query_context = await self._understand_query(query_text, workspace_id, db)
            
            # 2. Multi-Modal Search Strategy
            search_results = await self._execute_multi_modal_search(
                query_context, workspace_id, channel_id, db
            )
            
            # 3. Response Synthesis
            response_synthesis = await self._synthesize_response(
                query_context, search_results, db
            )
            
            # 4. Format Enhanced Response
            formatted_response = await self._format_enhanced_response(
                response_synthesis, query_context
            )
            
            # 5. Store Query and Response
            if query_id:
                await self._store_query_response(
                    query_id, query_context, response_synthesis, db
                )
            
            return {
                "status": "success",
                "response": formatted_response,
                "query_context": {
                    "type": query_context.query_type,
                    "intent": query_context.intent,
                    "temporal_scope": query_context.temporal_scope
                },
                "search_results_count": len(search_results),
                "confidence": response_synthesis.confidence,
                "sources_count": len(response_synthesis.sources)
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced query processing: {e}")
            return {
                "status": "error",
                "response": "I encountered an error while processing your query. Please try again.",
                "error": str(e)
            }

    async def _understand_query(
        self, 
        query_text: str, 
        workspace_id: int, 
        db: AsyncSession
    ) -> QueryContext:
        """Deep analysis of query to understand intent and context."""
        
        try:
            # Classify query intent
            intent = self._classify_intent(query_text)
            
            # Detect temporal scope
            temporal_scope = self._detect_temporal_scope(query_text)
            
            # Extract mentioned participants
            mentioned_participants = await self._extract_mentioned_participants(
                query_text, workspace_id, db
            )
            
            # Extract key concepts
            mentioned_concepts = await self._extract_key_concepts(query_text)
            
            # Determine query type based on analysis
            query_type = self._determine_query_type(
                query_text, temporal_scope, mentioned_participants, mentioned_concepts
            )
            
            # Assess specificity level
            specificity_level = self._assess_specificity_level(query_text)
            
            return QueryContext(
                query_text=query_text,
                query_type=query_type,
                intent=intent,
                temporal_scope=temporal_scope,
                mentioned_participants=mentioned_participants,
                mentioned_concepts=mentioned_concepts,
                specificity_level=specificity_level
            )
            
        except Exception as e:
            logger.error(f"Error understanding query: {e}")
            # Return basic context on error
            return QueryContext(
                query_text=query_text,
                query_type="semantic",
                intent="unknown",
                temporal_scope=None,
                mentioned_participants=[],
                mentioned_concepts=[],
                specificity_level="overview"
            )

    def _classify_intent(self, query_text: str) -> str:
        """Classify the intent of the query."""
        query_lower = query_text.lower()
        
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return "unknown"

    def _detect_temporal_scope(self, query_text: str) -> Optional[str]:
        """Detect temporal scope in the query."""
        query_lower = query_text.lower()
        
        for indicator, days in self.temporal_indicators.items():
            if indicator in query_lower:
                return indicator
        
        return None

    async def _extract_mentioned_participants(
        self, 
        query_text: str, 
        workspace_id: int, 
        db: AsyncSession
    ) -> List[str]:
        """Extract mentioned participants from the query."""
        try:
            # Look for @mentions
            import re
            mentions = re.findall(r'@(\w+)', query_text)
            
            # Look for names mentioned in the query
            query_words = query_text.lower().split()
            
            # Get all users in workspace
            user_query = select(User).where(User.workspace_id == workspace_id)
            result = await db.execute(user_query)
            users = result.scalars().all()
            
            mentioned_users = []
            for user in users:
                user_name_lower = user.name.lower()
                if (user_name_lower in query_words or 
                    any(word in user_name_lower for word in query_words) or
                    user.slack_id in mentions):
                    mentioned_users.append(user.slack_id)
            
            return mentioned_users
            
        except Exception as e:
            logger.error(f"Error extracting mentioned participants: {e}")
            return []

    async def _extract_key_concepts(self, query_text: str) -> List[str]:
        """Extract key concepts and topics from the query."""
        try:
            # Use AI to extract key concepts
            system_prompt = """Extract key concepts, topics, and technical terms from this query.
Return a JSON list of important concepts that would be useful for searching.
Focus on: technical terms, product names, process names, specific topics.

Examples:
- "database migration" → ["database", "migration", "postgresql", "mysql"]
- "deployment process" → ["deployment", "process", "production", "ci/cd"]
- "API authentication" → ["API", "authentication", "auth", "security"]

Return only the JSON array, nothing else."""

            response = await self.openai_service._make_request(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query_text}"}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            concepts = json.loads(response['choices'][0]['message']['content'])
            return concepts if isinstance(concepts, list) else []
            
        except Exception as e:
            logger.error(f"Error extracting key concepts: {e}")
            # Fallback to simple word extraction
            words = query_text.lower().split()
            return [word for word in words if len(word) > 3]

    def _determine_query_type(
        self, 
        query_text: str, 
        temporal_scope: Optional[str],
        mentioned_participants: List[str],
        mentioned_concepts: List[str]
    ) -> str:
        """Determine the appropriate search strategy based on query analysis."""
        
        # Temporal query if time indicators present
        if temporal_scope:
            return "temporal"
        
        # Participant query if specific people mentioned
        if mentioned_participants:
            return "participant"
        
        # Hybrid if multiple strong indicators
        indicators = sum([
            1 if temporal_scope else 0,
            1 if mentioned_participants else 0,
            1 if len(mentioned_concepts) > 2 else 0
        ])
        
        if indicators > 1:
            return "hybrid"
        
        # Default to semantic search
        return "semantic"

    def _assess_specificity_level(self, query_text: str) -> str:
        """Assess how specific vs. general the query is."""
        query_lower = query_text.lower()
        
        detail_indicators = [
            'how to', 'steps', 'exactly', 'specifically', 'details',
            'implementation', 'code', 'configuration', 'setup'
        ]
        
        overview_indicators = [
            'what is', 'overview', 'general', 'about', 'summary',
            'explain', 'understand', 'basic'
        ]
        
        detail_score = sum(1 for indicator in detail_indicators if indicator in query_lower)
        overview_score = sum(1 for indicator in overview_indicators if indicator in query_lower)
        
        if detail_score > overview_score:
            return "detailed"
        elif overview_score > 0:
            return "overview"
        else:
            return "implementation"

    async def _execute_multi_modal_search(
        self,
        query_context: QueryContext,
        workspace_id: int,
        channel_id: str,
        db: AsyncSession
    ) -> List[SearchResult]:
        """Execute multi-modal search strategy based on query context."""
        
        all_results = []
        
        try:
            # Execute different search strategies based on query type
            if query_context.query_type == "temporal":
                temporal_results = await self._temporal_search(
                    query_context, workspace_id, channel_id, db
                )
                all_results.extend(temporal_results)
            
            if query_context.query_type == "participant":
                participant_results = await self._participant_search(
                    query_context, workspace_id, db
                )
                all_results.extend(participant_results)
            
            if query_context.query_type in ["semantic", "hybrid"]:
                semantic_results = await self._semantic_search(
                    query_context, workspace_id, db
                )
                all_results.extend(semantic_results)
            
            # Always include knowledge base search
            knowledge_results = await self._knowledge_base_search(
                query_context, workspace_id, db
            )
            all_results.extend(knowledge_results)
            
            # Deduplicate and rank results
            unique_results = self._deduplicate_results(all_results)
            ranked_results = self._rank_results(unique_results, query_context)
            
            return ranked_results[:self.config['max_search_results']]
            
        except Exception as e:
            logger.error(f"Error in multi-modal search: {e}")
            return []

    async def _temporal_search(
        self,
        query_context: QueryContext,
        workspace_id: int,
        channel_id: str,
        db: AsyncSession
    ) -> List[SearchResult]:
        """Search conversations from specific time periods."""
        
        try:
            # Calculate time range
            if query_context.temporal_scope in self.temporal_indicators:
                hours_back = self.temporal_indicators[query_context.temporal_scope] * 24
            else:
                hours_back = self.config['temporal_query_hours']
            
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            # Search recent conversations
            query = select(Conversation).where(
                and_(
                    Conversation.workspace_id == workspace_id,
                    Conversation.updated_at >= cutoff_time,
                    Conversation.state.in_(["resolved", "paused"])
                )
            ).order_by(desc(Conversation.updated_at)).limit(20)
            
            result = await db.execute(query)
            conversations = result.scalars().all()
            
            search_results = []
            for conv in conversations:
                # Get messages for this conversation
                msg_query = select(Message).where(
                    Message.conversation_id == conv.id
                ).order_by(Message.created_at)
                
                msg_result = await db.execute(msg_query)
                messages = msg_result.scalars().all()
                
                # Calculate relevance based on content match
                relevance = await self._calculate_content_relevance(
                    query_context.query_text, messages
                )
                
                if relevance > 0.3:  # Minimum relevance threshold
                    search_results.append(SearchResult(
                        knowledge_item=None,
                        conversation=conv,
                        messages=messages,
                        relevance_score=relevance,
                        source_type="conversation",
                        attribution=await self._build_conversation_attribution(conv, messages, db),
                        context_snippet=await self._extract_context_snippet(messages, query_context)
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in temporal search: {e}")
            return []

    async def _participant_search(
        self,
        query_context: QueryContext,
        workspace_id: int,
        db: AsyncSession
    ) -> List[SearchResult]:
        """Search for conversations involving specific participants."""
        
        try:
            if not query_context.mentioned_participants:
                return []
            
            # Find conversations with these participants
            participant_conditions = [
                Message.slack_user_id == user_id 
                for user_id in query_context.mentioned_participants
            ]
            
            # Get conversations with messages from these users
            subquery = select(Message.conversation_id).where(
                and_(
                    *participant_conditions
                )
            ).distinct()
            
            query = select(Conversation).where(
                and_(
                    Conversation.workspace_id == workspace_id,
                    Conversation.id.in_(subquery)
                )
            ).order_by(desc(Conversation.updated_at)).limit(15)
            
            result = await db.execute(query)
            conversations = result.scalars().all()
            
            search_results = []
            for conv in conversations:
                # Get relevant messages from these participants
                msg_query = select(Message).where(
                    and_(
                        Message.conversation_id == conv.id,
                        Message.slack_user_id.in_(query_context.mentioned_participants)
                    )
                ).order_by(Message.created_at)
                
                msg_result = await db.execute(msg_query)
                participant_messages = msg_result.scalars().all()
                
                # Calculate relevance
                relevance = await self._calculate_content_relevance(
                    query_context.query_text, participant_messages
                )
                
                if relevance > 0.2:
                    search_results.append(SearchResult(
                        knowledge_item=None,
                        conversation=conv,
                        messages=participant_messages,
                        relevance_score=relevance + 0.2,  # Bonus for participant match
                        source_type="participant",
                        attribution=await self._build_conversation_attribution(conv, participant_messages, db),
                        context_snippet=await self._extract_context_snippet(participant_messages, query_context)
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in participant search: {e}")
            return []

    async def _semantic_search(
        self,
        query_context: QueryContext,
        workspace_id: int,
        db: AsyncSession
    ) -> List[SearchResult]:
        """Semantic search using vector embeddings."""
        
        try:
            # Use vector service for semantic search
            vector_results = await self.vector_service.search_similar(
                query_text=query_context.query_text,
                workspace_id=workspace_id,
                limit=10
            )
            
            search_results = []
            for vector_result in vector_results:
                # Get the knowledge item
                knowledge_query = select(KnowledgeItem).where(
                    KnowledgeItem.id == vector_result['knowledge_item_id']
                )
                knowledge_result = await db.execute(knowledge_query)
                knowledge_item = knowledge_result.scalar_one_or_none()
                
                if knowledge_item:
                    # Get associated conversation
                    conv_query = select(Conversation).where(
                        Conversation.id == knowledge_item.conversation_id
                    )
                    conv_result = await db.execute(conv_query)
                    conversation = conv_result.scalar_one_or_none()
                    
                    search_results.append(SearchResult(
                        knowledge_item=knowledge_item,
                        conversation=conversation,
                        messages=None,
                        relevance_score=vector_result['similarity_score'],
                        source_type="knowledge",
                        attribution=await self._build_knowledge_attribution(knowledge_item, db),
                        context_snippet=knowledge_item.summary or knowledge_item.content[:200]
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    async def _knowledge_base_search(
        self,
        query_context: QueryContext,
        workspace_id: int,
        db: AsyncSession
    ) -> List[SearchResult]:
        """Search the structured knowledge base."""
        
        try:
            # Build search conditions based on query context
            conditions = [KnowledgeItem.workspace_id == workspace_id]
            
            # Add concept-based filtering
            if query_context.mentioned_concepts:
                # Use full-text search on content
                concept_search = ' | '.join(query_context.mentioned_concepts)
                conditions.append(
                    text("to_tsvector('english', content) @@ plainto_tsquery('english', :search)")
                )
            
            # Filter by knowledge type based on intent
            intent_to_type = {
                'process': 'process_definition',
                'decision': 'decision_made',
                'troubleshooting': 'troubleshooting_guide',
                'resource': 'resource_recommendation'
            }
            
            if query_context.intent in intent_to_type:
                conditions.append(
                    KnowledgeItem.knowledge_type == intent_to_type[query_context.intent]
                )
            
            # Execute search
            if query_context.mentioned_concepts:
                query = select(KnowledgeItem).where(
                    and_(*conditions)
                ).params(search=' | '.join(query_context.mentioned_concepts)).limit(10)
            else:
                query = select(KnowledgeItem).where(
                    and_(*conditions)
                ).order_by(desc(KnowledgeItem.confidence_score)).limit(10)
            
            result = await db.execute(query)
            knowledge_items = result.scalars().all()
            
            search_results = []
            for item in knowledge_items:
                # Calculate relevance based on content similarity
                relevance = await self._calculate_knowledge_relevance(
                    query_context, item
                )
                
                if relevance > self.config['min_confidence_threshold']:
                    search_results.append(SearchResult(
                        knowledge_item=item,
                        conversation=None,
                        messages=None,
                        relevance_score=relevance,
                        source_type="knowledge",
                        attribution=await self._build_knowledge_attribution(item, db),
                        context_snippet=item.summary or item.content[:200]
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in knowledge base search: {e}")
            return []

    async def _synthesize_response(
        self,
        query_context: QueryContext,
        search_results: List[SearchResult],
        db: AsyncSession
    ) -> ResponseSynthesis:
        """Synthesize comprehensive response from search results."""
        
        try:
            if not search_results:
                return self._create_no_results_response(query_context)
            
            # Group results by type and relevance
            high_relevance = [r for r in search_results if r.relevance_score > 0.8]
            medium_relevance = [r for r in search_results if 0.6 <= r.relevance_score <= 0.8]
            
            # Use AI to synthesize response
            system_prompt = await self._build_synthesis_prompt(query_context)
            user_prompt = await self._build_synthesis_input(
                query_context, high_relevance, medium_relevance
            )
            
            response = await self.openai_service._make_request(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=self.config['max_response_length']
            )
            
            # Parse structured response
            synthesis_data = await self._parse_synthesis_response(
                response['choices'][0]['message']['content']
            )
            
            # Build source attribution
            sources = await self._build_source_attribution(search_results, db)
            
            # Calculate overall confidence
            confidence = self._calculate_response_confidence(search_results, synthesis_data)
            
            return ResponseSynthesis(
                answer=synthesis_data.get('answer', 'I could not find a clear answer to your question.'),
                sources=sources,
                confidence=confidence,
                response_type=synthesis_data.get('response_type', 'informational'),
                related_information=synthesis_data.get('related_information', []),
                next_steps=synthesis_data.get('next_steps', []),
                verification_links=synthesis_data.get('verification_links', [])
            )
            
        except Exception as e:
            logger.error(f"Error synthesizing response: {e}")
            return self._create_error_response(query_context)

    async def _build_synthesis_prompt(self, query_context: QueryContext) -> str:
        """Build specialized synthesis prompt based on query context."""
        
        base_prompt = f"""You are a team knowledge assistant providing SPECIFIC, ACTIONABLE answers from conversation history.

QUERY CONTEXT:
- Intent: {query_context.intent}
- Type: {query_context.query_type}
- Specificity: {query_context.specificity_level}
- Temporal scope: {query_context.temporal_scope or 'None'}

YOUR MISSION:
Transform search results into immediately useful answers with specific details, not vague summaries.

RESPONSE REQUIREMENTS:
- LEAD WITH THE ANSWER: Start with the specific information they need
- INCLUDE SPECIFICS: Names, dates, tools, exact steps, reasoning
- PROVIDE CONTEXT: Why this matters, what led to this
- CITE SOURCES: Who said what and when
- SUGGEST NEXT STEPS: What the user should do with this information

RESPONSE FORMAT (JSON):
{{
    "answer": "Direct, specific answer to the question with details and attribution",
    "response_type": "decision|process|status|troubleshooting|resource",
    "related_information": ["Additional relevant points"],
    "next_steps": ["Actionable steps the user can take"],
    "verification_links": ["References to original conversations"],
    "confidence_assessment": "Why this answer is reliable"
}}

QUALITY STANDARDS:
❌ AVOID: "The team discussed X" → TOO VAGUE
❌ AVOID: "There was a conversation about Y" → USELESS
✅ PROVIDE: "John (Senior Engineer) decided on March 15th to use PostgreSQL because of better JSON support. Next steps: Migration plan by March 22nd."

"""

        # Add intent-specific guidance
        intent_guidance = {
            'status': "Focus on current state, recent updates, and what's happening now.",
            'process': "Provide step-by-step instructions with clear ownership and timing.",
            'decision': "Explain what was decided, why, by whom, and what happens next.",
            'troubleshooting': "Give diagnostic steps and specific solutions with verification.",
            'resource': "Recommend specific tools with use cases and implementation guidance."
        }
        
        if query_context.intent in intent_guidance:
            base_prompt += f"\nSPECIFIC GUIDANCE: {intent_guidance[query_context.intent]}"
        
        return base_prompt

    async def _build_synthesis_input(
        self,
        query_context: QueryContext,
        high_relevance: List[SearchResult],
        medium_relevance: List[SearchResult]
    ) -> str:
        """Build input for synthesis AI."""
        
        input_parts = [f"QUESTION: {query_context.query_text}\n"]
        
        # Add high relevance results
        if high_relevance:
            input_parts.append("HIGH RELEVANCE SOURCES:")
            for i, result in enumerate(high_relevance[:3], 1):
                source_text = await self._format_result_for_synthesis(result)
                input_parts.append(f"{i}. {source_text}")
            input_parts.append("")
        
        # Add medium relevance results
        if medium_relevance:
            input_parts.append("ADDITIONAL SOURCES:")
            for i, result in enumerate(medium_relevance[:2], 1):
                source_text = await self._format_result_for_synthesis(result)
                input_parts.append(f"{i}. {source_text}")
            input_parts.append("")
        
        input_parts.append("Provide a comprehensive answer using the above sources.")
        
        return '\n'.join(input_parts)

    async def _format_result_for_synthesis(self, result: SearchResult) -> str:
        """Format search result for AI synthesis."""
        
        if result.source_type == "knowledge":
            return f"Knowledge: {result.knowledge_item.title}\n" \
                   f"Content: {result.knowledge_item.content[:500]}\n" \
                   f"Type: {result.knowledge_item.knowledge_type}\n" \
                   f"Confidence: {result.knowledge_item.confidence_score}"
        
        elif result.source_type == "conversation":
            participants = result.attribution.get('participants', [])
            return f"Conversation: {result.conversation.topic or 'Discussion'}\n" \
                   f"Participants: {', '.join(participants)}\n" \
                   f"Context: {result.context_snippet}\n" \
                   f"Date: {result.conversation.updated_at.strftime('%Y-%m-%d')}"
        
        else:
            return f"Source: {result.context_snippet}"

    # Helper methods for search and synthesis

    async def _calculate_content_relevance(
        self, 
        query_text: str, 
        messages: List[Message]
    ) -> float:
        """Calculate relevance score between query and messages."""
        
        if not messages:
            return 0.0
        
        query_words = set(query_text.lower().split())
        
        total_score = 0.0
        for message in messages:
            message_words = set(message.content.lower().split())
            overlap = len(query_words.intersection(message_words))
            score = overlap / max(len(query_words), 1)
            total_score += score
        
        return min(1.0, total_score / len(messages))

    async def _calculate_knowledge_relevance(
        self,
        query_context: QueryContext,
        knowledge_item: KnowledgeItem
    ) -> float:
        """Calculate relevance between query and knowledge item."""
        
        base_score = knowledge_item.confidence_score
        
        # Boost for matching knowledge type
        intent_type_match = {
            'process': 'process_definition',
            'decision': 'decision_made',
            'troubleshooting': 'troubleshooting_guide'
        }
        
        if query_context.intent in intent_type_match:
            if knowledge_item.knowledge_type == intent_type_match[query_context.intent]:
                base_score += 0.2
        
        # Check concept overlap
        content_lower = knowledge_item.content.lower()
        concept_matches = sum(1 for concept in query_context.mentioned_concepts 
                             if concept.lower() in content_lower)
        
        if query_context.mentioned_concepts:
            concept_score = concept_matches / len(query_context.mentioned_concepts)
            base_score = (base_score + concept_score) / 2
        
        return min(1.0, base_score)

    async def _build_conversation_attribution(
        self, 
        conversation: Conversation, 
        messages: List[Message], 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Build attribution information for conversation results."""
        
        # Get participant names
        participant_ids = list(set(msg.slack_user_id for msg in messages))
        participants = []
        
        for user_id in participant_ids:
            user_query = select(User).where(User.slack_id == user_id)
            user_result = await db.execute(user_query)
            user = user_result.scalar_one_or_none()
            participants.append(user.name if user else user_id)
        
        return {
            'conversation_id': conversation.id,
            'channel': conversation.slack_channel_id,
            'participants': participants,
            'date': conversation.updated_at.strftime('%Y-%m-%d %H:%M'),
            'message_count': len(messages),
            'topic': conversation.topic
        }

    async def _build_knowledge_attribution(
        self, 
        knowledge_item: KnowledgeItem, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Build attribution information for knowledge items."""
        
        return {
            'knowledge_id': knowledge_item.id,
            'title': knowledge_item.title,
            'type': knowledge_item.knowledge_type,
            'confidence': knowledge_item.confidence_score,
            'participants': knowledge_item.participants or [],
            'created': knowledge_item.created_at.strftime('%Y-%m-%d'),
            'conversation_id': knowledge_item.conversation_id
        }

    async def _extract_context_snippet(
        self, 
        messages: List[Message], 
        query_context: QueryContext
    ) -> str:
        """Extract relevant context snippet from messages."""
        
        if not messages:
            return ""
        
        # Find most relevant messages
        query_words = set(query_context.query_text.lower().split())
        
        best_message = None
        best_score = 0
        
        for message in messages:
            message_words = set(message.content.lower().split())
            overlap = len(query_words.intersection(message_words))
            if overlap > best_score:
                best_score = overlap
                best_message = message
        
        if best_message:
            return best_message.content[:200] + "..." if len(best_message.content) > 200 else best_message.content
        
        return messages[0].content[:200] + "..."

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results."""
        seen = set()
        unique_results = []
        
        for result in results:
            # Create identifier based on source
            if result.knowledge_item:
                identifier = f"knowledge_{result.knowledge_item.id}"
            elif result.conversation:
                identifier = f"conversation_{result.conversation.id}"
            else:
                identifier = f"other_{hash(result.context_snippet)}"
            
            if identifier not in seen:
                seen.add(identifier)
                unique_results.append(result)
        
        return unique_results

    def _rank_results(
        self, 
        results: List[SearchResult], 
        query_context: QueryContext
    ) -> List[SearchResult]:
        """Rank results by relevance and other factors."""
        
        def ranking_score(result: SearchResult) -> float:
            score = result.relevance_score
            
            # Boost recent results for temporal queries
            if query_context.temporal_scope and result.conversation:
                hours_ago = (datetime.utcnow() - result.conversation.updated_at).total_seconds() / 3600
                if hours_ago < 24:
                    score += 0.2
            
            # Boost knowledge items for process/decision queries
            if query_context.intent in ['process', 'decision'] and result.knowledge_item:
                score += 0.1
            
            return score
        
        return sorted(results, key=ranking_score, reverse=True)

    async def _parse_synthesis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI synthesis response."""
        try:
            # Try to parse as JSON
            if response_text.strip().startswith('{'):
                return json.loads(response_text)
            else:
                # Extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    # Fallback to treating as plain text answer
                    return {
                        'answer': response_text,
                        'response_type': 'informational',
                        'related_information': [],
                        'next_steps': [],
                        'verification_links': []
                    }
        except Exception as e:
            logger.error(f"Error parsing synthesis response: {e}")
            return {
                'answer': response_text[:500],
                'response_type': 'informational',
                'related_information': [],
                'next_steps': [],
                'verification_links': []
            }

    async def _build_source_attribution(
        self, 
        search_results: List[SearchResult], 
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Build comprehensive source attribution."""
        
        sources = []
        for result in search_results[:5]:  # Top 5 sources
            if result.relevance_score > 0.5:
                sources.append(result.attribution)
        
        return sources

    def _calculate_response_confidence(
        self, 
        search_results: List[SearchResult], 
        synthesis_data: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence in the response."""
        
        if not search_results:
            return 0.1
        
        # Average relevance of top results
        top_results = search_results[:3]
        avg_relevance = sum(r.relevance_score for r in top_results) / len(top_results)
        
        # Boost confidence if we have knowledge items
        knowledge_boost = 0.1 if any(r.knowledge_item for r in top_results) else 0.0
        
        # Boost confidence if answer is detailed
        answer_length = len(synthesis_data.get('answer', ''))
        length_boost = min(0.1, answer_length / 1000)
        
        confidence = avg_relevance + knowledge_boost + length_boost
        return min(0.95, max(0.1, confidence))

    def _create_no_results_response(self, query_context: QueryContext) -> ResponseSynthesis:
        """Create response when no results found."""
        return ResponseSynthesis(
            answer=f"I couldn't find any information about '{query_context.query_text}' in your team's conversations. This might be a new topic or it might not have been discussed in channels I have access to.",
            sources=[],
            confidence=0.1,
            response_type="no_results",
            related_information=[],
            next_steps=["Try rephrasing your question", "Ask in a team channel if this is a new topic"],
            verification_links=[]
        )

    def _create_error_response(self, query_context: QueryContext) -> ResponseSynthesis:
        """Create response when processing fails."""
        return ResponseSynthesis(
            answer="I encountered an error while processing your question. Please try again or rephrase your query.",
            sources=[],
            confidence=0.1,
            response_type="error",
            related_information=[],
            next_steps=["Try rephrasing your question", "Contact support if the issue persists"],
            verification_links=[]
        )

    async def _format_enhanced_response(
        self, 
        synthesis: ResponseSynthesis, 
        query_context: QueryContext
    ) -> Dict[str, Any]:
        """Format the final enhanced response."""
        
        # Build Slack-formatted response
        response_blocks = []
        
        # Main answer
        response_blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Answer:*\n{synthesis.answer}"
            }
        })
        
        # Sources section
        if synthesis.sources:
            source_text = "\n".join([
                f"• {source.get('title', 'Conversation')} ({source.get('date', 'Unknown date')})"
                for source in synthesis.sources[:3]
            ])
            
            response_blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Sources:*\n{source_text}"
                }
            })
        
        # Next steps
        if synthesis.next_steps:
            steps_text = "\n".join([f"• {step}" for step in synthesis.next_steps[:3]])
            response_blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Next Steps:*\n{steps_text}"
                }
            })
        
        return {
            "text": synthesis.answer,
            "blocks": response_blocks,
            "response_type": "in_channel" if synthesis.confidence > 0.7 else "ephemeral"
        }

    async def _store_query_response(
        self,
        query_id: int,
        query_context: QueryContext,
        synthesis: ResponseSynthesis,
        db: AsyncSession
    ):
        """Store query and response for learning."""
        
        try:
            # Update query record with response
            from sqlalchemy import update
            
            update_stmt = update(Query).where(Query.id == query_id).values(
                response={
                    'answer': synthesis.answer,
                    'confidence': synthesis.confidence,
                    'sources_count': len(synthesis.sources),
                    'response_type': synthesis.response_type,
                    'query_context': {
                        'type': query_context.query_type,
                        'intent': query_context.intent,
                        'temporal_scope': query_context.temporal_scope
                    },
                    'processed_at': datetime.utcnow().isoformat()
                }
            )
            
            await db.execute(update_stmt)
            
        except Exception as e:
            logger.error(f"Error storing query response: {e}")


# Celery task wrapper
@celery_app.task
def process_enhanced_query_task(
    query_id: Optional[int],
    workspace_id: int,
    user_id: int,
    channel_id: str,
    query_text: str
):
    """Celery task for enhanced query processing."""
    try:
        logger.info(f"Starting enhanced query processing task for query: {query_text}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                process_enhanced_query_task_async(query_id, workspace_id, user_id, channel_id, query_text)
            )
            return result
        finally:
            loop.close()
            asyncio.set_event_loop(None)
            
    except Exception as e:
        logger.error(f"Error in enhanced query processing task: {e}", exc_info=True)
        raise


async def process_enhanced_query_task_async(
    query_id: Optional[int],
    workspace_id: int,
    user_id: int,
    channel_id: str,
    query_text: str
) -> Dict[str, Any]:
    """Async wrapper for enhanced query processing."""
    
    session_factory = get_session_factory()
    async with session_factory() as db:
        try:
            processor = EnhancedQueryProcessor()
            result = await processor.process_enhanced_query(
                query_id, workspace_id, user_id, channel_id, query_text, db
            )
            
            await db.commit()
            return result
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error in async enhanced query processing: {e}")
            raise
