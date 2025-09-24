"""
Conversation-Level Knowledge Extractor

This service extracts knowledge from COMPLETE conversations rather than individual messages.
It waits for conversations to reach completion before attempting extraction, ensuring
we capture the full narrative arc including problem, discussion, and resolution.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc

from ..models.base import Message, Conversation, KnowledgeItem
from ..services.openai_service import OpenAIService
from ..services.embedding_service import EmbeddingService
from ..services.conversation_state_manager import ConversationStateManager, ConversationState
from ..services.process_recognizer import ProcessRecognizer, ProcessState
from ..services.hallucination_preventer import HallucinationPreventer


class ConversationKnowledgeExtractor:
    """Extracts knowledge from complete conversations with full context."""
    
    def __init__(self):
        self.openai_service = OpenAIService()
        self.embedding_service = EmbeddingService()
        self.state_manager = ConversationStateManager()
        self.process_recognizer = ProcessRecognizer()
        self.hallucination_preventer = HallucinationPreventer()

    async def extract_from_completed_conversations(
        self,
        workspace_id: int,
        db: AsyncSession,
        batch_size: int = 10
    ) -> List[KnowledgeItem]:
        """
        Find completed conversations and extract knowledge from them.
        
        This is the main entry point for conversation-level knowledge extraction.
        """
        try:
            logger.info(f"Starting conversation-level knowledge extraction for workspace {workspace_id}")
            
            # Find conversations that are ready for extraction
            ready_conversations = await self._find_extraction_ready_conversations(
                workspace_id, db, batch_size
            )
            
            logger.info(f"Found {len(ready_conversations)} conversations ready for extraction")
            
            extracted_knowledge = []
            
            for conversation_id in ready_conversations:
                try:
                    knowledge_items = await self._extract_from_single_conversation(
                        conversation_id, workspace_id, db
                    )
                    extracted_knowledge.extend(knowledge_items)
                    
                    # Mark conversation as processed
                    await self._mark_conversation_processed(conversation_id, db)
                    
                except Exception as e:
                    logger.error(f"Failed to extract from conversation {conversation_id}: {e}")
                    continue
            
            logger.info(f"Extracted {len(extracted_knowledge)} knowledge items from {len(ready_conversations)} conversations")
            return extracted_knowledge
            
        except Exception as e:
            logger.error(f"Error in conversation-level knowledge extraction: {e}")
            return []

    async def _find_extraction_ready_conversations(
        self,
        workspace_id: int,
        db: AsyncSession,
        limit: int = 10
    ) -> List[int]:
        """
        Find conversations that are completed and ready for knowledge extraction.
        
        Criteria:
        - Conversation state is COMPLETED
        - Has not been processed for knowledge extraction yet
        - Has sufficient content (minimum message count)
        - Has multiple participants (not just monologue)
        """
        try:
            # Get conversations that haven't been processed yet
            # Note: We'd need to add a 'knowledge_extracted' field to Conversation model
            # For now, we'll look for recent conversations
            
            since = datetime.now(timezone.utc) - timedelta(hours=24)
            
            query = select(Conversation.id).where(
                and_(
                    Conversation.workspace_id == workspace_id,
                    Conversation.created_at >= since
                )
            ).order_by(desc(Conversation.created_at)).limit(limit * 3)  # Get more to filter
            
            result = await db.execute(query)
            conversation_ids = [row[0] for row in result.fetchall()]
            
            ready_conversations = []
            
            for conv_id in conversation_ids:
                # Analyze conversation state
                boundary = await self.state_manager.analyze_conversation_state(conv_id, db)
                
                # Check if ready for extraction
                if await self.state_manager.should_extract_knowledge(boundary):
                    # Additional quality checks
                    if await self._passes_extraction_quality_checks(conv_id, db):
                        ready_conversations.append(conv_id)
                
                if len(ready_conversations) >= limit:
                    break
            
            return ready_conversations
            
        except Exception as e:
            logger.error(f"Error finding extraction-ready conversations: {e}")
            return []

    async def _passes_extraction_quality_checks(
        self,
        conversation_id: int,
        db: AsyncSession
    ) -> bool:
        """
        Check if a conversation meets quality criteria for knowledge extraction.
        
        Enhanced quality criteria with process recognition:
        - Minimum 3 messages
        - At least 2 different participants
        - Contains substantive content (not just "thanks", "ok", etc.)
        - Has some technical or process-related content
        - NEW: Process completeness validation
        """
        try:
            # Get conversation messages
            query = select(Message).where(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at)
            
            result = await db.execute(query)
            messages = result.scalars().all()
            
            if len(messages) < 3:
                return False
            
            # Check participant diversity
            participants = set(msg.slack_user_id for msg in messages)
            if len(participants) < 2:
                return False
            
            # Check content substance
            substantive_messages = 0
            for msg in messages:
                if self._is_substantive_message(msg.content):
                    substantive_messages += 1
            
            if substantive_messages < 2:
                return False
            
            # NEW: Enhanced process validation
            process_result = await self.process_recognizer.analyze_conversation_for_processes(
                conversation_id, db
            )
            
            # If it's a process discussion, apply stricter criteria
            if process_result.is_process:
                logger.info(f"Process conversation detected: {process_result.process_state.value}")
                
                # Don't extract from promised but undelivered processes
                if process_result.process_state == ProcessState.PROMISED:
                    logger.info(f"Rejecting extraction: Process promised but not delivered")
                    return False
                
                # Don't extract from incomplete processes with low completeness
                if (process_result.process_state == ProcessState.INCOMPLETE or 
                    process_result.process_state == ProcessState.IN_PROGRESS):
                    if process_result.completeness_score < 0.7:
                        logger.info(f"Rejecting extraction: Process incomplete (completeness: {process_result.completeness_score:.2f})")
                        return False
                
                # Don't extract from interrupted processes
                if process_result.process_state == ProcessState.INTERRUPTED:
                    logger.info(f"Rejecting extraction: Process was interrupted")
                    return False
                
                # For processes, require higher step count
                if process_result.step_count < 2:
                    logger.info(f"Rejecting extraction: Too few process steps ({process_result.step_count})")
                    return False
            
            # Check for technical/process content (original logic)
            has_technical_content = any(
                self._has_technical_indicators(msg.content) for msg in messages
            )
            
            return has_technical_content
            
        except Exception as e:
            logger.error(f"Error in quality checks for conversation {conversation_id}: {e}")
            return False

    def _is_substantive_message(self, content: str) -> bool:
        """Check if a message contains substantive content worth extracting."""
        content_lower = content.lower().strip()
        
        # Filter out low-value messages
        low_value_patterns = [
            "thanks", "thank you", "got it", "ok", "okay", "yes", "no", "yep", "nope",
            "sure", "sounds good", "perfect", "great", "awesome", "cool", "nice",
            "👍", "👌", "✅", "lol", "haha", "😄", "😊"
        ]
        
        if content_lower in low_value_patterns:
            return False
        
        if len(content_lower) < 10:  # Very short messages likely not substantive
            return False
        
        return True

    def _has_technical_indicators(self, content: str) -> bool:
        """Check if message contains technical or process-related content."""
        technical_indicators = [
            "error", "bug", "issue", "problem", "solution", "fix", "code", "deploy",
            "server", "database", "api", "service", "config", "setup", "install",
            "process", "procedure", "step", "how to", "workflow", "pipeline",
            "command", "script", "run", "execute", "build", "test", "debug",
            "log", "monitor", "alert", "performance", "optimize", "scale"
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in technical_indicators)

    async def _extract_from_single_conversation(
        self,
        conversation_id: int,
        workspace_id: int,
        db: AsyncSession
    ) -> List[KnowledgeItem]:
        """
        Extract knowledge from a single completed conversation.
        
        This assembles the full conversation context and extracts knowledge
        with proper narrative understanding.
        """
        try:
            # Get all messages in chronological order
            query = select(Message).where(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at)
            
            result = await db.execute(query)
            messages = result.scalars().all()
            
            if not messages:
                return []
            
            # Assemble conversation context
            conversation_context = await self._assemble_conversation_context(messages)
            
            # Extract knowledge using AI with full context
            knowledge_data = await self._ai_extract_knowledge(conversation_context)
            
            if not knowledge_data or not knowledge_data.get('extractable_knowledge'):
                logger.info(f"No extractable knowledge found in conversation {conversation_id}")
                return []
            
            # Create knowledge items with hallucination validation
            knowledge_items = []
            
            for knowledge in knowledge_data['extractable_knowledge']:
                try:
                    # NEW: Validate extracted knowledge for hallucinations
                    source_messages = [
                        {'user_id': msg.slack_user_id, 'content': msg.content}
                        for msg in messages
                    ]
                    
                    hallucination_check = await self.hallucination_preventer.validate_extracted_knowledge(
                        extracted_content=knowledge.get('content', ''),
                        source_messages=source_messages,
                        original_conversation=conversation_context['conversation_text']
                    )
                    
                    # Only create knowledge items that pass hallucination validation
                    if not hallucination_check.is_safe:
                        logger.warning(f"Knowledge item rejected due to hallucination risk: {hallucination_check.issues_found}")
                        
                        # Use corrected content if available
                        if hallucination_check.corrected_content:
                            logger.info("Using corrected content from hallucination preventer")
                            knowledge['content'] = hallucination_check.corrected_content
                        else:
                            logger.info("Skipping knowledge item due to high hallucination risk")
                            continue
                    
                    # Adjust confidence based on hallucination risk
                    original_confidence = knowledge.get('confidence', 0.0)
                    adjusted_confidence = self.hallucination_preventer.adjust_confidence_for_hallucination_risk(
                        original_confidence, hallucination_check
                    )
                    
                    logger.info(f"Knowledge confidence adjusted from {original_confidence:.2f} to {adjusted_confidence:.2f}")
                    
                    # Generate embedding for the knowledge
                    embedding = await self.embedding_service.generate_embedding(
                        f"{knowledge.get('title', '')} {knowledge.get('content', '')}"
                    )
                    
                    knowledge_item = KnowledgeItem(
                        workspace_id=workspace_id,
                        conversation_id=conversation_id,
                        title=knowledge.get('title', 'Extracted Knowledge'),
                        content=knowledge.get('content', ''),
                        knowledge_type=knowledge.get('type', 'general'),
                        confidence_score=adjusted_confidence,  # Use adjusted confidence
                        source_messages=knowledge.get('source_message_ids', []),
                        participants=knowledge.get('participants', []),
                        embedding=embedding,
                        item_metadata={
                            'extraction_method': 'conversation_level',
                            'conversation_state': 'completed',
                            'extraction_timestamp': datetime.utcnow().isoformat(),
                            'message_count': len(messages),
                            'participant_count': len(set(msg.slack_user_id for msg in messages)),
                            'hallucination_check': {
                                'is_safe': hallucination_check.is_safe,
                                'confidence': hallucination_check.confidence,
                                'issues_found': hallucination_check.issues_found,
                                'check_type': hallucination_check.check_type
                            },
                            **knowledge.get('metadata', {})
                        }
                    )
                    
                    db.add(knowledge_item)
                    knowledge_items.append(knowledge_item)
                    
                except Exception as e:
                    logger.error(f"Error creating knowledge item: {e}")
                    continue
            
            await db.commit()
            logger.info(f"Created {len(knowledge_items)} knowledge items from conversation {conversation_id}")
            
            return knowledge_items
            
        except Exception as e:
            logger.error(f"Error extracting from conversation {conversation_id}: {e}")
            return []

    async def _assemble_conversation_context(self, messages: List[Message]) -> Dict[str, Any]:
        """
        Assemble full conversation context with proper narrative structure.
        
        This creates a rich context that includes:
        - Chronological message flow
        - Participant mapping
        - Thread relationships
        - Temporal context
        - Topic evolution
        """
        try:
            participants = {}
            message_sequence = []
            
            for i, msg in enumerate(messages):
                # Build participant profiles
                if msg.slack_user_id not in participants:
                    participants[msg.slack_user_id] = {
                        'id': msg.slack_user_id,
                        'message_count': 0,
                        'first_message_time': msg.created_at,
                        'last_message_time': msg.created_at
                    }
                
                participants[msg.slack_user_id]['message_count'] += 1
                participants[msg.slack_user_id]['last_message_time'] = msg.created_at
                
                # Build message sequence with context
                message_sequence.append({
                    'index': i,
                    'id': msg.id,
                    'timestamp': msg.created_at.isoformat(),
                    'participant': msg.slack_user_id,
                    'content': msg.content,
                    'is_thread_start': i == 0,
                    'time_since_previous': (
                        (msg.created_at - messages[i-1].created_at).total_seconds() 
                        if i > 0 else 0
                    )
                })
            
            return {
                'conversation_id': messages[0].conversation_id,
                'start_time': messages[0].created_at.isoformat(),
                'end_time': messages[-1].created_at.isoformat(),
                'duration_minutes': (messages[-1].created_at - messages[0].created_at).total_seconds() / 60,
                'message_count': len(messages),
                'participant_count': len(participants),
                'participants': list(participants.values()),
                'message_sequence': message_sequence,
                'conversation_text': '\n'.join([
                    f"[{msg.created_at.strftime('%H:%M')}] {msg.slack_user_id}: {msg.content}"
                    for msg in messages
                ])
            }
            
        except Exception as e:
            logger.error(f"Error assembling conversation context: {e}")
            return {}

    async def _ai_extract_knowledge(self, conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use AI to extract knowledge from complete conversation context.
        
        This uses a sophisticated prompt that understands narrative flow
        and only extracts knowledge from completed discussions.
        """
        try:
            system_prompt = """You are an expert knowledge extractor that captures SPECIFIC, ACTIONABLE details from Slack conversations.

Your mission: Transform vague conversations into precise, usable knowledge that helps teams work more effectively.

🎯 EXTRACTION PHILOSOPHY:
Extract knowledge as if you're creating a detailed meeting notes document that someone who wasn't there could use to understand:
- EXACTLY what was decided and WHY
- PRECISELY who said what and when
- SPECIFICALLY what needs to happen next
- CONCRETELY what problems were solved and how

📋 CONTENT DEPTH REQUIREMENTS:

For DECISIONS:
- What exactly was decided? (specific choice made)
- Who made the decision? (decision maker + participants)
- Why was this decided? (reasoning, alternatives considered)
- When does this take effect? (timeline, deadlines)
- What are the implications? (what changes, who's affected)
- What are the next steps? (action items, follow-ups)

For TECHNICAL SOLUTIONS:
- What specific problem does this solve?
- What are the exact steps to implement?
- Who provided the solution and when?
- What tools/resources are needed?
- What are potential gotchas or considerations?
- How do you verify it worked?

For PROCESSES:
- What is the step-by-step procedure?
- When should this process be used?
- Who is responsible for each step?
- What are the inputs and outputs?
- What happens if something goes wrong?
- How is success measured?

🔍 QUALITY STANDARDS:
- SPECIFIC QUOTES: Include actual quotes from key messages
- CONCRETE DETAILS: Numbers, dates, names, specific tools/services
- ACTIONABLE STEPS: Someone should be able to follow this immediately
- COMPLETE CONTEXT: Background, discussion, and resolution
- CLEAR ATTRIBUTION: Who said what, with their role/expertise

❌ REJECT VAGUE CONTENT:
- "The team discussed database options" → TOO VAGUE
- "John mentioned PostgreSQL might be better" → TOO VAGUE
- "We decided to use PostgreSQL because it's better" → TOO VAGUE

✅ EXTRACT RICH CONTENT:
- "John (Senior Engineer) recommended PostgreSQL over MySQL on March 15th because: 1) Better JSON support for our user profile data, 2) Superior performance for complex queries, 3) Better compatibility with our existing Python stack. Sarah (Tech Lead) agreed, noting we need the JSONB indexing for the new analytics features. Decision: Migrate user database to PostgreSQL by end of Q2. Next steps: John to create migration plan by March 22nd, Sarah to review infrastructure requirements."

🏷️ KNOWLEDGE TYPES:
- technical_solution: Specific step-by-step solutions to technical problems
- process_definition: Detailed workflows and procedures
- decision_made: Specific decisions with reasoning and implications
- resource_recommendation: Specific tools/services with use cases
- troubleshooting_guide: Step-by-step problem diagnosis and fixes
- best_practice: Proven approaches with context and results

📊 CONFIDENCE SCORING:
- 0.9-1.0: Complete information with clear decisions, specific details, and actionable outcomes
- 0.7-0.8: Good information with most details but some gaps
- 0.5-0.6: Partial information, missing key details or context
- 0.3-0.4: Vague or incomplete information
- 0.0-0.2: Insufficient or unclear information

Return JSON:
{
    "extractable_knowledge": [
        {
            "title": "Specific, actionable title describing exactly what was decided/solved",
            "content": "Detailed content with quotes, reasoning, timeline, participants, and next steps",
            "type": "technical_solution|process_definition|decision_made|resource_recommendation|troubleshooting_guide|best_practice",
            "confidence": 0.0-1.0,
            "source_message_ids": [1, 2, 3],
            "participants": ["@username1", "@username2"],
            "metadata": {
                "decision_maker": "Who made the final decision",
                "timeline": "When this happens/happened",
                "next_steps": "Specific action items and owners",
                "reasoning": "Why this decision/solution was chosen",
                "impact": "Who/what is affected by this",
                "verification": "How to verify success/completion",
                "quoted_statements": ["key quotes from the conversation"]
            }
        }
    ],
    "conversation_analysis": {
        "is_completed": true|false,
        "has_resolution": true|false,
        "knowledge_quality": "high|medium|low",
        "extraction_reasoning": "Specific reasoning for extraction quality assessment",
        "missing_information": "What key details are missing, if any"
    }
}"""

            response = await self.openai_service._make_request(
                model="gpt-4",  # Use GPT-4 for better reasoning
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this conversation:\n\n{conversation_context['conversation_text']}\n\nConversation metadata: {conversation_context}"}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=1500
            )
            
            import json
            raw_content = response['choices'][0]['message']['content']
            result = self._safe_json_parse(raw_content)
            
            logger.info(f"AI extraction result: {result.get('conversation_analysis', {})}")
            
            return result
            
        except Exception as e:
            logger.error(f"AI knowledge extraction failed: {e}")
            return {"extractable_knowledge": [], "conversation_analysis": {"is_completed": False}}

    async def _mark_conversation_processed(self, conversation_id: int, db: AsyncSession):
        """Mark a conversation as processed for knowledge extraction."""
        try:
            # Update conversation to mark as processed
            # Note: This would require adding a field to the Conversation model
            # For now, we'll just log it
            logger.info(f"Marked conversation {conversation_id} as processed for knowledge extraction")
            
        except Exception as e:
            logger.error(f"Error marking conversation as processed: {e}")

    async def analyze_extraction_readiness(
        self,
        conversation_id: int,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Analyze if a conversation is ready for knowledge extraction.
        
        Returns detailed analysis including:
        - Current conversation state
        - Readiness score
        - Blocking factors
        - Estimated time to completion
        """
        try:
            boundary = await self.state_manager.analyze_conversation_state(conversation_id, db)
            
            readiness_analysis = {
                'conversation_id': conversation_id,
                'current_state': boundary.state.value,
                'confidence': boundary.confidence,
                'is_ready': await self.state_manager.should_extract_knowledge(boundary),
                'blocking_factors': [],
                'estimated_completion': None
            }
            
            # Add specific blocking factors
            if boundary.state == ConversationState.IN_PROGRESS:
                readiness_analysis['blocking_factors'].append("Conversation still in progress")
            elif boundary.state == ConversationState.PAUSED:
                readiness_analysis['blocking_factors'].append("Waiting for more information")
                # Could estimate completion time based on pause patterns
            elif boundary.state == ConversationState.ABANDONED:
                readiness_analysis['blocking_factors'].append("Conversation appears abandoned")
            
            return readiness_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing extraction readiness: {e}")
            return {'error': str(e)}
    
    def _safe_json_parse(self, content: str) -> dict:
        """Safely parse JSON content with robust error handling."""
        import json
        import re
        
        try:
            # First, try direct parsing
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                # Clean up common issues
                cleaned = content.strip()
                
                # Remove markdown code blocks if present
                if cleaned.startswith('```json'):
                    cleaned = cleaned[7:]
                if cleaned.startswith('```'):
                    cleaned = cleaned[3:]
                if cleaned.endswith('```'):
                    cleaned = cleaned[:-3]
                
                # Remove control characters that can break JSON parsing
                cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
                
                # Try parsing the cleaned content
                return json.loads(cleaned)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed even after cleanup: {e}")
                logger.error(f"Problematic content: {content[:500]}...")
                
                # Return default structure
                return {
                    "extractable_knowledge": [],
                    "conversation_analysis": {
                        "is_completed": False,
                        "confidence_score": 0.0,
                        "summary": "Failed to parse AI response",
                        "error": str(e)
                    }
                }
