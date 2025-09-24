"""
Simplified Knowledge Extractor

Focuses on extracting actionable knowledge from team conversations:
- Decisions made by the team
- Step-by-step processes shared
- Important information and context
- Who said what and when
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func

from ..models.base import Message, Conversation, KnowledgeItem
from ..services.openai_service import OpenAIService
from ..services.embedding_service import EmbeddingService

class SimpleKnowledgeExtractor:
    """Simplified knowledge extractor focused on team knowledge."""
    
    def __init__(self):
        self.openai_service = OpenAIService()
        self.embedding_service = EmbeddingService()

    async def extract_from_conversations(
        self,
        workspace_id: int,
        db: AsyncSession,
        batch_size: int = 5
    ) -> List[KnowledgeItem]:
        """
        Find conversations and extract knowledge from them.
        Simple approach: look for conversations with multiple messages and participants.
        """
        try:
            logger.info(f"Starting knowledge extraction for workspace {workspace_id}")
            
            # Find conversations that might have extractable knowledge
            conversations = await self._find_conversations_to_process(workspace_id, db, batch_size)
            
            extracted_knowledge = []
            
            for conversation_id in conversations:
                try:
                    knowledge_items = await self._extract_from_single_conversation(
                        conversation_id, db
                    )
                    extracted_knowledge.extend(knowledge_items)
                    
                    # Mark conversation as processed
                    await self._mark_conversation_processed(conversation_id, db)
                    
                except Exception as e:
                    logger.error(f"Error extracting from conversation {conversation_id}: {e}")
                    continue
            
            logger.info(f"Extracted {len(extracted_knowledge)} knowledge items from {len(conversations)} conversations")
            return extracted_knowledge
            
        except Exception as e:
            logger.error(f"Error in conversation knowledge extraction: {e}")
            return []

    async def _find_conversations_to_process(
        self,
        workspace_id: int,
        db: AsyncSession,
        limit: int = 5
    ) -> List[int]:
        """Find conversations that might have extractable knowledge."""
        try:
            # Look for conversations from the last 24 hours with multiple messages
            since = datetime.now(timezone.utc) - timedelta(hours=24)
            
            # Find conversations with at least 3 messages and 2 participants
            query = select(
                Conversation.id,
                func.count(Message.id).label('message_count'),
                func.count(func.distinct(Message.slack_user_id)).label('participant_count')
            ).select_from(
                Conversation
            ).join(
                Message, Message.conversation_id == Conversation.id
            ).where(
                and_(
                    Conversation.workspace_id == workspace_id,
                    Conversation.created_at >= since
                )
            ).group_by(
                Conversation.id
            ).having(
                and_(
                    func.count(Message.id) >= 3,  # At least 3 messages
                    func.count(func.distinct(Message.slack_user_id)) >= 2  # At least 2 people
                )
            ).order_by(
                desc(func.count(Message.id))  # Process conversations with more messages first
            ).limit(limit)
            
            result = await db.execute(query)
            conversations = [row[0] for row in result.fetchall()]
            
            logger.info(f"Found {len(conversations)} conversations to process")
            return conversations
            
        except Exception as e:
            logger.error(f"Error finding conversations to process: {e}")
            return []

    async def _extract_from_single_conversation(
        self,
        conversation_id: int,
        db: AsyncSession
    ) -> List[KnowledgeItem]:
        """Extract knowledge from a single conversation."""
        try:
            # Get all messages in the conversation
            query = select(Message).where(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at)
            
            result = await db.execute(query)
            messages = result.scalars().all()
            
            if len(messages) < 3:  # Skip very short conversations
                return []
            
            # Get conversation info
            conv_query = select(Conversation).where(Conversation.id == conversation_id)
            conv_result = await db.execute(conv_query)
            conversation = conv_result.scalar_one_or_none()
            
            if not conversation:
                return []
            
            # Prepare conversation text
            conversation_text = self._format_conversation_for_ai(messages)
            
            # Extract knowledge using AI
            extraction_result = await self._ai_extract_knowledge(
                conversation_text, conversation, messages
            )
            
            if not extraction_result.get('extractable_knowledge'):
                logger.info(f"No extractable knowledge found in conversation {conversation_id}")
                return []
            
            # Create knowledge items
            knowledge_items = []
            for knowledge_data in extraction_result['extractable_knowledge']:
                try:
                    # Generate embedding for the knowledge
                    embedding = await self.embedding_service.generate_embedding(
                        knowledge_data.get('content', '')
                    )
                    
                    knowledge_item = KnowledgeItem(
                        workspace_id=conversation.workspace_id,
                        conversation_id=conversation_id,
                        knowledge_type=knowledge_data.get('type', 'general'),
                        content=knowledge_data.get('content', ''),
                        confidence_score=knowledge_data.get('confidence', 0.5),
                        source_messages=[msg.id for msg in messages],
                        participants=list(set(msg.slack_user_id for msg in messages)),
                        metadata={
                            'source_channel_id': conversation.slack_channel_id,
                            'source_channel': knowledge_data.get('source_channel', 'unknown'),
                            'created_date': conversation.created_at.isoformat(),
                            'participants': list(set(msg.slack_user_id for msg in messages)),
                            'message_count': len(messages),
                            'extraction_method': 'simple_ai',
                            'summary': knowledge_data.get('summary', ''),
                            'keywords': knowledge_data.get('keywords', [])
                        },
                        embedding=embedding,
                        created_at=datetime.now(timezone.utc)
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

    def _format_conversation_for_ai(self, messages: List[Message]) -> str:
        """Format conversation messages for AI processing."""
        formatted_lines = []
        
        for msg in messages:
            timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M")
            user_id = msg.slack_user_id
            text = msg.text or ""
            
            formatted_lines.append(f"[{timestamp}] {user_id}: {text}")
        
        return "\n".join(formatted_lines)

    async def _ai_extract_knowledge(
        self,
        conversation_text: str,
        conversation: Conversation,
        messages: List[Message]
    ) -> Dict[str, Any]:
        """Use AI to extract knowledge from conversation text."""
        
        system_prompt = """You are a team knowledge extractor. Your job is to identify and extract valuable information from team conversations.

FOCUS ON:
1. Decisions made by the team (what was decided and why)
2. Processes and step-by-step instructions shared
3. Important technical information or solutions
4. Key insights or learnings
5. Action items and commitments

IGNORE:
- Casual chat and greetings
- Off-topic discussions
- Incomplete thoughts or questions without answers

For each piece of extractable knowledge, provide:
- type: "decision", "process", "solution", "insight", or "action_item"
- content: The actual knowledge (be specific and detailed)
- summary: One sentence summary
- confidence: 0.1-1.0 (how confident you are this is valuable knowledge)
- keywords: List of relevant search terms

Return JSON format:
{
  "extractable_knowledge": [
    {
      "type": "decision",
      "content": "The team decided to use PostgreSQL instead of MySQL because...",
      "summary": "Database choice decision",
      "confidence": 0.8,
      "keywords": ["database", "postgresql", "mysql", "decision"]
    }
  ],
  "conversation_summary": "Brief summary of what was discussed"
}

Only extract knowledge if there's genuinely valuable information. Return empty array if it's just casual conversation."""

        try:
            response = await self.openai_service.chat_completion([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract knowledge from this team conversation:\n\n{conversation_text}"}
            ], model="gpt-3.5-turbo", temperature=0.1, max_tokens=1000)
            
            raw_content = response['choices'][0]['message']['content']
            result = self._safe_json_parse(raw_content)
            
            logger.info(f"AI extraction result: {len(result.get('extractable_knowledge', []))} items found")
            return result
            
        except Exception as e:
            logger.error(f"AI knowledge extraction failed: {e}")
            return {"extractable_knowledge": [], "conversation_summary": "Failed to extract knowledge"}

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
                    "conversation_summary": "Failed to parse AI response"
                }

    async def _mark_conversation_processed(self, conversation_id: int, db: AsyncSession):
        """Mark a conversation as processed for knowledge extraction."""
        try:
            # For now, just log it. In a production system, you'd add a field to track this
            logger.info(f"Marked conversation {conversation_id} as processed for knowledge extraction")
            
        except Exception as e:
            logger.error(f"Error marking conversation as processed: {e}")

    async def extract_from_recent_messages(
        self,
        channel_id: str,
        workspace_id: int,
        db: AsyncSession,
        hours_back: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Extract knowledge from recent messages in a channel.
        Used for real-time queries when knowledge base doesn't have the answer.
        """
        try:
            since = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            
            # Get recent messages from the channel
            query = select(Message).where(
                and_(
                    Message.slack_channel_id == channel_id,
                    Message.created_at >= since
                )
            ).order_by(Message.created_at)
            
            result = await db.execute(query)
            messages = result.scalars().all()
            
            if len(messages) < 2:
                return []
            
            # Format for AI analysis
            conversation_text = self._format_conversation_for_ai(messages)
            
            # Quick extraction for recent content
            system_prompt = """Extract key information from this recent team discussion. Focus on:
- Decisions being made
- Solutions being discussed
- Important updates or status changes
- Action items or next steps

Be concise and only extract if there's genuinely useful information.

Return JSON: {"insights": [{"content": "...", "type": "..."}]}"""
            
            response = await self.openai_service.chat_completion([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text}
            ], model="gpt-3.5-turbo", temperature=0.1, max_tokens=500)
            
            result = self._safe_json_parse(response['choices'][0]['message']['content'])
            
            return result.get('insights', [])
            
        except Exception as e:
            logger.error(f"Error extracting from recent messages: {e}")
            return []
