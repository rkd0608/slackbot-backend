"""
Enhanced Knowledge Extractor with conversation-level processing and multi-stage verification.

This implements the core improvement: extracting knowledge from complete conversations
with proper verification and quality gates.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from loguru import logger

from ..core.database import get_session_factory
from ..models.base import Conversation, Message, KnowledgeItem, User
from ..services.openai_service import OpenAIService
from ..services.enhanced_conversation_state_manager import ConversationState
from ..workers.celery_app import celery_app


@dataclass
class ExtractionCandidate:
    """Represents a potential knowledge item before verification."""
    title: str
    content: str
    knowledge_type: str
    confidence: float
    source_message_ids: List[int]
    participants: List[str]
    metadata: Dict[str, Any]


@dataclass
class VerificationResult:
    """Result of knowledge extraction verification."""
    is_valid: bool
    is_complete: bool
    is_actionable: bool
    confidence_adjustment: float
    issues_found: List[str]
    corrected_content: Optional[str]
    quality_score: float


class EnhancedKnowledgeExtractor:
    """
    Sophisticated knowledge extraction with conversation-level analysis.
    
    This replaces the simple message-level extraction with a comprehensive
    system that processes complete conversations.
    """
    
    def __init__(self):
        self.openai_service = OpenAIService()
        
        # Quality thresholds
        self.config = {
            'min_confidence_threshold': 0.7,
            'min_quality_score': 0.6,
            'max_extraction_attempts': 3,
            'min_content_words': 10,
            'max_content_words': 2000
        }
        
        # Knowledge type configurations
        self.knowledge_types = {
            'technical_solution': {
                'required_fields': ['problem', 'solution', 'steps', 'verification'],
                'min_confidence': 0.8
            },
            'process_definition': {
                'required_fields': ['process_name', 'trigger', 'steps', 'owner'],
                'min_confidence': 0.7
            },
            'decision_made': {
                'required_fields': ['decision', 'reasoning', 'decision_maker', 'timeline'],
                'min_confidence': 0.8
            },
            'resource_recommendation': {
                'required_fields': ['resource', 'use_case', 'benefits', 'considerations'],
                'min_confidence': 0.6
            },
            'troubleshooting_guide': {
                'required_fields': ['problem', 'diagnosis', 'solution', 'prevention'],
                'min_confidence': 0.7
            },
            'best_practice': {
                'required_fields': ['practice', 'context', 'benefits', 'implementation'],
                'min_confidence': 0.6
            }
        }

    async def extract_knowledge_from_complete_conversation(
        self,
        conversation_id: int,
        workspace_id: int,
        db: AsyncSession
    ) -> List[KnowledgeItem]:
        """
        Extract knowledge from a complete conversation using multi-stage verification.
        
        This is the main entry point for the enhanced extraction system.
        """
        try:
            logger.info(f"Starting enhanced knowledge extraction for conversation {conversation_id}")
            
            # 1. Verify conversation is ready for extraction
            if not await self._verify_extraction_readiness(conversation_id, db):
                logger.info(f"Conversation {conversation_id} not ready for extraction")
                return []
            
            # 2. Assemble conversation context
            context = await self._assemble_conversation_context(conversation_id, db)
            if not context:
                logger.warning(f"Could not assemble context for conversation {conversation_id}")
                return []
            
            # 3. Stage 1: Initial Extraction
            candidates = await self._stage1_initial_extraction(context)
            if not candidates:
                logger.info(f"No knowledge candidates found in conversation {conversation_id}")
                return []
            
            # 4. Stage 2: Source Verification
            verified_candidates = await self._stage2_source_verification(candidates, context)
            
            # 5. Stage 3: Completeness Validation
            complete_candidates = await self._stage3_completeness_validation(verified_candidates)
            
            # 6. Stage 4: Quality Scoring
            final_candidates = await self._stage4_quality_scoring(complete_candidates, context)
            
            # 7. Store knowledge items
            knowledge_items = await self._store_knowledge_items(
                final_candidates, conversation_id, workspace_id, db
            )
            
            # 8. Mark conversation as processed
            await self._mark_conversation_processed(conversation_id, db)
            
            logger.info(f"Successfully extracted {len(knowledge_items)} knowledge items from conversation {conversation_id}")
            return knowledge_items
            
        except Exception as e:
            logger.error(f"Error extracting knowledge from conversation {conversation_id}: {e}")
            return []

    async def _verify_extraction_readiness(self, conversation_id: int, db: AsyncSession) -> bool:
        """Verify that conversation is ready for knowledge extraction."""
        
        query = select(Conversation).where(Conversation.id == conversation_id)
        result = await db.execute(query)
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            return False
        
        # Check if already processed
        if conversation.extraction_completed_at:
            logger.info(f"Conversation {conversation_id} already processed")
            return False
        
        # Check state requirements
        if conversation.state != ConversationState.RESOLVED.value:
            return False
        
        # Check minimum requirements
        if conversation.message_count < 3:
            return False
        
        if conversation.participant_count < 1:
            return False
        
        return True

    async def _assemble_conversation_context(self, conversation_id: int, db: AsyncSession) -> Optional[Dict[str, Any]]:
        """Assemble rich conversation context with all necessary information."""
        
        # Get conversation details
        conv_query = select(Conversation).where(Conversation.id == conversation_id)
        conv_result = await db.execute(conv_query)
        conversation = conv_result.scalar_one_or_none()
        
        if not conversation:
            return None
        
        # Get all messages in chronological order
        msg_query = select(Message).where(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at)
        
        msg_result = await db.execute(msg_query)
        messages = msg_result.scalars().all()
        
        if not messages:
            return None
        
        # Get participant information
        participants = await self._get_participant_info(messages, db)
        
        # Build timeline
        timeline = await self._build_conversation_timeline(messages, participants)
        
        # Analyze conversation structure
        structure = await self._analyze_conversation_structure(messages)
        
        return {
            'conversation_id': conversation_id,
            'conversation': conversation,
            'messages': messages,
            'participants': participants,
            'timeline': timeline,
            'structure': structure,
            'topic': conversation.topic,
            'duration_hours': (messages[-1].created_at - messages[0].created_at).total_seconds() / 3600,
            'resolution_indicators': conversation.resolution_indicators or []
        }

    async def _stage1_initial_extraction(self, context: Dict[str, Any]) -> List[ExtractionCandidate]:
        """Stage 1: Initial knowledge extraction with structured prompts."""
        
        try:
            # Use specialized prompts for different knowledge types
            all_candidates = []
            
            # Extract different types of knowledge with specialized prompts
            for knowledge_type, config in self.knowledge_types.items():
                candidates = await self._extract_knowledge_type(context, knowledge_type, config)
                all_candidates.extend(candidates)
            
            # Filter by minimum confidence
            filtered_candidates = [
                candidate for candidate in all_candidates
                if candidate.confidence >= self.config['min_confidence_threshold']
            ]
            
            logger.info(f"Stage 1: Extracted {len(filtered_candidates)} candidates from {len(all_candidates)} total")
            return filtered_candidates
            
        except Exception as e:
            logger.error(f"Error in Stage 1 extraction: {e}")
            return []

    async def _extract_knowledge_type(
        self, 
        context: Dict[str, Any], 
        knowledge_type: str, 
        config: Dict[str, Any]
    ) -> List[ExtractionCandidate]:
        """Extract specific type of knowledge with specialized prompts."""
        
        try:
            # Get specialized prompt for this knowledge type
            system_prompt = self._get_specialized_prompt(knowledge_type, config)
            
            # Format conversation for analysis
            conversation_text = self._format_conversation_for_extraction(context)
            
            user_prompt = f"""Analyze this conversation and extract {knowledge_type} knowledge:

CONVERSATION:
{conversation_text}

PARTICIPANTS:
{self._format_participants(context['participants'])}

TIMELINE:
Duration: {context['duration_hours']:.1f} hours
Topic: {context['topic'] or 'Not specified'}
Resolution indicators: {', '.join(context['resolution_indicators'])}

Extract knowledge following the specified format. Only extract if there's genuinely valuable, actionable information."""

            response = await self.openai_service._make_request(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse response
            candidates = await self._parse_extraction_response(
                response['choices'][0]['message']['content'],
                knowledge_type,
                context
            )
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error extracting {knowledge_type}: {e}")
            return []

    def _get_specialized_prompt(self, knowledge_type: str, config: Dict[str, Any]) -> str:
        """Get specialized extraction prompt for each knowledge type."""
        
        base_prompt = """You are an expert knowledge extractor that captures SPECIFIC, ACTIONABLE details from team conversations.

Your mission: Transform conversations into precise, usable knowledge that helps teams work more effectively.

EXTRACTION PHILOSOPHY:
- Extract knowledge as if creating detailed meeting notes for someone who wasn't there
- Focus on ACTIONABLE information that can be immediately used
- Include SPECIFIC details: names, dates, tools, exact steps, reasoning
- Provide COMPLETE context: background, discussion, and resolution
- Ensure CLEAR attribution: who said what and when

QUALITY STANDARDS:
- SPECIFIC QUOTES: Include actual quotes from key messages
- CONCRETE DETAILS: Numbers, dates, names, specific tools/services
- ACTIONABLE STEPS: Someone should be able to follow this immediately
- COMPLETE CONTEXT: Background, discussion, and resolution
- CLEAR ATTRIBUTION: Who said what, with their role/expertise

❌ REJECT VAGUE CONTENT:
- "The team discussed options" → TOO VAGUE
- "Someone mentioned a solution" → TOO VAGUE
- "We decided to use X because it's better" → TOO VAGUE

✅ EXTRACT RICH CONTENT:
- "John (Senior Engineer) recommended PostgreSQL over MySQL on March 15th because: 1) Better JSON support for user profiles, 2) Superior performance for complex queries, 3) Better Python compatibility. Decision: Migrate by Q2 end. Next: John creates migration plan by March 22nd."
"""

        specialized_prompts = {
            'technical_solution': base_prompt + """

🔧 TECHNICAL SOLUTION EXTRACTION:

Extract step-by-step technical solutions to specific problems.

REQUIRED COMPONENTS:
- PROBLEM: What specific technical issue was being solved?
- SOLUTION: What exact approach/method was chosen?
- IMPLEMENTATION STEPS: Detailed, numbered steps to implement
- TOOLS/TECHNOLOGIES: Specific tools, libraries, services mentioned
- VERIFICATION: How to test/verify the solution works
- GOTCHAS: Potential issues or considerations mentioned
- ATTRIBUTION: Who provided the solution and when

EXAMPLE OUTPUT:
{
    "title": "Fix Docker container memory leak in production",
    "problem": "Production containers running out of memory after 24 hours",
    "solution": "Implement proper garbage collection and connection pooling",
    "implementation_steps": [
        "1. Add NODE_OPTIONS='--max-old-space-size=4096' to Dockerfile",
        "2. Implement connection pooling with pg-pool library",
        "3. Add memory monitoring with prometheus metrics"
    ],
    "tools_used": ["Docker", "Node.js", "PostgreSQL", "Prometheus"],
    "verification": "Monitor memory usage over 48 hours, should stay below 3GB",
    "provided_by": "Sarah Chen (DevOps Lead)",
    "date_provided": "March 15, 2024",
    "confidence": 0.9
}

Return JSON array of technical solutions found.""",

            'process_definition': base_prompt + """

📋 PROCESS DEFINITION EXTRACTION:

Extract detailed workflow and procedure definitions.

REQUIRED COMPONENTS:
- PROCESS_NAME: Clear, specific name for the process
- TRIGGER_CONDITIONS: When should this process be used?
- PREREQUISITES: What must be in place first?
- DETAILED_STEPS: Numbered, specific steps with who does what
- INPUTS_OUTPUTS: What goes in, what comes out
- SUCCESS_CRITERIA: How to know it worked
- FAILURE_HANDLING: What to do when things go wrong
- PROCESS_OWNER: Who is responsible for this process
- FREQUENCY: How often this should happen

EXAMPLE OUTPUT:
{
    "title": "Code deployment process for production",
    "process_name": "Production Deployment Workflow",
    "trigger_conditions": "After successful staging tests and PM approval",
    "prerequisites": ["All tests passing", "Staging deployment successful", "Database migrations ready"],
    "detailed_steps": [
        "1. DevOps Lead creates deployment branch from main",
        "2. Run automated security scans (15 min)",
        "3. Deploy to blue environment first",
        "4. Run smoke tests on blue environment",
        "5. Switch traffic from green to blue",
        "6. Monitor for 30 minutes"
    ],
    "success_criteria": "Zero errors in logs, response time <200ms, all health checks green",
    "process_owner": "DevOps Team",
    "frequency": "2-3 times per week",
    "confidence": 0.85
}

Return JSON array of process definitions found.""",

            'decision_made': base_prompt + """

⚖️ DECISION EXTRACTION:

Extract specific decisions made during discussions.

REQUIRED COMPONENTS:
- DECISION: What exactly was decided?
- ALTERNATIVES_CONSIDERED: What other options were discussed?
- REASONING: Why was this specific choice made?
- DECISION_MAKER: Who had final authority?
- PARTICIPANTS: Who was involved in the discussion?
- TIMELINE: When does this take effect?
- IMPLEMENTATION: What are the next steps?
- SUCCESS_CRITERIA: How will success be measured?
- IMPACT: Who/what is affected by this decision?

EXAMPLE OUTPUT:
{
    "title": "Database choice for new analytics service",
    "decision": "Use PostgreSQL instead of MongoDB for analytics data",
    "alternatives_considered": ["MongoDB", "ClickHouse", "BigQuery"],
    "reasoning": "Better SQL compatibility, existing team expertise, superior JSONB performance for our use case",
    "decision_maker": "Sarah Chen (Tech Lead)",
    "participants": ["John Smith (Senior Engineer)", "Mike Johnson (Data Analyst)"],
    "timeline": "Migration to begin April 1st, complete by April 30th",
    "implementation": "John to create migration plan by March 25th, Mike to update analytics queries",
    "impact": "All analytics queries will need updating, 2-week migration period",
    "confidence": 0.9
}

Return JSON array of decisions found.""",

            'resource_recommendation': base_prompt + """

🛠️ RESOURCE RECOMMENDATION EXTRACTION:

Extract specific tool, service, or resource recommendations.

REQUIRED COMPONENTS:
- RESOURCE: What specific tool/service/resource was recommended?
- USE_CASE: What problem does this solve?
- BENEFITS: Why is this better than alternatives?
- CONSIDERATIONS: Limitations, costs, or gotchas mentioned?
- IMPLEMENTATION: How to get started or implement?
- RECOMMENDED_BY: Who made the recommendation and their expertise?
- ALTERNATIVES_MENTIONED: What other options were discussed?

EXAMPLE OUTPUT:
{
    "title": "Sentry for error monitoring recommendation",
    "resource": "Sentry.io error monitoring service",
    "use_case": "Replace manual log checking for production error tracking",
    "benefits": ["Real-time error alerts", "Error grouping and deduplication", "Performance monitoring", "Slack integration"],
    "considerations": ["$50/month for team plan", "Need to add SDK to all services", "Learning curve for advanced features"],
    "implementation": "Start with free tier, add to main API first, then expand to other services",
    "recommended_by": "Alex Kim (Senior Developer)",
    "alternatives_mentioned": ["Rollbar", "Bugsnag", "Custom logging"],
    "confidence": 0.8
}

Return JSON array of resource recommendations found.""",

            'troubleshooting_guide': base_prompt + """

🔍 TROUBLESHOOTING GUIDE EXTRACTION:

Extract step-by-step troubleshooting procedures.

REQUIRED COMPONENTS:
- PROBLEM_DESCRIPTION: What specific issue does this address?
- SYMPTOMS: How to recognize this problem?
- DIAGNOSIS_STEPS: How to confirm this is the issue?
- SOLUTION_STEPS: Detailed steps to fix the problem?
- VERIFICATION: How to confirm the fix worked?
- PREVENTION: How to prevent this in the future?
- ESCALATION: When to escalate and to whom?
- TOOLS_NEEDED: What tools or access is required?

EXAMPLE OUTPUT:
{
    "title": "Fix API rate limiting errors",
    "problem_description": "Users getting 429 errors during peak hours",
    "symptoms": ["429 status codes in logs", "User reports of 'too many requests'", "Increased response times"],
    "diagnosis_steps": [
        "1. Check rate limiting logs in CloudWatch",
        "2. Identify which endpoints are hitting limits",
        "3. Check if it's legitimate traffic or potential abuse"
    ],
    "solution_steps": [
        "1. Temporarily increase rate limits for affected endpoints",
        "2. Implement request queuing for burst traffic",
        "3. Add user-specific rate limiting rules"
    ],
    "verification": "Monitor 429 error rate drops below 1%, response times normalize",
    "prevention": "Implement gradual rate limit increases, add monitoring alerts",
    "provided_by": "DevOps Team",
    "confidence": 0.85
}

Return JSON array of troubleshooting guides found.""",

            'best_practice': base_prompt + """

💡 BEST PRACTICE EXTRACTION:

Extract proven approaches and methodologies.

REQUIRED COMPONENTS:
- PRACTICE_NAME: What is this best practice called?
- CONTEXT: When/where should this be applied?
- BENEFITS: What improvements does this provide?
- IMPLEMENTATION: How to put this into practice?
- EVIDENCE: What results or experience supports this?
- CONSIDERATIONS: When might this not apply?
- RELATED_PRACTICES: What other practices complement this?
- SOURCE: Who shared this and what's their experience?

EXAMPLE OUTPUT:
{
    "title": "Database migration best practices",
    "practice_name": "Zero-downtime database migrations",
    "context": "When updating production database schema with active users",
    "benefits": ["No service interruption", "Reduced risk", "Ability to rollback quickly"],
    "implementation": [
        "1. Always make schema changes backward compatible",
        "2. Deploy application changes before schema changes",
        "3. Use feature flags to control new functionality",
        "4. Test migrations on production-like data"
    ],
    "evidence": "Successfully used for 15+ migrations over past year, zero downtime incidents",
    "considerations": "Requires more planning time, some changes may need multiple releases",
    "source": "Database Team lead with 5 years migration experience",
    "confidence": 0.8
}

Return JSON array of best practices found."""
        }

        return specialized_prompts.get(knowledge_type, base_prompt + f"\n\nExtract {knowledge_type} knowledge from the conversation.")

    async def _stage2_source_verification(
        self, 
        candidates: List[ExtractionCandidate], 
        context: Dict[str, Any]
    ) -> List[ExtractionCandidate]:
        """Stage 2: Verify extractions against source material."""
        
        verified_candidates = []
        
        for candidate in candidates:
            try:
                verification = await self._verify_against_sources(candidate, context)
                
                if verification.is_valid:
                    # Adjust confidence based on verification
                    candidate.confidence = min(0.95, candidate.confidence + verification.confidence_adjustment)
                    
                    # Use corrected content if available
                    if verification.corrected_content:
                        candidate.content = verification.corrected_content
                    
                    # Add verification metadata
                    candidate.metadata['verification'] = {
                        'verified_at': datetime.utcnow().isoformat(),
                        'issues_found': verification.issues_found,
                        'quality_score': verification.quality_score
                    }
                    
                    verified_candidates.append(candidate)
                else:
                    logger.info(f"Candidate rejected in verification: {candidate.title}")
                    
            except Exception as e:
                logger.error(f"Error verifying candidate {candidate.title}: {e}")
                continue
        
        logger.info(f"Stage 2: {len(verified_candidates)} candidates passed verification")
        return verified_candidates

    async def _stage3_completeness_validation(
        self, 
        candidates: List[ExtractionCandidate]
    ) -> List[ExtractionCandidate]:
        """Stage 3: Validate completeness and actionability."""
        
        complete_candidates = []
        
        for candidate in candidates:
            try:
                completeness = await self._validate_completeness(candidate)
                
                if completeness.is_complete and completeness.is_actionable:
                    candidate.metadata['completeness'] = {
                        'validated_at': datetime.utcnow().isoformat(),
                        'quality_score': completeness.quality_score,
                        'actionable': completeness.is_actionable
                    }
                    complete_candidates.append(candidate)
                else:
                    logger.info(f"Candidate incomplete: {candidate.title}")
                    
            except Exception as e:
                logger.error(f"Error validating completeness for {candidate.title}: {e}")
                continue
        
        logger.info(f"Stage 3: {len(complete_candidates)} candidates passed completeness validation")
        return complete_candidates

    async def _stage4_quality_scoring(
        self, 
        candidates: List[ExtractionCandidate], 
        context: Dict[str, Any]
    ) -> List[ExtractionCandidate]:
        """Stage 4: Final quality scoring and filtering."""
        
        final_candidates = []
        
        for candidate in candidates:
            try:
                quality_score = await self._calculate_quality_score(candidate, context)
                
                if quality_score >= self.config['min_quality_score']:
                    candidate.metadata['final_quality_score'] = quality_score
                    final_candidates.append(candidate)
                else:
                    logger.info(f"Candidate failed quality threshold: {candidate.title} (score: {quality_score})")
                    
            except Exception as e:
                logger.error(f"Error calculating quality score for {candidate.title}: {e}")
                continue
        
        logger.info(f"Stage 4: {len(final_candidates)} candidates passed final quality scoring")
        return final_candidates

    # Helper methods for extraction stages

    def _format_conversation_for_extraction(self, context: Dict[str, Any]) -> str:
        """Format conversation for AI analysis."""
        messages = context['messages']
        participants = context['participants']
        
        formatted_lines = []
        for msg in messages:
            timestamp = msg.created_at.strftime("%H:%M")
            user_info = participants.get(msg.slack_user_id, {'name': msg.slack_user_id})
            user_display = f"{user_info.get('name', msg.slack_user_id)}"
            
            formatted_lines.append(f"[{timestamp}] {user_display}: {msg.content}")
        
        return '\n'.join(formatted_lines)

    def _format_participants(self, participants: Dict[str, Dict[str, Any]]) -> str:
        """Format participant information."""
        formatted = []
        for user_id, info in participants.items():
            name = info.get('name', user_id)
            role = info.get('role', 'Unknown')
            message_count = info.get('message_count', 0)
            formatted.append(f"- {name} ({role}): {message_count} messages")
        
        return '\n'.join(formatted)

    async def _parse_extraction_response(
        self, 
        response_text: str, 
        knowledge_type: str, 
        context: Dict[str, Any]
    ) -> List[ExtractionCandidate]:
        """Parse AI response into extraction candidates."""
        try:
            # Try to parse as JSON array
            if response_text.strip().startswith('['):
                items = json.loads(response_text)
            else:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    items = json.loads(json_match.group())
                else:
                    logger.warning(f"Could not parse extraction response for {knowledge_type}")
                    return []
            
            candidates = []
            for item in items:
                if isinstance(item, dict) and 'title' in item:
                    candidate = ExtractionCandidate(
                        title=item.get('title', 'Untitled'),
                        content=json.dumps(item, indent=2),  # Store full structured content
                        knowledge_type=knowledge_type,
                        confidence=item.get('confidence', 0.5),
                        source_message_ids=[msg.id for msg in context['messages']],
                        participants=list(context['participants'].keys()),
                        metadata={
                            'extraction_method': 'structured_prompt',
                            'raw_response': item,
                            'extracted_at': datetime.utcnow().isoformat()
                        }
                    )
                    candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error parsing extraction response: {e}")
            return []

    async def _verify_against_sources(
        self, 
        candidate: ExtractionCandidate, 
        context: Dict[str, Any]
    ) -> VerificationResult:
        """Verify extraction against source messages."""
        try:
            # Create verification prompt
            conversation_text = self._format_conversation_for_extraction(context)
            
            system_prompt = """You are a knowledge verification expert. Your job is to verify that extracted knowledge accurately represents what was actually said in the conversation.

VERIFICATION CRITERIA:
1. ACCURACY: Does the extracted information match what was actually said?
2. COMPLETENESS: Are all key details from the conversation included?
3. NO HALLUCINATION: Is there any information not present in the source?
4. PROPER ATTRIBUTION: Are quotes and attributions accurate?

Return JSON:
{
    "is_valid": true|false,
    "is_complete": true|false,
    "confidence_adjustment": -0.5 to +0.3,
    "issues_found": ["list of specific issues"],
    "corrected_content": "corrected version if needed",
    "quality_score": 0.0-1.0
}"""

            user_prompt = f"""Verify this extracted knowledge against the source conversation:

EXTRACTED KNOWLEDGE:
{candidate.content}

SOURCE CONVERSATION:
{conversation_text}

Verify accuracy, completeness, and check for any hallucinated information."""

            response = await self.openai_service._make_request(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = json.loads(response['choices'][0]['message']['content'])
            
            return VerificationResult(
                is_valid=result.get('is_valid', False),
                is_complete=result.get('is_complete', False),
                is_actionable=True,  # Will be checked in next stage
                confidence_adjustment=result.get('confidence_adjustment', 0.0),
                issues_found=result.get('issues_found', []),
                corrected_content=result.get('corrected_content'),
                quality_score=result.get('quality_score', 0.5)
            )
            
        except Exception as e:
            logger.error(f"Error in source verification: {e}")
            return VerificationResult(
                is_valid=False, is_complete=False, is_actionable=False,
                confidence_adjustment=-0.2, issues_found=[f"Verification failed: {e}"],
                corrected_content=None, quality_score=0.3
            )

    async def _validate_completeness(self, candidate: ExtractionCandidate) -> VerificationResult:
        """Validate that extraction is complete and actionable."""
        try:
            # Check knowledge type requirements
            type_config = self.knowledge_types.get(candidate.knowledge_type, {})
            required_fields = type_config.get('required_fields', [])
            
            # Parse content to check for required fields
            try:
                content_data = json.loads(candidate.content)
            except:
                content_data = {}
            
            missing_fields = []
            for field in required_fields:
                if field not in content_data or not content_data[field]:
                    missing_fields.append(field)
            
            is_complete = len(missing_fields) == 0
            
            # Check actionability
            word_count = len(candidate.content.split())
            is_actionable = (
                word_count >= self.config['min_content_words'] and
                word_count <= self.config['max_content_words'] and
                'steps' in candidate.content.lower() or 'how' in candidate.content.lower()
            )
            
            quality_score = 1.0 - (len(missing_fields) / max(len(required_fields), 1))
            
            return VerificationResult(
                is_valid=True,
                is_complete=is_complete,
                is_actionable=is_actionable,
                confidence_adjustment=0.0,
                issues_found=missing_fields,
                corrected_content=None,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Error validating completeness: {e}")
            return VerificationResult(
                is_valid=False, is_complete=False, is_actionable=False,
                confidence_adjustment=-0.1, issues_found=[f"Validation failed: {e}"],
                corrected_content=None, quality_score=0.2
            )

    async def _calculate_quality_score(
        self, 
        candidate: ExtractionCandidate, 
        context: Dict[str, Any]
    ) -> float:
        """Calculate final quality score for knowledge item."""
        
        # Base score from confidence
        score = candidate.confidence
        
        # Adjust based on verification results
        if 'verification' in candidate.metadata:
            verification_score = candidate.metadata['verification'].get('quality_score', 0.5)
            score = (score + verification_score) / 2
        
        # Adjust based on completeness
        if 'completeness' in candidate.metadata:
            completeness_score = candidate.metadata['completeness'].get('quality_score', 0.5)
            score = (score + completeness_score) / 2
        
        # Adjust based on conversation quality
        conversation_quality = min(1.0, context['duration_hours'] / 2.0)  # Longer conversations often have more context
        participant_quality = min(1.0, len(context['participants']) / 3.0)  # More participants often means more validation
        
        score = score * (0.7 + 0.15 * conversation_quality + 0.15 * participant_quality)
        
        return min(0.95, max(0.1, score))

    async def _store_knowledge_items(
        self,
        candidates: List[ExtractionCandidate],
        conversation_id: int,
        workspace_id: int,
        db: AsyncSession
    ) -> List[KnowledgeItem]:
        """Store verified knowledge items in the database."""
        
        knowledge_items = []
        
        for candidate in candidates:
            try:
                # Create summary from structured content
                summary = await self._generate_summary(candidate)
                
                knowledge_item = KnowledgeItem(
                    workspace_id=workspace_id,
                    conversation_id=conversation_id,
                    title=candidate.title,
                    summary=summary,
                    content=candidate.content,
                    knowledge_type=candidate.knowledge_type,
                    confidence_score=candidate.confidence,
                    source_messages=candidate.source_message_ids,
                    participants=candidate.participants,
                    item_metadata=candidate.metadata
                )
                
                db.add(knowledge_item)
                knowledge_items.append(knowledge_item)
                
            except Exception as e:
                logger.error(f"Error storing knowledge item {candidate.title}: {e}")
                continue
        
        await db.flush()  # Get IDs
        return knowledge_items

    async def _generate_summary(self, candidate: ExtractionCandidate) -> str:
        """Generate a concise summary of the knowledge item."""
        try:
            # Try to extract key points from structured content
            content_data = json.loads(candidate.content)
            
            # Create summary based on knowledge type
            if candidate.knowledge_type == 'decision_made':
                decision = content_data.get('decision', '')
                reasoning = content_data.get('reasoning', '')
                return f"Decision: {decision}. Reasoning: {reasoning[:100]}..."
            
            elif candidate.knowledge_type == 'technical_solution':
                problem = content_data.get('problem', '')
                solution = content_data.get('solution', '')
                return f"Solution for: {problem}. Approach: {solution[:100]}..."
            
            else:
                # Generic summary
                return candidate.title[:200] + "..."
                
        except:
            return candidate.title[:200] + "..."

    async def _mark_conversation_processed(self, conversation_id: int, db: AsyncSession):
        """Mark conversation as processed for knowledge extraction."""
        from sqlalchemy import update
        
        update_stmt = update(Conversation).where(
            Conversation.id == conversation_id
        ).values(
            extraction_completed_at=datetime.utcnow()
        )
        
        await db.execute(update_stmt)

    async def _get_participant_info(self, messages: List[Message], db: AsyncSession) -> Dict[str, Dict[str, Any]]:
        """Get participant information for the conversation."""
        user_ids = list(set(msg.slack_user_id for msg in messages))
        
        participants = {}
        for user_id in user_ids:
            # Get user info if available
            user_query = select(User).where(User.slack_id == user_id)
            user_result = await db.execute(user_query)
            user = user_result.scalar_one_or_none()
            
            # Count messages from this user
            message_count = sum(1 for msg in messages if msg.slack_user_id == user_id)
            
            participants[user_id] = {
                'name': user.name if user else user_id,
                'role': user.role if user else 'Unknown',
                'message_count': message_count
            }
        
        return participants

    async def _build_conversation_timeline(
        self, 
        messages: List[Message], 
        participants: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build chronological timeline of conversation."""
        timeline = []
        
        for msg in messages:
            user_info = participants.get(msg.slack_user_id, {})
            timeline.append({
                'timestamp': msg.created_at.isoformat(),
                'user': user_info.get('name', msg.slack_user_id),
                'content': msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                'message_id': msg.id
            })
        
        return timeline

    async def _analyze_conversation_structure(self, messages: List[Message]) -> Dict[str, Any]:
        """Analyze the structure and flow of the conversation."""
        if not messages:
            return {}
        
        # Basic analysis
        total_messages = len(messages)
        duration = (messages[-1].created_at - messages[0].created_at).total_seconds() / 3600
        avg_message_length = sum(len(msg.content.split()) for msg in messages) / total_messages
        
        # Question/answer patterns
        questions = sum(1 for msg in messages if '?' in msg.content)
        
        return {
            'total_messages': total_messages,
            'duration_hours': duration,
            'avg_message_length': avg_message_length,
            'question_count': questions,
            'messages_per_hour': total_messages / max(duration, 0.1)
        }


# Celery task wrapper
@celery_app.task
def extract_knowledge_from_complete_conversation(conversation_id: int, workspace_id: int):
    """Celery task for enhanced knowledge extraction."""
    try:
        logger.info(f"Starting enhanced knowledge extraction task for conversation {conversation_id}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                extract_knowledge_from_complete_conversation_async(conversation_id, workspace_id)
            )
            return result
        finally:
            loop.close()
            asyncio.set_event_loop(None)
            
    except Exception as e:
        logger.error(f"Error in enhanced knowledge extraction task: {e}", exc_info=True)
        raise


async def extract_knowledge_from_complete_conversation_async(conversation_id: int, workspace_id: int) -> Dict[str, Any]:
    """Async wrapper for enhanced knowledge extraction."""
    
    session_factory = get_session_factory()
    async with session_factory() as db:
        try:
            extractor = EnhancedKnowledgeExtractor()
            knowledge_items = await extractor.extract_knowledge_from_complete_conversation(
                conversation_id, workspace_id, db
            )
            
            await db.commit()
            
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "knowledge_items_extracted": len(knowledge_items),
                "extraction_completed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error in async knowledge extraction: {e}")
            raise
