"""
Team Memory Engine - The brain that makes the bot contextually aware.

This engine tracks:
1. Project states and decisions
2. Team member relationships and expertise
3. Conversation threads and context
4. Decision history and outcomes
5. Implicit references ("it", "that process", "the migration")

The goal is to make the bot act like a team member who remembers everything.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func
from sqlalchemy.orm import selectinload

from ..models.base import Message, Conversation, User, KnowledgeItem
from .openai_service import OpenAIService


@dataclass
class ProjectContext:
    """Current state of a project or topic."""
    name: str
    current_phase: str
    key_participants: List[str]
    recent_decisions: List[Dict[str, Any]]
    active_issues: List[str]
    next_steps: List[str]
    last_update: datetime
    confidence: float = 0.8


@dataclass
class TeamMemberProfile:
    """What we know about a team member."""
    user_id: str
    name: str
    expertise_areas: List[str]
    current_projects: List[str]
    typical_working_hours: Optional[str]
    communication_style: str
    recent_focus: List[str]
    collaboration_network: List[str]  # Who they work with most


@dataclass
class ConversationMemory:
    """Memory of a conversation thread."""
    conversation_id: int
    main_topic: str
    participants: List[str]
    decisions_made: List[Dict[str, Any]]
    action_items: List[str]
    unresolved_questions: List[str]
    context_references: Dict[str, str]  # "it" -> "database migration"
    sentiment_evolution: List[float]
    last_updated: datetime


@dataclass
class ContextualReference:
    """Resolved implicit reference like 'it', 'that process', etc."""
    original_text: str
    resolved_reference: str
    context_source: str
    confidence: float
    timestamp: datetime


class TeamMemoryEngine:
    """
    The contextual brain of the bot that remembers everything and connects the dots.

    This is what makes the bot feel like a real team member instead of a dumb GPT wrapper.
    """

    def __init__(self):
        self.openai_service = OpenAIService()
        self.project_memory: Dict[str, ProjectContext] = {}
        self.team_profiles: Dict[str, TeamMemberProfile] = {}
        self.conversation_memory: Dict[int, ConversationMemory] = {}
        self.decision_history: List[Dict[str, Any]] = []

    async def understand_query_context(
        self,
        query: str,
        user_id: str,
        channel_id: str,
        workspace_id: int,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        The main intelligence function that understands what the user REALLY wants.

        This goes beyond simple keyword matching to understand:
        - What project/topic they're referring to
        - Who they typically work with
        - What decisions are relevant
        - What implicit references mean
        """
        try:
            logger.info(f"🧠 Understanding context for query: {query}")

            # 1. Resolve implicit references ("it", "that", "the migration")
            resolved_references = await self._resolve_implicit_references(
                query, user_id, channel_id, workspace_id, db
            )

            # 2. Identify relevant projects and topics
            relevant_projects = await self._identify_relevant_projects(
                query, resolved_references, user_id, workspace_id, db
            )

            # 3. Build team context (who's involved, what they know)
            team_context = await self._build_team_context(
                query, relevant_projects, user_id, workspace_id, db
            )

            # 4. Find related decisions and outcomes
            relevant_decisions = await self._find_relevant_decisions(
                query, relevant_projects, workspace_id, db
            )

            # 5. Understand user's role and perspective
            user_perspective = await self._analyze_user_perspective(
                user_id, relevant_projects, workspace_id, db
            )

            context = {
                "resolved_query": self._build_resolved_query(query, resolved_references),
                "implicit_references": resolved_references,
                "relevant_projects": relevant_projects,
                "team_context": team_context,
                "relevant_decisions": relevant_decisions,
                "user_perspective": user_perspective,
                "context_confidence": self._calculate_context_confidence(
                    resolved_references, relevant_projects, team_context
                ),
                "suggested_search_terms": self._generate_enhanced_search_terms(
                    query, resolved_references, relevant_projects
                )
            }

            logger.info(f"🎯 Context confidence: {context['context_confidence']:.2f}")
            return context

        except Exception as e:
            logger.error(f"Error understanding query context: {e}")
            return {
                "resolved_query": query,
                "context_confidence": 0.1,
                "error": str(e)
            }

    async def _resolve_implicit_references(
        self,
        query: str,
        user_id: str,
        channel_id: str,
        workspace_id: int,
        db: AsyncSession
    ) -> List[ContextualReference]:
        """
        Resolve implicit references like 'it', 'that process', 'the migration'.

        This is CRITICAL for making the bot feel intelligent.
        """
        references = []

        # Common implicit reference patterns
        implicit_patterns = {
            r'\bit\b': 'it',
            r'\bthat\b': 'that',
            r'\bthis\b': 'this',
            r'\bthe process\b': 'the process',
            r'\bthe migration\b': 'the migration',
            r'\bthe issue\b': 'the issue',
            r'\bthe problem\b': 'the problem',
            r'\bthe project\b': 'the project',
            r'\bthe deployment\b': 'the deployment',
            r'\bthe update\b': 'the update',
            r'\bthe change\b': 'the change'
        }

        query_lower = query.lower()
        found_references = []

        for pattern, ref_type in implicit_patterns.items():
            import re
            if re.search(pattern, query_lower):
                found_references.append(ref_type)

        if not found_references:
            return references

        # Get recent conversation context to resolve references
        recent_messages = await self._get_recent_conversation_context(
            user_id, channel_id, workspace_id, db, hours_back=4
        )

        for ref in found_references:
            resolved = await self._resolve_single_reference(
                ref, query, recent_messages, workspace_id, db
            )
            if resolved:
                references.append(resolved)

        return references

    async def _resolve_single_reference(
        self,
        reference: str,
        query: str,
        recent_messages: List[Dict[str, Any]],
        workspace_id: int,
        db: AsyncSession
    ) -> Optional[ContextualReference]:
        """Resolve a single implicit reference using AI and context."""
        try:
            # Build context from recent messages
            context_text = "\n".join([
                f"{msg['user']}: {msg['content']}"
                for msg in recent_messages[-10:]  # Last 10 messages
            ])

            prompt = f"""
            You are resolving implicit references in team conversations.

            Query: "{query}"
            Reference to resolve: "{reference}"

            Recent conversation context:
            {context_text}

            What does "{reference}" refer to in this query? Be specific and concise.

            If you can't determine the reference clearly, respond with "UNCLEAR".

            Examples:
            - "it" might refer to "the database migration"
            - "that process" might refer to "the deployment process"
            - "the issue" might refer to "the authentication bug"

            Response (be specific):"""

            response = await self.openai_service.generate_completion(
                prompt=prompt,
                max_completion_tokens=50,
                temperature=1.0
            )

            resolved_text = response.strip()

            if resolved_text == "UNCLEAR" or len(resolved_text) < 3:
                return None

            return ContextualReference(
                original_text=reference,
                resolved_reference=resolved_text,
                context_source="conversation_history",
                confidence=0.7,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error resolving reference '{reference}': {e}")
            return None

    async def _identify_relevant_projects(
        self,
        query: str,
        resolved_references: List[ContextualReference],
        user_id: str,
        workspace_id: int,
        db: AsyncSession
    ) -> List[ProjectContext]:
        """Identify what projects/topics this query is about."""
        projects = []

        # Enhanced query with resolved references
        enhanced_query = query
        for ref in resolved_references:
            enhanced_query += f" {ref.resolved_reference}"

        # Look for project indicators in knowledge base
        try:
            # Search for related knowledge items to identify projects
            search_query = select(KnowledgeItem).where(
                and_(
                    KnowledgeItem.workspace_id == workspace_id,
                    func.lower(KnowledgeItem.title).contains(enhanced_query.lower())
                )
            ).limit(10)

            result = await db.execute(search_query)
            knowledge_items = result.scalars().all()

            # Extract project names from knowledge items
            project_names = set()
            for item in knowledge_items:
                # Look for project-like patterns in titles
                title_lower = item.title.lower()
                for keyword in ['migration', 'deployment', 'api', 'database', 'auth', 'login', 'project']:
                    if keyword in title_lower:
                        project_names.add(keyword.title())

            # Create project contexts
            for project_name in project_names:
                project_context = await self._build_project_context(
                    project_name, workspace_id, db
                )
                if project_context:
                    projects.append(project_context)

        except Exception as e:
            logger.error(f"Error identifying relevant projects: {e}")

        return projects

    async def _build_project_context(
        self,
        project_name: str,
        workspace_id: int,
        db: AsyncSession
    ) -> Optional[ProjectContext]:
        """Build context for a specific project."""
        try:
            # Get recent knowledge items related to this project
            search_query = select(KnowledgeItem).where(
                and_(
                    KnowledgeItem.workspace_id == workspace_id,
                    func.lower(KnowledgeItem.title).contains(project_name.lower())
                )
            ).order_by(desc(KnowledgeItem.created_at)).limit(5)

            result = await db.execute(search_query)
            knowledge_items = result.scalars().all()

            if not knowledge_items:
                return None

            # Extract participants from knowledge items
            participants = set()
            recent_decisions = []

            for item in knowledge_items:
                if item.extracted_data:
                    # Look for mentions of users
                    content = str(item.extracted_data)
                    # Simple extraction - in real implementation, use NER
                    import re
                    mentions = re.findall(r'@(\w+)', content)
                    participants.update(mentions)

                    # Look for decision indicators
                    if any(word in content.lower() for word in ['decided', 'agreed', 'chose']):
                        recent_decisions.append({
                            'content': content[:200] + '...',
                            'date': item.created_at,
                            'source': item.title
                        })

            return ProjectContext(
                name=project_name,
                current_phase="active",  # Could be determined by analysis
                key_participants=list(participants)[:5],
                recent_decisions=recent_decisions[:3],
                active_issues=[],
                next_steps=[],
                last_update=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error building project context for {project_name}: {e}")
            return None

    async def _build_team_context(
        self,
        query: str,
        relevant_projects: List[ProjectContext],
        user_id: str,
        workspace_id: int,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Build context about team members and their expertise."""
        try:
            # Get user information
            user_result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = user_result.scalar_one_or_none()

            # Build team context
            team_context = {
                "query_author": {
                    "name": user.name if user else "Unknown",
                    "slack_id": user.slack_id if user else "Unknown"
                },
                "relevant_team_members": [],
                "expertise_mapping": {},
                "collaboration_patterns": {}
            }

            # Add team members from relevant projects
            for project in relevant_projects:
                for participant in project.key_participants:
                    if participant not in [tm['name'] for tm in team_context["relevant_team_members"]]:
                        team_context["relevant_team_members"].append({
                            "name": participant,
                            "role_in_project": "contributor",
                            "relevance": "project_participant"
                        })

            return team_context

        except Exception as e:
            logger.error(f"Error building team context: {e}")
            return {}

    async def _find_relevant_decisions(
        self,
        query: str,
        relevant_projects: List[ProjectContext],
        workspace_id: int,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Find decisions that are relevant to this query."""
        decisions = []

        for project in relevant_projects:
            decisions.extend(project.recent_decisions)

        # Sort by relevance and recency
        decisions.sort(key=lambda x: x['date'], reverse=True)

        return decisions[:5]  # Top 5 most relevant decisions

    async def _analyze_user_perspective(
        self,
        user_id: str,
        relevant_projects: List[ProjectContext],
        workspace_id: int,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Understand the user's role and perspective."""
        try:
            user_result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = user_result.scalar_one_or_none()

            perspective = {
                "user_role": "team_member",
                "involvement_level": "participant",
                "likely_needs": [],
                "context_awareness": "medium"
            }

            # Determine involvement in relevant projects
            for project in relevant_projects:
                if user and user.slack_id in project.key_participants:
                    perspective["involvement_level"] = "key_participant"
                    perspective["context_awareness"] = "high"
                    break

            return perspective

        except Exception as e:
            logger.error(f"Error analyzing user perspective: {e}")
            return {"user_role": "team_member"}

    def _build_resolved_query(self, original_query: str, references: List[ContextualReference]) -> str:
        """Build an enhanced query with resolved references."""
        resolved_query = original_query

        for ref in references:
            # Replace implicit references with resolved ones
            resolved_query = resolved_query.replace(
                ref.original_text,
                f"{ref.original_text} ({ref.resolved_reference})"
            )

        return resolved_query

    def _calculate_context_confidence(
        self,
        references: List[ContextualReference],
        projects: List[ProjectContext],
        team_context: Dict[str, Any]
    ) -> float:
        """Calculate how confident we are in our context understanding."""
        confidence = 0.3  # Base confidence

        # Boost for resolved references
        if references:
            confidence += 0.2 * len(references)

        # Boost for identified projects
        if projects:
            confidence += 0.3 * len(projects)

        # Boost for team context
        if team_context.get("relevant_team_members"):
            confidence += 0.2

        return min(confidence, 0.95)

    def _generate_enhanced_search_terms(
        self,
        query: str,
        references: List[ContextualReference],
        projects: List[ProjectContext]
    ) -> List[str]:
        """Generate enhanced search terms based on context understanding."""
        terms = [query]

        # Add resolved references
        for ref in references:
            terms.append(ref.resolved_reference)

        # Add project names
        for project in projects:
            terms.append(project.name)

        return terms

    async def _get_recent_conversation_context(
        self,
        user_id: str,
        channel_id: str,
        workspace_id: int,
        db: AsyncSession,
        hours_back: int = 4
    ) -> List[Dict[str, Any]]:
        """Get recent conversation context for reference resolution."""
        try:
            # Get recent messages in the channel
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

            query = select(Message).join(Conversation).join(User).where(
                and_(
                    Conversation.workspace_id == workspace_id,
                    Conversation.slack_channel_id == channel_id,
                    Message.created_at >= cutoff_time
                )
            ).options(
                selectinload(Message.user),
                selectinload(Message.conversation)
            ).order_by(desc(Message.created_at)).limit(20)

            result = await db.execute(query)
            messages = result.scalars().all()

            context = []
            for message in reversed(messages):  # Reverse to chronological order
                try:
                    user_name = message.user.name if message.user else "Unknown"
                    context.append({
                        "user": user_name,
                        "content": message.content,
                        "timestamp": message.created_at
                    })
                except AttributeError:
                    # Handle case where user relationship isn't loaded
                    context.append({
                        "user": "Unknown",
                        "content": message.content,
                        "timestamp": message.created_at
                    })

            return context

        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return []