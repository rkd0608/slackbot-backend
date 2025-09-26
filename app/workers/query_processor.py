"""Query processor worker for handling user questions and generating AI responses."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from loguru import logger
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from .celery_app import celery_app
from ..core.config import settings
from ..models.base import Query, User, Workspace
from ..services.hallucination_preventer import HallucinationPreventer
from ..services.intent_classifier import IntentClassifier, IntentClassificationResult
from ..services.openai_service import OpenAIService
from ..services.response_formatter import ResponseFormatter, ContentData
from ..services.slack_service import SlackService
from ..services.suggestion_service import SuggestionService
from ..services.team_memory_engine import TeamMemoryEngine
from ..services.vector_service import VectorService


def get_async_session():
    """Create a new async session for each task."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    
    engine = create_async_engine(settings.database_url)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return AsyncSessionLocal

@celery_app.task
def process_query_async(
    query_id: Optional[int],
    workspace_id: int,
    user_id: int,
    channel_id: str,
    query_text: str,
    response_url: Optional[str] = None,
    is_slash_command: bool = False,
    thread_ts: Optional[str] = None
):
    """Process a user query and generate a response."""
    try:
        logger.info(f"Processing query: {query_text} from user {user_id}")
        
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                process_query(
                    query_id=query_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    channel_id=channel_id,
                    query_text=query_text,
                    response_url=response_url,
                    is_slash_command=is_slash_command,
                    thread_ts=thread_ts
                )
            )
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error("Error in query processing task: {}", str(e), exc_info=True)
        raise

async def process_query(
    query_id: Optional[int],
    workspace_id: int,
    user_id: int,
    channel_id: str,
    query_text: str,
    response_url: Optional[str] = None,
    is_slash_command: bool = False,
    thread_ts: Optional[str] = None
) -> Dict[str, Any]:
    """Process a user query and generate a response."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            # Ensure we start with a clean transaction
            await db.rollback()
            # Initialize services
            intent_classifier = IntentClassifier()
            suggestion_service = SuggestionService()
            hallucination_preventer = HallucinationPreventer()
            
            # 1. SMART context understanding with Team Memory Engine
            memory_engine = TeamMemoryEngine()
            query_context = await memory_engine.understand_query_context(
                query_text.strip(),
                user_id,
                channel_id,
                workspace_id,
                db
            )

            # Use the context-enhanced query
            sanitized_query = query_context.get("resolved_query", query_text.strip())
            context_confidence = query_context.get("context_confidence", 0.3)

            logger.info(f"Smart Context Analysis:")
            logger.info(f"  Original query: {query_text}")
            logger.info(f"  Context confidence: {context_confidence:.2f}")
            logger.info(f"  Resolved query: {sanitized_query}")

            if query_context.get("implicit_references"):
                refs = [f"{r.original_text} → {r.resolved_reference}" for r in query_context["implicit_references"]]
                logger.info(f"  Resolved references: {refs}")

            if query_context.get("relevant_projects"):
                projects = [p.name for p in query_context["relevant_projects"]]
                logger.info(f"  Relevant projects: {projects}")

            # Use enhanced search terms for better knowledge retrieval
            enhanced_search_terms = query_context.get("suggested_search_terms", [sanitized_query])
            
            # 2. Get Slack user ID and analyze conversational intelligence
            user_result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = user_result.scalar_one_or_none()
            slack_user_id = user.slack_id if user else "unknown"
            
            logger.info("Using basic conversational intelligence analysis")
            
            # Check if this query requires advanced AI intelligence
            advanced_indicators = [
                "understand", "explain", "analyze", "compare", "evaluate", "assess",
                "why", "how does", "how to", "what's the difference", "pros and cons", 
                "implications", "impact", "strategy", "approach", "solution",
                "best practice", "recommendation", "should we", "which is better",
                "what is", "what are", "how is", "how are", "tell me about",
                "describe", "overview", "summary", "like", "similar", "different",
                "benefits", "drawbacks", "advantages", "disadvantages", "features",
                "characteristics", "properties", "aspects", "considerations"
            ]
            
            # Check if this is a temporal query that should use direct conversation analysis
            temporal_indicators = [
                "today", "this morning", "this afternoon", "yesterday", 
                "last hour", "recent", "recently", "just now", "earlier",
                "what was discussed", "what happened", "what's been talked about"
            ]
            
            is_advanced_query = any(indicator in sanitized_query.lower() for indicator in advanced_indicators)
            is_temporal_query = any(indicator in sanitized_query.lower() for indicator in temporal_indicators)
            
            # Additional intelligence detection for complex queries
            if not is_advanced_query and not is_temporal_query:
                # Check for question complexity indicators
                complexity_indicators = [
                    len(sanitized_query.split()) > 6,  # Longer queries often need more intelligence
                    "?" in sanitized_query,  # Questions often benefit from advanced processing
                    any(word in sanitized_query.lower() for word in ["programming", "technology", "development", "architecture", "database", "system", "framework", "language", "tool", "platform"]),  # Technical topics
                    sanitized_query.lower().startswith(("what", "how", "why", "when", "where", "which", "who"))  # Question words
                ]
                
                # If multiple complexity indicators are present, treat as advanced
                if sum(complexity_indicators) >= 2:
                    is_advanced_query = True
                    logger.info("Detected complex query through fallback intelligence detection")
            
            # Advanced queries - simplified fallback since advanced AI service doesn't exist
            if is_advanced_query and not is_temporal_query:
                logger.info("Detected advanced query - using standard processing")
                # Fall through to standard processing below
            
            if is_temporal_query:
                logger.info("Detected temporal query - using standard processing")
                # Fall through to standard processing below
            
            # Create or get the query record
            if query_id is None:
                query = Query(
                    workspace_id=workspace_id,
                    user_id=user_id,
                    text=sanitized_query,
                    response={},
                    created_at=datetime.utcnow()
                )
                db.add(query)
                await db.flush()
                query_id = query.id
            else:
                result = await db.execute(
                    select(Query).where(Query.id == query_id)
                )
                query = result.scalar_one_or_none()
                if not query:
                    raise ValueError(f"Query {query_id} not found")
            
            # 2. Run intent classification first to determine if search is needed
            import asyncio
            try:
                intent_result = await asyncio.wait_for(
                    intent_classifier.classify_intent(
                        message_text=sanitized_query,
                        user_id=slack_user_id,
                        channel_id=channel_id,
                        workspace_id=workspace_id,
                        thread_ts=None,  # Could be extracted from context if available
                        db=db
                    ),
                    timeout=2.0  # 2 second timeout for intent classification
                )
                
            except asyncio.TimeoutError:
                logger.warning("Intent classification timed out, defaulting to knowledge query")
                intent_result = IntentClassificationResult(
                    intent="knowledge_query",
                    confidence=0.3,
                    classification_method="timeout_fallback",
                    contextual_metadata={},
                    entities=[],
                    temporal_scope=None,
                    is_conversational_response=False,
                    requires_knowledge_search=True
                )
            except Exception as e:
                logger.error(f"Intent classification failed: {e}")
                intent_result = IntentClassificationResult(
                    intent="knowledge_query",
                    confidence=0.5,
                    classification_method="fallback",
                    contextual_metadata={},
                    entities=[],
                    temporal_scope=None,
                    is_conversational_response=False,
                    requires_knowledge_search=True
                )
            
            # 3. Smart knowledge search - only when actually needed
            search_results = []
            should_search_knowledge = (
                intent_result.requires_knowledge_search or
                (intent_result.intent == "social_interaction" and len(sanitized_query.split()) > 5) or  # Only complex social interactions
                (intent_result.intent not in ["social_interaction", "ignore"] and len(sanitized_query.split()) > 3)  # Non-social substantial queries
            )

            if should_search_knowledge:
                try:
                    vector_service = VectorService()

                    # Use enhanced search with multiple terms from context analysis
                    all_search_results = []

                    for search_term in enhanced_search_terms[:3]:  # Top 3 search terms
                        logger.info(f"Searching with enhanced term: {search_term}")
                        term_results = await asyncio.wait_for(
                            vector_service.hybrid_search(
                                query=search_term,
                                workspace_id=workspace_id,
                                channel_id=channel_id,
                                limit=3,  # Fewer per term, but multiple terms
                                db=db
                            ),
                            timeout=2.0  # 2 second timeout per search
                        )
                        all_search_results.extend(term_results)

                    # Remove duplicates and keep top results
                    seen_ids = set()
                    search_results = []
                    for result in all_search_results:
                        if hasattr(result, 'id') and result.id not in seen_ids:
                            seen_ids.add(result.id)
                            search_results.append(result)
                        elif not hasattr(result, 'id'):
                            search_results.append(result)

                        if len(search_results) >= 8:  # Max 8 total results
                            break
                except asyncio.TimeoutError:
                    logger.warning("Knowledge search timed out")
                    search_results = []
                except Exception as e:
                    logger.error(f"Knowledge search failed: {e}")
                    search_results = []
            
            logger.info(f"Found {len(search_results)} relevant knowledge items")
            logger.info(f"Intent: {intent_result.intent} (confidence: {intent_result.confidence:.2f})")
            
            # 4. Generate AI response and suggestions in parallel (for all substantial queries)
            ai_response = None
            suggestions = []

            # Enhanced analysis including Team Memory context
            intent_analysis = {
                "intent": intent_result.intent,
                "confidence": intent_result.confidence,
                "entities": intent_result.entities,
                "temporal_scope": intent_result.temporal_scope
            }

            # Enhanced context analysis with Team Memory intelligence
            enhanced_context_analysis = {
                "intelligence_level": "advanced" if context_confidence > 0.6 else "basic",
                "conversation_context": query_context.get("team_context", {}),
                "user_intent": intent_result.intent,
                "context_confidence": context_confidence,
                "resolved_references": query_context.get("implicit_references", []),
                "relevant_projects": query_context.get("relevant_projects", []),
                "user_perspective": query_context.get("user_perspective", {}),
                "smart_analysis": True  # Flag to use enhanced prompts
            }

            if should_search_knowledge:
                try:
                    
                    ai_response, suggestions = await asyncio.wait_for(
                        asyncio.gather(
                            generate_enhanced_ai_response(sanitized_query, search_results, intent_analysis, enhanced_context_analysis, db),
                            suggestion_service.generate_suggestions(
                                query=sanitized_query,
                                search_results=search_results,
                                intent_analysis=intent_analysis
                            ),
                            return_exceptions=True
                        ),
                        timeout=5.0  # 5 second timeout for enhanced AI calls
                    )
                    
                    # Handle exceptions
                    if isinstance(ai_response, Exception):
                        logger.error(f"AI response generation failed: {ai_response}")
                        ai_response = await generate_ai_response(sanitized_query, search_results)
                    if isinstance(suggestions, Exception):
                        logger.error(f"Suggestion generation failed: {suggestions}")
                        suggestions = []
                        
                except asyncio.TimeoutError:
                    logger.warning("AI response/suggestions timed out, using basic response")
                    ai_response = await generate_ai_response(sanitized_query, search_results)
                    suggestions = []
            else:
                # For non-knowledge queries, create appropriate response
                ai_response = await _create_non_knowledge_response(intent_result, sanitized_query)
            
            # 5. Format the response for Slack
            if should_search_knowledge:
                # Use new ResponseFormatter for all substantial queries
                response_formatter = ResponseFormatter()
                
                # Convert AI response to ContentData format
                content_data = await _convert_to_content_data(
                    ai_response, search_results, suggestions, intent_result
                )
                
                # Format response using new system
                slack_response = await response_formatter.format_response(
                    content_data=content_data,
                    intent_result=intent_result,
                    query_text=sanitized_query,
                    query_id=query_id
                )
            else:
                # For non-knowledge queries, use simple response format
                slack_response = {
                    "response_type": "in_channel",
                    "text": ai_response.get("response", "I'm here to help!"),
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": ai_response.get("response", "I'm here to help!")
                            }
                        }
                    ]
                }
            
            logger.info(f"Slack response structure: {slack_response}")
            logger.info(f"Slack response type: {type(slack_response)}")
            
            # Store the enhanced response (with error handling)
            try:
                query.response = {
                    "query_text": sanitized_query,
                    "original_query": query_text,
                    "resolved_query": sanitized_query if sanitized_query != query_text.strip() else None,
                    "search_results_count": len(search_results),
                    "ai_response": ai_response,
                    "intent_analysis": intent_analysis,
                    "suggestions": suggestions,
                    "processed_at": datetime.utcnow().isoformat(),
                    "is_slash_command": is_slash_command,
                    "processing_time": "optimized_parallel"
                }
                
                logger.info("About to commit to database")
                await db.commit()
                logger.info("Database commit successful")
            except Exception as db_error:
                logger.error(f"Database save failed, but continuing with response: {db_error}")
                # Don't re-raise the error - we still want to send the response to Slack
            
            # Send response to Slack if we have a response URL
            logger.info(f"Response URL: {response_url}")
            if response_url:
                logger.info("Sending response to Slack via response URL")
                await send_slack_response(response_url, slack_response)
            
            # Always send response to the channel (for both @mentions and slash commands)
            logger.info(f"Sending response to Slack channel {channel_id} with thread_ts: {thread_ts}")
            await send_channel_response(channel_id, workspace_id, slack_response, thread_ts)
            
            logger.info(f"Successfully processed query {query_id}")
            
            return {
                "status": "success",
                "query_id": query_id,
                "response": slack_response,
                "search_results_count": len(search_results)
            }
            
        except Exception as e:
            logger.error("Error processing query: {}", str(e), exc_info=True)
            try:
                await db.rollback()
            except Exception as rollback_error:
                logger.error("Error during rollback: {}", str(rollback_error))
            
            # Only send error response if we haven't already sent a successful response
            # Check if we have a successful response ready
            if 'slack_response' not in locals():
                error_response = {
                    "response_type": "ephemeral",
                    "text": "Sorry, I encountered an error processing your question. Please try again in a moment."
                }
                
                if response_url:
                    await send_slack_response(response_url, error_response)
                
                # Always send error response to the channel
                await send_channel_response(channel_id, workspace_id, error_response)
            
            raise

async def generate_enhanced_ai_response(
    query: str, 
    search_results: List[Dict[str, Any]], 
    intent_analysis: Dict[str, Any],
    context_analysis: Optional[Dict[str, Any]] = None,
    db: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """Generate an enhanced AI-powered response using OpenAI with intent awareness."""
    try:
        openai_service = OpenAIService()
        
        # Create context from search results
        context = []
        for result in search_results[:3]:  # Use top 3 results
            context.append({
                "title": result.get("title", ""),
                "summary": result.get("summary", ""),
                "content": result.get("content", ""),
                "confidence": result.get("confidence", 0.0),
                "type": result.get("type", "unknown")
            })
        
        # Enhanced prompt with Team Memory Engine context
        if context_analysis and context_analysis.get("smart_analysis"):
            # Create SUPER-SMART system prompt with full context
            intent = intent_analysis.get("intent", "general_info")
            entities = intent_analysis.get("entities", [])
            scope = intent_analysis.get("temporal_scope", "general")

            # Extract Team Memory context
            resolved_refs = context_analysis.get("resolved_references", [])
            relevant_projects = context_analysis.get("relevant_projects", [])
            user_perspective = context_analysis.get("user_perspective", {})
            context_confidence = context_analysis.get("context_confidence", 0.0)

            # Build context strings
            references_str = ""
            if resolved_refs:
                refs = [f"'{ref.original_text}' refers to '{ref.resolved_reference}'" for ref in resolved_refs]
                references_str = f"\nRESOLVED REFERENCES: {'; '.join(refs)}"

            projects_str = ""
            if relevant_projects:
                project_names = [p.name for p in relevant_projects]
                projects_str = f"\nRELEVANT PROJECTS: {', '.join(project_names)}"

            user_context_str = f"\nUSER CONTEXT: {user_perspective.get('involvement_level', 'participant')}"

            system_prompt = f"""You are an ADVANCED team knowledge assistant with perfect memory of all team conversations.

You understand context like a brilliant teammate who remembers everything and connects the dots.

CONTEXT ANALYSIS:
- Intent: {intent}
- Confidence: {context_confidence:.2f}
- Temporal Scope: {scope}
- Entities: {entities}{references_str}{projects_str}{user_context_str}

YOUR ADVANCED CAPABILITIES:
- You understand implicit references ("it", "that process", "the migration")
- You connect conversations across different time periods
- You know who's working on what projects
- You remember decisions and their outcomes
- You provide context about why things were decided

CRITICAL: EXTRACT TECHNICAL DETAILS FROM RAW MESSAGE CONTENT:
When you see conversation search results with raw message content, you MUST:
1. Look for specific commands, scripts, procedures in the raw message text
2. Extract exact syntax, file names, command flags, and technical specifications
3. Include code snippets, configuration details, and step-by-step procedures
4. Don't just summarize - provide the actual technical content verbatim

SPECIAL INSTRUCTION FOR CONVERSATION RESULTS:
If you see results with type:"conversation" containing raw Slack message content, these contain the ACTUAL detailed responses from team members. Extract the specific technical information from these messages, including:
- Command examples (pg_dump, scripts, etc.)
- Code blocks marked with ```
- Specific procedures and steps
- File names and configurations
- Error messages and solutions

Example: If you see a conversation result containing "pg_dump --jobs=4 --format=directory", include this EXACT command in your response.

SUPER-SMART RESPONSE REQUIREMENTS:

YOUR MISSION:
Transform stored knowledge into immediately useful answers that help the user take action or understand exactly what happened.

RESPONSE REQUIREMENTS:

For DECISIONS:
- What exactly was decided?
- Who made the decision and when?
- What was the reasoning?
- What are the next steps?
- Who is responsible for what?

For TECHNICAL SOLUTIONS:
- What specific problem does this solve?
- What are the exact steps?
- What tools/resources are needed?
- Who provided this solution?
- Any important warnings or considerations?

For PROCESSES:
- What is the step-by-step procedure?
- When should this be used?
- Who is responsible for each step?
- What are the success criteria?

RESPONSE QUALITY STANDARDS:
- LEAD WITH THE ANSWER: Start with the specific information they need
- INCLUDE SPECIFICS: Names, dates, tools, exact steps, reasoning
- PROVIDE CONTEXT: Why this matters, what led to this decision
- CITE SOURCES: Who said what and when
- SUGGEST NEXT STEPS: What the user should do with this information

AVOID VAGUE RESPONSES:
- "The team discussed database migration" -> TOO VAGUE
- "There was a conversation about this topic" -> USELESS
- "Some decisions were made" -> UNHELPFUL

PROVIDE RICH RESPONSES:
- "John (Senior Engineer) decided on March 15th to migrate from MySQL to PostgreSQL because of better JSON support needed for the new analytics features. Timeline: Migration by end of Q2. Next steps: John is creating the migration plan by March 22nd, Sarah is reviewing infrastructure requirements."

SOURCE ATTRIBUTION:
Always include:
- WHO provided the information (with their role if available)
- WHEN the conversation happened
- WHICH channel or thread this came from"""
        else:
            # Fallback to basic smart prompt
            system_prompt = """You are a team knowledge assistant who provides SPECIFIC, ACTIONABLE answers from your team's conversation history.

YOUR MISSION:
Transform stored knowledge into immediately useful answers with specific details, not vague summaries.

RESPONSE REQUIREMENTS:
- LEAD WITH THE ANSWER: Start with the specific information they need
- INCLUDE SPECIFICS: Names, dates, tools, exact steps, reasoning
- PROVIDE CONTEXT: Why this matters, what led to this decision
- CITE SOURCES: Who said what and when
- SUGGEST NEXT STEPS: What the user should do with this information

IMPORTANT RULES:
1. Extract ALL relevant details from the knowledge items
2. If knowledge is incomplete, specify exactly what's missing
3. Prioritize actionable information over background
4. Use direct quotes when they add value
5. Connect related pieces of knowledge when relevant"""

        user_message = f"""Question: {query}

Available Knowledge (including raw conversation messages that may contain specific technical details):
{json.dumps(context, indent=2)}

CRITICAL INSTRUCTION:
The data above includes conversation search results from Slack messages. These messages contain the ACTUAL technical responses from team members with specific details like:
- Exact command line syntax
- Code snippets in ``` blocks
- Script names and file paths
- Configuration changes
- Step-by-step procedures

You MUST extract and present these technical details verbatim. Do not just say "changes are needed" - show the EXACT changes, commands, and procedures that were discussed."""

        # Generate enhanced response with technical detail extraction
        response = await openai_service._make_request(
            model="gpt-4o-mini",  # Use GPT-4o-mini for better technical understanding
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,  # Lower temperature for precise technical responses
            max_completion_tokens=2000  # Much more tokens for detailed technical responses
        )
        
        content = response['choices'][0]['message']['content']
        
        return {
            "response": content,
            "model_used": "gpt-4o-mini",
            "generated_at": datetime.utcnow().isoformat(),
            "context_used": len(context),
            "intent_aware": True
        }
        
    except Exception as e:
        logger.error(f"Error generating enhanced AI response: {e}")
        return {
            "response": "I'm sorry, I encountered an error while generating a response. Please try again.",
            "model_used": "fallback",
            "generated_at": datetime.utcnow().isoformat(),
            "context_used": 0,
            "error": str(e),
            "fallback": True
        }

async def generate_ai_response(query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate an AI-powered response using OpenAI."""
    try:
        openai_service = OpenAIService()
        
        # Create context from search results
        context = []
        for result in search_results[:3]:  # Use top 3 results
            context.append({
                "title": result.get("title", ""),
                "summary": result.get("summary", ""),
                "content": result.get("content", ""),
                "confidence": result.get("confidence", 0.0),
                "type": result.get("type", "unknown")
            })
        
        # Create prompt for AI response
        system_prompt = """You are a team knowledge assistant who provides SPECIFIC, ACTIONABLE answers from your teams conversation history.

YOUR MISSION:
Transform stored knowledge into immediately useful answers with specific details, not vague summaries.

RESPONSE REQUIREMENTS:
- LEAD WITH THE ANSWER: Start with the specific information they need
- INCLUDE SPECIFICS: Names, dates, tools, exact steps, reasoning  
- PROVIDE CONTEXT: Why this matters, what led to this decision
- CITE SOURCES: Who said what and when
- SUGGEST NEXT STEPS: What the user should do with this information

AVOID VAGUE RESPONSES:
- "The team discussed database migration" -> TOO VAGUE
- "There was a conversation about this topic" -> USELESS

PROVIDE RICH RESPONSES:
- "John decided on March 15th to migrate from MySQL to PostgreSQL because of better JSON support needed for analytics. Timeline: Migration by end of Q2. Next steps: John is creating the migration plan by March 22nd."

IMPORTANT RULES:
1. Extract ALL relevant details from the knowledge items
2. If knowledge is incomplete, specify exactly what's missing
3. Prioritize actionable information over background
4. Use direct quotes when they add value
5. Connect related pieces of knowledge when relevant"""

        user_message = f"""Question: {query}

Available Knowledge (including raw conversation messages that may contain specific technical details):
{json.dumps(context, indent=2)}

CRITICAL INSTRUCTION:
The data above includes conversation search results from Slack messages. These messages contain the ACTUAL technical responses from team members with specific details like:
- Exact command line syntax
- Code snippets in ``` blocks
- Script names and file paths
- Configuration changes
- Step-by-step procedures

You MUST extract and present these technical details verbatim. Do not just say "changes are needed" - show the EXACT changes, commands, and procedures that were discussed."""

        # Generate enhanced response with technical detail extraction
        response = await openai_service._make_request(
            model="gpt-4o-mini",  # Use GPT-4o-mini for better technical understanding
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,  # Lower temperature for precise technical responses
            max_completion_tokens=2000  # Much more tokens for detailed technical responses
        )
        
        content = response['choices'][0]['message']['content']
        
        # NEW: Validate AI response for hallucinations
        hallucination_preventer = HallucinationPreventer()
        hallucination_check = await hallucination_preventer.validate_ai_response(
            ai_response=content,
            query=query,
            search_results=search_results
        )
        
        # Use corrected content if hallucinations were detected
        if not hallucination_check.is_safe and hallucination_check.corrected_content:
            logger.warning(f"AI response corrected due to hallucination risk: {hallucination_check.issues_found}")
            content = hallucination_check.corrected_content
        elif not hallucination_check.is_safe:
            logger.warning(f"AI response has hallucination risk but no correction available: {hallucination_check.issues_found}")
            # Add disclaimer to response
            content = f"{content}\n\n*Note: This response may contain inferred information. Please verify details.*"
        
        return {
            "response": content,
            "model_used": "gpt-4o-mini",
            "generated_at": datetime.utcnow().isoformat(),
            "context_used": len(context),
            "hallucination_check": {
                "is_safe": hallucination_check.is_safe,
                "confidence": hallucination_check.confidence,
                "issues_found": hallucination_check.issues_found
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return {
            "response": "I'm sorry, I couldn't generate an AI response at the moment. Here are the relevant knowledge items I found:",
            "error": str(e),
            "fallback": True
        }


async def send_slack_response(response_url: str, response_data: Dict[str, Any]):
    """Send response to Slack using the response URL."""
    try:
        import aiohttp
        import json
        
        # Slack expects JSON payload, not form data
        headers = {
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                response_url, 
                json=response_data,
                headers=headers
            ) as resp:
                if resp.status == 200:
                    logger.info("Successfully sent response to Slack")
                else:
                    response_text = await resp.text()
                    logger.error(f"Failed to send response to Slack: {resp.status} - {response_text}")
                    
    except Exception as e:
        logger.error(f"Error sending response to Slack: {e}")

async def _create_non_knowledge_response(intent_result: IntentClassificationResult, query_text: str) -> Dict[str, Any]:
    """Create appropriate response for non-knowledge queries."""
    if intent_result.intent == "social_interaction":
        return {
            "response": "Hi there! I'm here to help with questions about your team's knowledge and processes. What would you like to know?",
            "fallback": True
        }
    elif intent_result.intent == "conversational_response":
        return {
            "response": "I understand. Is there anything else I can help you with?",
            "fallback": True
        }
    elif intent_result.intent == "ignore":
        return {
            "response": "I didn't quite catch that. Could you please rephrase your question?",
            "fallback": True
        }
    else:
        return {
            "response": "I'm here to help! What would you like to know?",
            "fallback": True
        }

async def _convert_to_content_data(
    ai_response: Optional[Dict[str, Any]], 
    search_results: List[Dict[str, Any]], 
    suggestions: List[str],
    intent_result: IntentClassificationResult
) -> ContentData:
    """Convert AI response and search results to ContentData format."""
    
    # Extract main content
    main_content = ""
    if ai_response and isinstance(ai_response, dict):
        main_content = ai_response.get("response", "")
    
    # Convert search results to sources format
    sources = []
    for result in search_results:
        source = {
            "title": result.get("title", "Conversation"),
            "date": result.get("created_at", "Unknown date"),
            "channel": result.get("channel_name", "Unknown channel"),
            "excerpt": (result.get("summary") or result.get("content") or "")[:200]
        }
        sources.append(source)
    
    # Extract next steps from suggestions
    next_steps = suggestions[:5] if suggestions else []
    
    # Extract related information
    related_info = []
    if ai_response and isinstance(ai_response, dict):
        if ai_response.get("related_information"):
            related_info = ai_response["related_information"][:3]
    
    # Determine content type and extract specific data
    decision_info = None
    process_steps = None
    technical_details = None
    
    if intent_result.intent == "decision_rationale" and ai_response:
        decision_info = {
            "outcome": ai_response.get("decision_outcome", ""),
            "decision_maker": ai_response.get("decision_maker", ""),
            "date": ai_response.get("decision_date", ""),
            "rationale": ai_response.get("rationale", "")
        }
    elif intent_result.intent == "process" and ai_response:
        process_steps = ai_response.get("process_steps", [])
    elif intent_result.intent == "troubleshooting" and ai_response:
        technical_details = {
            "problem": ai_response.get("problem_description", ""),
            "solution": ai_response.get("solution", ""),
            "implementation": ai_response.get("implementation_steps", []),
            "tools": ai_response.get("required_tools", []),
            "verification": ai_response.get("verification_steps", "")
        }
    
    return ContentData(
        main_content=main_content,
        sources=sources,
        next_steps=next_steps,
        related_info=related_info,
        decision_info=decision_info,
        process_steps=process_steps,
        technical_details=technical_details,
        verification_links=[]
    )

async def send_channel_response(channel_id: str, workspace_id: int, response_data: Dict[str, Any], thread_ts: Optional[str] = None):
    """Send response to the Slack channel."""
    try:
        slack_service = SlackService()
        
        # Get workspace tokens
        async_session = get_async_session()
        async with async_session() as db:
            result = await db.execute(
                select(Workspace).where(Workspace.id == workspace_id)
            )
            workspace = result.scalar_one_or_none()
            
            if workspace and hasattr(workspace, 'tokens') and workspace.tokens:
                bot_token = workspace.tokens.get("access_token")
                if bot_token:
                    # Send message to channel (in thread if thread_ts provided)
                    result = await slack_service.send_message(
                        channel=channel_id,
                        text=response_data.get("text", "Response generated"),
                        thread_ts=thread_ts,  # Use thread_ts if provided for threaded responses
                        blocks=response_data.get("blocks"),  # Include rich formatting blocks
                        token=bot_token
                    )
                    logger.info(f"Sent response to channel {channel_id}")
                else:
                    logger.error("No bot token found in workspace tokens")
            else:
                logger.error("No workspace or tokens found")
                    
    except Exception as e:
        logger.error(f"Error sending channel response: {e}", exc_info=True)

@celery_app.task
def process_pending_queries():
    """Process any pending queries that haven't been processed yet."""
    try:
        logger.info("Processing pending queries...")
        
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(process_pending_queries_async())
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error processing pending queries: {e}", exc_info=True)
        raise

async def process_pending_queries_async() -> Dict[str, Any]:
    """Process queries that don't have responses yet."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            # Find queries without responses
            result = await db.execute(
                select(Query)
                .where(
                    and_(
                        Query.response.is_(None),
                        Query.created_at >= datetime.utcnow() - timedelta(hours=24)  # Only recent queries
                    )
                )
                .limit(10)  # Process in batches
            )
            
            pending_queries = result.scalars().all()
            
            processed_count = 0
            for query in pending_queries:
                try:
                    # Get user and workspace info
                    user_result = await db.execute(
                        select(User).where(User.id == query.user_id)
                    )
                    user = user_result.scalar_one_or_none()
                    
                    if user:
                        await process_query(
                            query_id=query.id,
                            workspace_id=query.workspace_id,
                            user_id=query.user_id,
                            channel_id="",  # Unknown for pending queries
                            query_text=query.text,
                            response_url=None,
                            is_slash_command=False
                        )
                        processed_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing pending query {query.id}: {e}")
                    continue
            
            logger.info(f"Processed {processed_count} pending queries")
            return {"status": "success", "processed_count": processed_count}
            
        except Exception as e:
            logger.error(f"Error processing pending queries: {e}", exc_info=True)
            await db.rollback()
            raise
