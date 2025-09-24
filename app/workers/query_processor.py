"""Query processor worker for handling user questions and generating AI responses."""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from loguru import logger

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..models.base import Query, KnowledgeItem, User, Workspace
from ..services.vector_service import VectorService
from ..services.slack_service import SlackService
from ..services.openai_service import OpenAIService
from ..services.intent_classifier import IntentClassifier
from ..services.suggestion_service import SuggestionService
from ..services.security_service import SecurityService
from ..services.context_resolver import ContextResolver
from ..services.hallucination_preventer import HallucinationPreventer
from ..services.conversational_intelligence import ConversationalIntelligence
from ..services.direct_conversation_analyzer import DirectConversationAnalyzer
from ..services.advanced_ai_intelligence import AdvancedAIIntelligence
from .celery_app import celery_app

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
    is_slash_command: bool = False
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
                    is_slash_command=is_slash_command
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
    is_slash_command: bool = False
) -> Dict[str, Any]:
    """Process a user query and generate a response."""
    async_session = get_async_session()
    async with async_session() as db:
        try:
            # Initialize services
            security_service = SecurityService()
            intent_classifier = IntentClassifier()
            suggestion_service = SuggestionService()
            context_resolver = ContextResolver()
            hallucination_preventer = HallucinationPreventer()
            
            # 1. Context resolution - resolve references like "it", "the process", etc.
            resolved_query, contextual_references = await context_resolver.resolve_query_context(
                query=query_text.strip(),
                user_id=user_id,
                channel_id=channel_id,
                workspace_id=workspace_id,
                db=db
            )
            
            logger.info(f"Original query: {query_text}")
            if resolved_query != query_text.strip():
                logger.info(f"Resolved query: {resolved_query}")
                logger.info(f"Context references resolved: {len(contextual_references)}")
            
            # Use the resolved query for processing
            sanitized_query = resolved_query
            
            # 2. Get Slack user ID and analyze conversational intelligence
            user_result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = user_result.scalar_one_or_none()
            slack_user_id = user.slack_id if user else "unknown"
            
            conv_intelligence = ConversationalIntelligence()
            context_analysis = await conv_intelligence.analyze_query_context(
                query=sanitized_query,
                user_id=slack_user_id,
                channel_id=channel_id,
                workspace_id=workspace_id,
                db=db
            )
            
            logger.info(f"Conversational intelligence analysis: {context_analysis.get('intelligence_level', 'basic')}")
            
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
            
            # Use Advanced AI Intelligence for sophisticated queries
            if is_advanced_query and not is_temporal_query:
                logger.info("Detected advanced query requiring sophisticated AI analysis")
                
                # Initialize Advanced AI Intelligence
                advanced_ai = AdvancedAIIntelligence()
                
                # Prepare comprehensive context data
                context_data = {
                    "conversational_intelligence": context_analysis,
                    "user_info": {"user_id": user_id, "slack_id": slack_user_id},
                    "channel_info": {"channel_id": channel_id},
                    "query_metadata": {
                        "original_query": query_text,
                        "resolved_query": sanitized_query,
                        "contextual_references": contextual_references
                    }
                }
                
                # Get sophisticated AI analysis and response
                advanced_result = await advanced_ai.analyze_and_respond(
                    query=sanitized_query,
                    context_data=context_data,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    channel_id=channel_id,
                    db=db
                )
                
                if advanced_result["status"] == "success":
                    # Create a query record for advanced processing
                    advanced_query = Query(
                        workspace_id=workspace_id,
                        user_id=user_id,
                        text=sanitized_query,
                        response={},
                        created_at=datetime.utcnow()
                    )
                    db.add(advanced_query)
                    await db.flush()
                    advanced_query_id = advanced_query.id
                    
                    # Format sophisticated AI response
                    ai_response = advanced_result["response"]
                    intelligence_metrics = advanced_result.get("intelligence_metrics", {})
                    
                    response_text = f"🧠 **Intelligent Analysis**\n\n{ai_response['response_text']}"
                    
                    # Add intelligence indicators if high sophistication
                    if intelligence_metrics.get("response_sophistication", 0) > 0.8:
                        response_text += f"\n\n*Analysis Mode: {advanced_result.get('reasoning_mode', 'analytical').title()}*"
                        response_text += f"\n*Complexity Score: {intelligence_metrics.get('complexity_score', 0.5):.1f}/1.0*"
                    
                    slack_response = {
                        "response_type": "in_channel",
                        "text": response_text,
                        "blocks": [
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": response_text
                                }
                            }
                        ]
                    }
                    
                    # Store the advanced response
                    advanced_query.response = {
                        "query_text": sanitized_query,
                        "original_query": query_text,
                        "analysis_type": "advanced_ai_intelligence",
                        "reasoning_mode": advanced_result.get("reasoning_mode"),
                        "intelligence_metrics": intelligence_metrics,
                        "response_text": response_text,
                        "generated_at": datetime.utcnow().isoformat()
                    }
                    
                    await db.commit()
                    
                    # Send sophisticated response immediately
                    if response_url:
                        await send_slack_response(response_url, slack_response)
                    
                    if is_slash_command:
                        await send_channel_response(channel_id, workspace_id, slack_response)
                    
                    logger.info(f"Successfully processed advanced query {advanced_query_id} with {advanced_result.get('reasoning_mode')} reasoning")
                    return {
                        "status": "success",
                        "query_id": advanced_query_id,
                        "response": slack_response,
                        "analysis_type": "advanced_ai_intelligence",
                        "reasoning_mode": advanced_result.get("reasoning_mode")
                    }
            
            is_temporal_query = any(indicator in sanitized_query.lower() for indicator in temporal_indicators)
            
            if is_temporal_query:
                logger.info("Detected temporal query, using direct conversation analysis")
                
                # Use direct conversation analyzer for temporal queries
                analyzer = DirectConversationAnalyzer()
                
                # Determine time range based on query
                hours_back = 24  # Default to today
                if "this morning" in sanitized_query.lower():
                    hours_back = 12
                elif "last hour" in sanitized_query.lower() or "recently" in sanitized_query.lower():
                    hours_back = 2
                elif "yesterday" in sanitized_query.lower():
                    hours_back = 48
                
                direct_analysis = await analyzer.analyze_recent_discussions(
                    query=sanitized_query,
                    workspace_id=workspace_id,
                    channel_id=channel_id,
                    db=db,
                    hours_back=hours_back
                )
                
                if direct_analysis["status"] == "success":
                    # Create a temporary query record for temporal queries
                    temp_query = Query(
                        workspace_id=workspace_id,
                        user_id=user_id,
                        text=sanitized_query,
                        response={},
                        created_at=datetime.utcnow()
                    )
                    db.add(temp_query)
                    await db.flush()
                    temp_query_id = temp_query.id
                    
                    # Format direct analysis result for response
                    analysis_result = direct_analysis["analysis"]
                    
                    response_text = f"📋 **Recent Discussions Analysis**\n\n{analysis_result['analysis']}\n\n**Time Range:** {analysis_result['time_span']}\n**Conversations Analyzed:** {analysis_result['conversations_count']}"
                    
                    if analysis_result.get("channels_involved"):
                        response_text += f"\n**Channels:** {', '.join(analysis_result['channels_involved'])}"
                    
                    if analysis_result.get("participants"):
                        response_text += f"\n**Participants:** {', '.join(analysis_result['participants'])}"
                    
                    slack_response = {
                        "response_type": "in_channel",
                        "text": response_text,
                        "blocks": [
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": response_text
                                }
                            }
                        ]
                    }
                    
                    # Store the response
                    temp_query.response = {
                        "query_text": sanitized_query,
                        "original_query": query_text,
                        "analysis_type": "direct_temporal",
                        "time_range_hours": hours_back,
                        "conversations_analyzed": direct_analysis["conversations_analyzed"],
                        "response_text": response_text,
                        "generated_at": datetime.utcnow().isoformat()
                    }
                    
                    await db.commit()
                    
                    # Send response immediately for temporal queries
                    if response_url:
                        await send_slack_response(response_url, slack_response)
                    
                    if is_slash_command:
                        await send_channel_response(channel_id, workspace_id, slack_response)
                    
                    logger.info(f"Successfully processed temporal query {temp_query_id} with direct analysis")
                    return {
                        "status": "success",
                        "query_id": temp_query_id,
                        "response": slack_response,
                        "analysis_type": "direct_temporal"
                    }
            
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
            
            # 2. Start knowledge search immediately (don't wait for intent)
            vector_service = VectorService()
            search_task = vector_service.hybrid_search(
                query=sanitized_query,
                workspace_id=workspace_id,
                channel_id=channel_id,
                limit=5,
                db=db
            )
            
            # 3. Run intent classification in parallel with search
            import asyncio
            try:
                # Run both operations in parallel with timeout
                search_results, intent_analysis = await asyncio.wait_for(
                    asyncio.gather(
                        search_task,
                        intent_classifier.classify_intent(sanitized_query),
                        return_exceptions=True
                    ),
                    timeout=3.0  # 3 second timeout
                )
                
                # Handle exceptions
                if isinstance(search_results, Exception):
                    logger.error(f"Search failed: {search_results}")
                    search_results = []
                if isinstance(intent_analysis, Exception):
                    logger.error(f"Intent classification failed: {intent_analysis}")
                    intent_analysis = {"intent": "general_info", "confidence": 0.5}
                    
            except asyncio.TimeoutError:
                logger.warning("Search/intent classification timed out, using fallbacks")
                search_results = []
                intent_analysis = {"intent": "general_info", "confidence": 0.5}
            
            logger.info(f"Found {len(search_results)} relevant knowledge items")
            logger.info(f"Intent: {intent_analysis.get('intent', 'unknown')} (confidence: {intent_analysis.get('confidence', 0):.2f})")
            
            # 4. Generate AI response and suggestions in parallel
            try:
                ai_response, suggestions = await asyncio.wait_for(
                    asyncio.gather(
                        generate_enhanced_ai_response(sanitized_query, search_results, intent_analysis, context_analysis, db),
                        suggestion_service.generate_suggestions(
                            query=sanitized_query,
                            search_results=search_results,
                            intent_analysis=intent_analysis
                        ),
                        return_exceptions=True
                    ),
                    timeout=4.0  # 4 second timeout for AI calls
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
            
            # 5. Format the enhanced response for Slack
            slack_response = await format_enhanced_slack_response(
                sanitized_query, 
                search_results, 
                ai_response, 
                suggestions,
                intent_analysis,
                query_id,
                context_analysis,
                db
            )
            
            logger.info(f"Slack response structure: {slack_response}")
            logger.info(f"Slack response type: {type(slack_response)}")
            
            # Store the enhanced response
            query.response = {
                "query_text": sanitized_query,
                "original_query": query_text,
                "resolved_query": resolved_query if resolved_query != query_text.strip() else None,
                "contextual_references": [
                    {
                        "original_term": ref.original_term,
                        "resolved_entity": ref.resolved_entity,
                        "confidence": ref.confidence,
                        "resolution_type": ref.resolution_type
                    }
                    for ref in contextual_references
                ],
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
            
            # Send response to Slack if we have a response URL
            logger.info(f"Response URL: {response_url}")
            if response_url:
                logger.info("Sending response to Slack via response URL")
                await send_slack_response(response_url, slack_response)
            
            # If it's a slash command, also send to the channel
            if is_slash_command:
                logger.info("Sending response to Slack channel")
                await send_channel_response(channel_id, workspace_id, slack_response)
            
            logger.info(f"Successfully processed query {query_id}")
            
            return {
                "status": "success",
                "query_id": query_id,
                "response": slack_response,
                "search_results_count": len(search_results)
            }
            
        except Exception as e:
            logger.error("Error processing query: {}", str(e), exc_info=True)
            await db.rollback()
            
            # Send error response to Slack
            error_response = {
                "response_type": "ephemeral",
                "text": "❌ Sorry, I encountered an error processing your question. Please try again in a moment."
            }
            
            if response_url:
                await send_slack_response(response_url, error_response)
            
            if is_slash_command:
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
        
        # Use intelligent prompt if context analysis is available
        if context_analysis and context_analysis.get("intelligence_level") in ["high", "medium"]:
            conv_intelligence = ConversationalIntelligence()
            system_prompt = await conv_intelligence.generate_intelligent_response_prompt(
                query, search_results, context_analysis, db
            )
        else:
            # Create intent-aware system prompt
            intent = intent_analysis.get("intent", "general_info")
            entities = intent_analysis.get("entities", {})
            scope = intent_analysis.get("scope", "general")
            
            system_prompt = f"""You are a team knowledge assistant who provides SPECIFIC, ACTIONABLE answers from your team's conversation history.

DETECTED INTENT: {intent}
SCOPE: {scope}
ENTITIES: {entities}

🎯 YOUR MISSION:
Transform stored knowledge into immediately useful answers that help the user take action or understand exactly what happened.

📋 RESPONSE REQUIREMENTS:

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

🔍 RESPONSE QUALITY STANDARDS:
- LEAD WITH THE ANSWER: Start with the specific information they need
- INCLUDE SPECIFICS: Names, dates, tools, exact steps, reasoning
- PROVIDE CONTEXT: Why this matters, what led to this decision
- CITE SOURCES: Who said what and when
- SUGGEST NEXT STEPS: What the user should do with this information

❌ AVOID VAGUE RESPONSES:
- "The team discussed database migration" → TOO VAGUE
- "There was a conversation about this topic" → USELESS
- "Some decisions were made" → UNHELPFUL

✅ PROVIDE RICH RESPONSES:
- "John (Senior Engineer) decided on March 15th to migrate from MySQL to PostgreSQL because of better JSON support needed for the new analytics features. Timeline: Migration by end of Q2. Next steps: John is creating the migration plan by March 22nd, Sarah is reviewing infrastructure requirements."

🏷️ SOURCE ATTRIBUTION:
Always include:
- WHO provided the information (with their role if available)
- WHEN the conversation happened
- WHICH channel or thread this came from
- Links to original discussions when possible

IMPORTANT RULES:
1. Extract ALL relevant details from the knowledge items
2. If knowledge is incomplete, specify exactly what's missing
3. Prioritize actionable information over background
4. Use direct quotes when they add value
5. Connect related pieces of knowledge when relevant
6. Tailor depth to the user's intent: {intent}"""

        user_message = f"""Question: {query}

Available Knowledge:
{json.dumps(context, indent=2)}

Please provide a helpful answer based on this knowledge. If the knowledge is insufficient, say so clearly."""

        # Generate response
        response = await openai_service._make_request(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        content = response['choices'][0]['message']['content']
        
        return {
            "response": content,
            "model_used": "gpt-3.5-turbo",
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
        system_prompt = """You are a team knowledge assistant who provides SPECIFIC, ACTIONABLE answers from your team's conversation history.

🎯 YOUR MISSION:
Transform stored knowledge into immediately useful answers with specific details, not vague summaries.

📋 RESPONSE REQUIREMENTS:
- LEAD WITH THE ANSWER: Start with the specific information they need
- INCLUDE SPECIFICS: Names, dates, tools, exact steps, reasoning  
- PROVIDE CONTEXT: Why this matters, what led to this decision
- CITE SOURCES: Who said what and when
- SUGGEST NEXT STEPS: What the user should do with this information

❌ AVOID VAGUE RESPONSES:
- "The team discussed database migration" → TOO VAGUE
- "There was a conversation about this topic" → USELESS

✅ PROVIDE RICH RESPONSES:
- "John decided on March 15th to migrate from MySQL to PostgreSQL because of better JSON support needed for analytics. Timeline: Migration by end of Q2. Next steps: John is creating the migration plan by March 22nd."

IMPORTANT RULES:
1. Extract ALL relevant details from the knowledge items
2. If knowledge is incomplete, specify exactly what's missing
3. Prioritize actionable information over background
4. Use direct quotes when they add value
5. Connect related pieces of knowledge when relevant"""

        user_message = f"""Question: {query}

Available Knowledge:
{json.dumps(context, indent=2)}

Please provide a helpful answer based on this knowledge. If the knowledge is insufficient, say so clearly."""

        # Generate response
        response = await openai_service._make_request(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=500
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
            "model_used": "gpt-3.5-turbo",
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

def _assess_response_quality(search_results: List[Dict[str, Any]], ai_response: Dict[str, Any], context_analysis: Optional[Dict[str, Any]] = None) -> str:
    """Assess the quality of the response to determine formatting approach."""
    # Check if AI response indicates it found useful information
    ai_text = ai_response.get('response', '').lower()
    
    # If we have high intelligence context, this is likely a good response
    if context_analysis and context_analysis.get("intelligence_level") == "high":
        # Check if this is a follow-up or status response
        if any(pattern in ai_text for pattern in ["follow-up", "status check", "what happened"]):
            return "excellent"
    
    # First check for positive indicators - if AI provides specific information, it's good
    positive_indicators = [
        "decided", "decision", "team chose", "recommended", "agreed",
        "timeline", "next steps", "action items", "responsible",
        "because", "due to", "reasoning", "steps:", "procedure",
        "solution", "approach", "method", "process"
    ]
    
    has_positive_content = any(indicator in ai_text for indicator in positive_indicators)
    
    # Check for completely negative responses
    complete_no_info_indicators = [
        "i'm sorry, but the provided",
        "couldn't find any",
        "no information available",
        "don't have information about",
        "no discussions about",
        "no relevant information"
    ]
    
    # Only classify as no_info if it's a completely negative response
    is_complete_no_info = any(indicator in ai_text for indicator in complete_no_info_indicators)
    
    # If it has positive content but mentions some details are missing, it's still partial
    partial_info_indicators = [
        "unfortunately", "however", "details about", "specific steps", 
        "timeline are not available", "more information needed"
    ]
    
    has_partial_limitations = any(indicator in ai_text for indicator in partial_info_indicators)
    
    # Determine quality based on content analysis
    if is_complete_no_info:
        return "no_info"
    elif has_positive_content:
        if has_partial_limitations:
            return "partial"  # Has good info but acknowledges gaps
        else:
            return "excellent"  # Has good info without major gaps
    
    # Fallback to search results analysis
    high_confidence_results = [r for r in search_results if r.get('confidence', 0) > 0.7]
    knowledge_results = [r for r in search_results if r.get('type') != 'conversation']
    
    if high_confidence_results or (knowledge_results and len(knowledge_results) > 1):
        return "excellent"
    elif search_results:
        return "partial"
    else:
        return "no_info"

def _extract_source_attribution(search_results: List[Dict[str, Any]]) -> str:
    """Extract clean source attribution from search results."""
    if not search_results:
        return ""
    
    # Look for the best source (highest confidence or most relevant)
    best_result = max(search_results, key=lambda x: x.get('confidence', 0))
    
    # Extract user and timestamp info from conversation results
    if best_result.get('type') == 'conversation':
        user_id = best_result.get('slack_user_id', '')
        if user_id:
            return f"Discussion with @{user_id}"
    
    # For knowledge items, try to extract from metadata
    metadata = best_result.get('metadata', {})
    if metadata:
        source_user = metadata.get('source_user_id', '')
        if source_user:
            return f"Shared by @{source_user}"
    
    return "Team discussion"

def _format_excellent_response(query: str, search_results: List[Dict[str, Any]], ai_response: Dict[str, Any]) -> str:
    """Format response when we have good, actionable information."""
    content = ai_response.get('response', '')
    
    # Extract the main topic/action from the query for a good title
    title = _generate_response_title(query, search_results)
    
    response = f"✅ **{title}**\n\n{content}"
    return response

def _format_partial_response(query: str, search_results: List[Dict[str, Any]], ai_response: Dict[str, Any]) -> str:
    """Format response when we have some information but it's incomplete."""
    content = ai_response.get('response', '')
    
    # Check if this is actually partial or if AI is being overly cautious
    cautious_indicators = ["sorry", "don't have", "unfortunately", "not provided", "specific steps are not"]
    if any(indicator in content.lower() for indicator in cautious_indicators):
        # AI is being overly cautious, try to extract useful info from search results
        useful_results = [r for r in search_results if r.get('confidence', 0) > 0.5]
        conversation_results = [r for r in search_results if r.get('type') == 'conversation']
        
        if useful_results or conversation_results:
            title = _generate_response_title(query, search_results)
            
            # Extract the useful part of the AI response (before the disclaimer)
            useful_content = content.split("Unfortunately")[0].split("However")[0].strip()
            if len(useful_content) < 50:  # If too short, use search results
                useful_content = _extract_useful_content(useful_results or conversation_results)
            
            response = f"⚠️ **Partial Information Found**\n\n**What I found:** {useful_content}"
            
            # Add what's missing with specific guidance
            if "steps" in query.lower() or "how to" in query.lower():
                response += f"\n\n**What's missing:** Detailed step-by-step instructions"
                response += f"\n\n**Next steps:** Ask @{_extract_likely_expert(search_results)} for the complete process"
            else:
                response += f"\n\n**What's missing:** Complete details weren't captured in our knowledge base"
                response += f"\n\n**Next steps:** Check with your team for more information"
            return response
    
    title = _generate_response_title(query, search_results)
    return f"⚠️ **{title}**\n\n{content}"

def _format_no_info_response(query: str, search_results: List[Dict[str, Any]], suggestions: List[str]) -> str:
    """Format response when no relevant information is found."""
    response = f"🔍 **No information found in your team's knowledge base**\n\n"
    response += f"I couldn't find any discussions about {_extract_query_topic(query)} in your Slack history.\n\n"
    response += "**Suggestions:**\n"
    response += "• Ask your team - they might have discussed this in a channel I haven't indexed yet\n"
    response += "• Check if there's existing documentation for this process\n"
    response += "• Tag someone who might know about this topic"
    
    return response

def _generate_response_title(query: str, search_results: List[Dict[str, Any]]) -> str:
    """Generate a clean, descriptive title for the response."""
    # Look for process-related keywords
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['steps', 'how to', 'process', 'migrate', 'restart']):
        if 'kafka' in query_lower:
            return "Kafka Connector Process"
        elif 'migration' in query_lower or 'migrate' in query_lower:
            return "Database Migration Steps"
        elif 'restart' in query_lower:
            return "Restart Process"
        else:
            return "Process Steps"
    elif any(word in query_lower for word in ['what is', 'explain', 'about']):
        return "Information Found"
    else:
        return "Team Knowledge"

def _extract_query_topic(query: str) -> str:
    """Extract the main topic from a user query."""
    # Simple extraction - could be made more sophisticated
    query_lower = query.lower()
    
    if 'migration' in query_lower:
        return "database migration"
    elif 'kafka' in query_lower:
        return "Kafka processes"
    elif 'restart' in query_lower:
        return "restart procedures"
    else:
        return "this topic"

def _extract_useful_content(search_results: List[Dict[str, Any]]) -> str:
    """Extract the most useful content from search results."""
    # Find the best result
    best_result = max(search_results, key=lambda x: x.get('confidence', 0))
    
    content = best_result.get('content', '')
    if len(content) > 200:
        content = content[:200] + "..."
    
    return content

async def _resolve_user_mentions_in_response(response_text: str, db: AsyncSession) -> str:
    """Replace Slack user IDs with actual usernames in response text."""
    import re
    
    # Find all @U... patterns
    user_id_pattern = r'@(U[A-Z0-9]+)'
    user_ids = re.findall(user_id_pattern, response_text)
    
    resolved_text = response_text
    for slack_id in user_ids:
        # Look up the actual username
        result = await db.execute(
            select(User).where(User.slack_id == slack_id)
        )
        user = result.scalar_one_or_none()
        if user:
            # Replace @U0417Q7L4A1 with @username
            # Clean up the username - if it's "User_UXXXXXX", use a friendlier format
            raw_username = user.name or slack_id
            if raw_username.startswith("User_U"):
                # Extract the user ID part and make it friendlier
                username = raw_username.replace("User_", "").lower()
            else:
                username = raw_username
            resolved_text = resolved_text.replace(f'@{slack_id}', f'@{username}')
    
    return resolved_text

def _extract_likely_expert(search_results: List[Dict[str, Any]]) -> str:
    """Extract the most likely expert from search results."""
    # Look for users mentioned in conversation results
    for result in search_results:
        if result.get('type') == 'conversation':
            metadata = result.get('metadata', {})
            user_id = metadata.get('slack_user_id')
            if user_id:
                return user_id
    
    # Fallback to a generic suggestion
    return "your team"

async def format_enhanced_slack_response(
    query: str, 
    search_results: List[Dict[str, Any]], 
    ai_response: Dict[str, Any], 
    suggestions: List[str],
    intent_analysis: Dict[str, Any],
    query_id: Optional[int] = None,
    context_analysis: Optional[Dict[str, Any]] = None,
    db: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """Format clean, user-friendly response for Slack display."""
    try:
        logger.info(f"AI response structure: {ai_response}")
        
        # Determine response quality and format accordingly
        response_quality = _assess_response_quality(search_results, ai_response, context_analysis)
        
        if response_quality == "excellent":
            response_text = _format_excellent_response(query, search_results, ai_response)
        elif response_quality == "partial":
            response_text = _format_partial_response(query, search_results, ai_response)
        else:
            response_text = _format_no_info_response(query, search_results, suggestions)
        
        # Add clean source attribution if available
        source_info = _extract_source_attribution(search_results)
        if source_info:
            response_text += f"\n\n**Source:** {source_info}"
        
        # Resolve user mentions to actual usernames
        if db:
            response_text = await _resolve_user_mentions_in_response(response_text, db)
        
        return {
            "response_type": "in_channel",
            "text": response_text,
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": response_text
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "👍 Helpful"
                            },
                            "style": "primary",
                            "action_id": "feedback_helpful",
                            "value": str(query_id) if query_id else "0"
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "👎 Not Helpful"
                            },
                            "action_id": "feedback_not_helpful",
                            "value": str(query_id) if query_id else "0"
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "🔗 View Sources"
                            },
                            "action_id": "view_sources",
                            "value": str(query_id) if query_id else "0"
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "⚠️ Report Issue"
                            },
                            "action_id": "report_issue",
                            "value": str(query_id) if query_id else "0"
                        }
                    ]
                }
            ]
        }
        
    except Exception as e:
        logger.error(f"Error formatting enhanced Slack response: {e}")
        return {
            "response_type": "ephemeral",
            "text": "❌ Error formatting response. Please try again."
        }

def format_slack_response(query: str, search_results: List[Dict[str, Any]], ai_response: Dict[str, Any]) -> Dict[str, Any]:
    """Format the response for Slack display."""
    try:
        # Start with the AI response
        logger.info(f"AI response structure: {ai_response}")
        if ai_response.get("fallback"):
            # Use fallback response
            response_text = ai_response["response"]
        else:
            response_text = f"🤖 *AI Response:*\n{ai_response['response']}"
        
        # Add knowledge items
        if search_results:
            response_text += "\n\n📚 *Relevant Knowledge:*\n"
            
            for i, result in enumerate(search_results[:3], 1):
                confidence_emoji = "🟢" if result.get("confidence", 0) >= 0.8 else "🟡" if result.get("confidence", 0) >= 0.6 else "🔴"
                response_text += f"\n{i}. {confidence_emoji} *{result.get('title', 'Untitled')}*\n"
                response_text += f"   {result.get('summary', 'No summary')}\n"
                response_text += f"   Confidence: {result.get('confidence', 0):.1%}"
        else:
            response_text += "\n\n❌ *No relevant knowledge found*\nI couldn't find any information related to your question in our knowledge base."
        
        # Add footer
        response_text += f"\n\n💡 *Question:* {query}"
        
        return {
            "response_type": "in_channel",
            "text": response_text,
            "attachments": [
                {
                    "text": f"Processed at {datetime.utcnow().strftime('%H:%M:%S UTC')}",
                    "color": "#36a64f"
                }
            ]
        }
        
    except Exception as e:
        logger.error(f"Error formatting Slack response: {e}")
        return {
            "response_type": "ephemeral",
            "text": "❌ Error formatting response. Please try again."
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

async def send_channel_response(channel_id: str, workspace_id: int, response_data: Dict[str, Any]):
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
                    # Send message to channel
                    result = await slack_service.send_message(
                        channel=channel_id,
                        text=response_data.get("text", "Response generated"),
                        thread_ts=None,  # Main message, not threaded
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
