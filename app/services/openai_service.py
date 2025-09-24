"""OpenAI service for AI-powered knowledge extraction with rate limiting and error handling."""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from loguru import logger
import openai
from openai import AsyncOpenAI
import backoff

from ..core.config import settings

class OpenAIService:
    """Service for interacting with OpenAI APIs with rate limiting and error handling."""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,            timeout=30.0  # 30 second timeout for API calls (increased for synthetic generation)
        )
        self.rate_limit_tokens = 0
        self.rate_limit_requests = 0
        self.last_request_time = 0
        self.min_request_interval = 0.05  # 50ms between requests (faster)
        
        # Rate limiting configuration
        self.max_tokens_per_minute = 90000  # GPT-4 rate limit
        self.max_requests_per_minute = 3500  # GPT-4 rate limit
        self.token_reset_time = None
        self.request_reset_time = None
        
        # Retry configuration
        self.max_retries = 3
        self.base_delay = 1.0
        self.max_delay = 60.0
        
        logger.info("OpenAI service initialized")
    
    async def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        
        # Wait minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        # Check token rate limit
        if self.token_reset_time and current_time < self.token_reset_time:
            if self.rate_limit_tokens >= self.max_tokens_per_minute:
                wait_time = self.token_reset_time - current_time
                logger.warning(f"Token rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Check request rate limit
        if self.request_reset_time and current_time < self.request_reset_time:
            if self.rate_limit_requests >= self.max_requests_per_minute:
                wait_time = self.request_reset_time - current_time
                logger.warning(f"Request rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _update_rate_limits(self, response: Dict[str, Any]):
        """Update rate limit tracking from API response."""
        # Update token usage
        if 'usage' in response:
            tokens_used = response['usage'].get('total_tokens', 0)
            self.rate_limit_tokens += tokens_used
            
            # Reset token counter every minute
            if not self.token_reset_time or time.time() >= self.token_reset_time:
                self.token_reset_time = time.time() + 60
                self.rate_limit_tokens = tokens_used
        
        # Update request counter
        self.rate_limit_requests += 1
        
        # Reset request counter every minute
        if not self.request_reset_time or time.time() >= self.request_reset_time:
            self.request_reset_time = time.time() + 60
            self.rate_limit_requests = 1
    
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError),
        max_tries=3
    )
    async def _make_request(self, **kwargs) -> Dict[str, Any]:
        """Make a request to OpenAI with retry logic."""
        try:
            await self._wait_for_rate_limit()
            
            response = await self.client.chat.completions.create(**kwargs)
            
            # Update rate limit tracking
            self._update_rate_limits(response.model_dump())
            
            return response.model_dump()
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            # Wait for rate limit to reset
            await asyncio.sleep(60)
            raise
            
        except openai.APITimeoutError as e:
            logger.warning(f"OpenAI API timeout: {e}")
            raise
            
        except openai.APIConnectionError as e:
            logger.warning(f"OpenAI API connection error: {e}")
            raise
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI request: {e}")
            raise
    
    async def extract_knowledge(
        self, 
        conversation_context: List[Dict[str, Any]], 
        message_text: str,
        message_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract knowledge from a message using OpenAI."""
        try:
            # Prepare conversation context
            context_messages = self._prepare_conversation_context(conversation_context)
            
            # Create system prompt for knowledge extraction
            system_prompt = self._create_knowledge_extraction_prompt()
            
            # Create user message
            user_message = self._create_user_message(message_text, message_metadata)
            
            # Make API request
            response = await self._make_request(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *context_messages,
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response['choices'][0]['message']['content']
            extraction_result = json.loads(content)
            
            # Validate and enhance the result
            enhanced_result = self._enhance_extraction_result(
                extraction_result, message_text, message_metadata
            )
            
            logger.info(f"Knowledge extraction completed for message: {message_metadata.get('message_id')}")
            return enhanced_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            return self._create_fallback_extraction(message_text, message_metadata)
            
        except Exception as e:
            logger.error(f"Error in knowledge extraction: {e}")
            return self._create_fallback_extraction(message_text, message_metadata)
    
    def _prepare_conversation_context(self, conversation_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare conversation context for the AI model."""
        context_messages = []
        
        for msg in conversation_context[-10:]:  # Last 10 messages for context
            role = "assistant" if msg.get("is_bot") else "user"
            content = msg.get("text", "")
            
            if content:
                context_messages.append({
                    "role": role,
                    "content": f"{msg.get('user_name', 'User')}: {content}"
                })
        
        return context_messages
    
    def _create_knowledge_extraction_prompt(self) -> str:
        """Create the system prompt for knowledge extraction."""
        return """You are an expert knowledge extraction system. Your task is to identify and extract valuable knowledge from Slack conversations.

KNOWLEDGE TYPES TO EXTRACT:
1. DECISIONS: Clear decisions made by individuals or teams
2. PROCESSES: Workflows, procedures, or methodologies described
3. SOLUTIONS: Technical solutions, workarounds, or answers to problems
4. INSIGHTS: Valuable observations, learnings, or realizations
5. REQUIREMENTS: Specifications, needs, or constraints mentioned

EXTRACTION RULES:
- Only extract knowledge that is explicitly stated or clearly implied
- Maintain high confidence standards - if uncertain, mark confidence as low
- Provide specific, actionable information
- Include relevant context and participants
- Avoid speculation or assumptions

OUTPUT FORMAT (JSON):
{
  "knowledge_items": [
    {
      "type": "decision|process|solution|insight|requirement",
      "title": "Brief descriptive title",
      "summary": "1-2 sentence summary",
      "content": "Detailed description",
      "confidence": 0.0-1.0,
      "participants": ["user1", "user2"],
      "tags": ["tag1", "tag2"],
      "source_context": "Relevant conversation context"
    }
  ],
  "overall_confidence": 0.0-1.0,
  "extraction_notes": "Any notes about the extraction process"
}"""
    
    def _create_user_message(self, message_text: str, message_metadata: Dict[str, Any]) -> str:
        """Create the user message for knowledge extraction."""
        return f"""Please extract knowledge from this message:

MESSAGE TEXT:
{message_text}

MESSAGE METADATA:
- Type: {message_metadata.get('message_type', 'unknown')}
- Significance Score: {message_metadata.get('significance_score', 0)}
- Is Thread Reply: {message_metadata.get('is_thread_reply', False)}
- Word Count: {message_metadata.get('word_count', 0)}

Extract any valuable knowledge following the specified format and rules."""
    
    def _enhance_extraction_result(
        self, 
        extraction_result: Dict[str, Any], 
        message_text: str, 
        message_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance the extraction result with additional metadata."""
        enhanced_result = extraction_result.copy()
        
        # Add extraction metadata
        enhanced_result["extraction_metadata"] = {
            "extracted_at": datetime.utcnow().isoformat(),
            "model_used": "gpt-3.5-turbo",
            "message_id": message_metadata.get("message_id"),
            "message_type": message_metadata.get("message_type"),
            "significance_score": message_metadata.get("significance_score"),
            "processing_time": time.time() - self.last_request_time
        }
        
        # Validate knowledge items
        if "knowledge_items" in enhanced_result:
            for item in enhanced_result["knowledge_items"]:
                # Ensure required fields
                item.setdefault("confidence", 0.5)
                item.setdefault("participants", [])
                item.setdefault("tags", [])
                item.setdefault("source_context", message_text[:200])
                
                # Validate confidence score
                item["confidence"] = max(0.0, min(1.0, float(item["confidence"])))
        
        return enhanced_result
    
    def _create_fallback_extraction(self, message_text: str, message_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback extraction result when AI extraction fails."""
        return {
            "knowledge_items": [],
            "overall_confidence": 0.0,
            "extraction_notes": "AI extraction failed, using fallback method",
            "extraction_metadata": {
                "extracted_at": datetime.utcnow().isoformat(),
                "model_used": "fallback",
                "message_id": message_metadata.get("message_id"),
                "message_type": message_metadata.get("message_type"),
                "significance_score": message_metadata.get("significance_score"),
                "error": "AI extraction failed"
            }
        }
    
    async def verify_extraction(
        self, 
        extracted_knowledge: Dict[str, Any], 
        source_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify extracted knowledge against source messages to prevent hallucination."""
        try:
            # Create verification prompt
            system_prompt = """You are a fact-checking system. Verify if the extracted knowledge is supported by the source messages.

Your task is to:
1. Check if each knowledge item is directly supported by the source messages
2. Identify any fabricated or unsupported information
3. Provide confidence scores for verification
4. Suggest corrections if needed

Output format (JSON):
{
  "verification_results": [
    {
      "knowledge_item_index": 0,
      "is_supported": true/false,
      "confidence": 0.0-1.0,
      "supporting_evidence": ["quote1", "quote2"],
      "issues": ["issue1", "issue2"],
      "suggestions": ["suggestion1", "suggestion2"]
    }
  ],
  "overall_verification_score": 0.0-1.0,
  "hallucination_detected": true/false
}"""
            
            # Prepare source messages for verification
            source_text = "\n\n".join([
                f"{msg.get('user_name', 'User')}: {msg.get('text', '')}"
                for msg in source_messages
            ])
            
            user_message = f"""Please verify this extracted knowledge against the source messages:

EXTRACTED KNOWLEDGE:
{json.dumps(extracted_knowledge, indent=2)}

SOURCE MESSAGES:
{source_text}

Verify each knowledge item and identify any hallucinations or unsupported claims."""
            
            # Make verification request
            response = await self._make_request(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,  # Zero temperature for verification
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            content = response['choices'][0]['message']['content']
            verification_result = json.loads(content)
            
            # Add verification metadata
            verification_result["verification_metadata"] = {
                "verified_at": datetime.utcnow().isoformat(),
                "model_used": "gpt-3.5-turbo",
                "source_message_count": len(source_messages)
            }
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Error in knowledge verification: {e}")
            return {
                "verification_results": [],
                "overall_verification_score": 0.0,
                "hallucination_detected": False,
                "verification_metadata": {
                    "verified_at": datetime.utcnow().isoformat(),
                    "model_used": "fallback",
                    "error": str(e)
                }
            }
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get the current status of the OpenAI service."""
        return {
            "status": "operational",
            "rate_limits": {
                "tokens_used_this_minute": self.rate_limit_tokens,
                "requests_this_minute": self.rate_limit_requests,
                "token_reset_in": max(0, (self.token_reset_time or 0) - time.time()),
                "request_reset_in": max(0, (self.request_reset_time or 0) - time.time())
            },
            "last_request_time": self.last_request_time,
            "api_key_configured": bool(settings.openai_api_key)
        }
