"""Service for generating follow-up question suggestions based on query context."""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

from .openai_service import OpenAIService
from .intent_classifier import IntentClassifier

class SuggestionService:
    """Service for generating contextual follow-up suggestions."""
    
    def __init__(self):
        self.openai_service = OpenAIService()
        self.intent_classifier = IntentClassifier()
        
        # Template suggestions by intent
        self.intent_suggestions = {
            "process": [
                "What tools are needed for this process?",
                "Who is responsible for this process?",
                "What are the common issues with this process?",
                "How long does this process typically take?",
                "What are the prerequisites for this process?"
            ],
            "decision_rationale": [
                "What alternatives were considered?",
                "Who made this decision?",
                "When was this decision made?",
                "What factors influenced this decision?",
                "Has this decision been reviewed recently?"
            ],
            "resource_link": [
                "Who maintains this resource?",
                "When was this resource last updated?",
                "Are there similar resources available?",
                "What's the best way to access this resource?",
                "Is there documentation for this resource?"
            ],
            "general_info": [
                "Can you provide more details about this?",
                "What are the key components of this?",
                "How does this relate to other systems?",
                "What are the benefits of this approach?",
                "Are there any limitations to consider?"
            ],
            "status_check": [
                "What's the expected completion date?",
                "Who is working on this?",
                "What are the current blockers?",
                "How can I help with this?",
                "What's the next milestone?"
            ],
            "person_related": [
                "What is their role in this project?",
                "How can I contact them?",
                "What other projects are they working on?",
                "Who reports to them?",
                "What are their main responsibilities?"
            ],
            "timeline_related": [
                "What are the key milestones?",
                "Who is managing this timeline?",
                "What could cause delays?",
                "How is progress being tracked?",
                "What happens if we miss the deadline?"
            ]
        }
        
        # Context-based suggestion patterns
        self.context_patterns = {
            "migration": [
                "What's the migration strategy?",
                "What are the risks involved?",
                "How long will the migration take?",
                "What's the rollback plan?",
                "Who is leading the migration?"
            ],
            "database": [
                "What's the current database schema?",
                "What are the performance considerations?",
                "How is data being backed up?",
                "What's the maintenance schedule?",
                "Who has access to the database?"
            ],
            "deployment": [
                "What's the deployment process?",
                "How is the deployment monitored?",
                "What's the rollback procedure?",
                "Who approves deployments?",
                "What environments are available?"
            ],
            "api": [
                "What's the API documentation?",
                "How is the API versioned?",
                "What are the rate limits?",
                "How is API usage monitored?",
                "Who maintains the API?"
            ],
            "security": [
                "What security measures are in place?",
                "How are permissions managed?",
                "What's the incident response plan?",
                "How often are security audits done?",
                "Who handles security issues?"
            ]
        }
    
    async def generate_suggestions(
        self, 
        query: str, 
        search_results: List[Dict[str, Any]], 
        intent_analysis: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate contextual follow-up suggestions based on query and results."""
        try:
            # Get intent analysis if not provided
            if not intent_analysis:
                intent_analysis = await self.intent_classifier.classify_intent(query)
            
            # Generate suggestions using multiple methods
            intent_suggestions = self._get_intent_based_suggestions(intent_analysis)
            context_suggestions = self._get_context_based_suggestions(query, search_results)
            ai_suggestions = await self._generate_ai_suggestions(query, search_results, intent_analysis)
            
            # Combine and rank suggestions
            all_suggestions = intent_suggestions + context_suggestions + ai_suggestions
            
            # Remove duplicates and rank by relevance
            unique_suggestions = self._deduplicate_suggestions(all_suggestions)
            ranked_suggestions = self._rank_suggestions(unique_suggestions, query, intent_analysis)
            
            # Return top 3-5 suggestions
            return ranked_suggestions[:5]
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return self._get_fallback_suggestions(query)
    
    def _get_intent_based_suggestions(self, intent_analysis: Dict[str, Any]) -> List[str]:
        """Get suggestions based on detected intent."""
        intent = intent_analysis.get("intent", "general_info")
        return self.intent_suggestions.get(intent, self.intent_suggestions["general_info"]).copy()
    
    def _get_context_based_suggestions(self, query: str, search_results: List[Dict[str, Any]]) -> List[str]:
        """Get suggestions based on context keywords found in query and results."""
        query_lower = query.lower()
        suggestions = []
        
        # Check for context keywords
        for context, context_suggestions in self.context_patterns.items():
            if context in query_lower:
                suggestions.extend(context_suggestions)
        
        # Check search results for context
        for result in search_results:
            content = result.get("content", "").lower()
            for context, context_suggestions in self.context_patterns.items():
                if context in content and context not in query_lower:
                    suggestions.extend(context_suggestions[:2])  # Limit per context
        
        return suggestions
    
    async def _generate_ai_suggestions(
        self, 
        query: str, 
        search_results: List[Dict[str, Any]], 
        intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate AI-powered contextual suggestions."""
        try:
            # Prepare context from search results
            context_summary = self._summarize_search_results(search_results)
            
            system_prompt = """You are an expert at generating helpful follow-up questions. 
            Based on the user's original question and the available knowledge context, 
            generate 3-5 relevant follow-up questions that would be valuable for the user.

            Guidelines:
            - Questions should be specific and actionable
            - Focus on practical next steps or related topics
            - Avoid overly broad or generic questions
            - Consider the user's likely role and needs
            - Make questions that can be answered with available knowledge

            Return a JSON array of question strings."""

            user_message = f"""Original Question: {query}

            Intent Analysis: {intent_analysis.get('intent', 'unknown')}
            Entities Found: {intent_analysis.get('entities', {})}
            Scope: {intent_analysis.get('scope', 'general')}

            Available Knowledge Context:
            {context_summary}

            Generate 3-5 relevant follow-up questions."""

            response = await self.openai_service._make_request(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_completion_tokens=300
            )
            
            content = response['choices'][0]['message']['content']
            suggestions = json.loads(content)
            
            return suggestions if isinstance(suggestions, list) else []
            
        except Exception as e:
            logger.error(f"Error generating AI suggestions: {e}")
            return []
    
    def _summarize_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Create a summary of search results for AI context."""
        if not search_results:
            return "No specific knowledge found."
        
        summary_parts = []
        for i, result in enumerate(search_results[:3], 1):
            title = result.get("title", "Untitled")
            summary = result.get("summary", "No summary available")
            confidence = result.get("confidence", 0.0)
            
            summary_parts.append(f"{i}. {title} (Confidence: {confidence:.1%})\n   {summary}")
        
        return "\n\n".join(summary_parts)
    
    def _deduplicate_suggestions(self, suggestions: List[str]) -> List[str]:
        """Remove duplicate suggestions while preserving order."""
        seen = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            # Normalize for comparison
            normalized = suggestion.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    def _rank_suggestions(
        self, 
        suggestions: List[str], 
        original_query: str, 
        intent_analysis: Dict[str, Any]
    ) -> List[str]:
        """Rank suggestions by relevance to the original query."""
        query_words = set(original_query.lower().split())
        intent = intent_analysis.get("intent", "general_info")
        entities = intent_analysis.get("entities", {})
        
        def score_suggestion(suggestion: str) -> float:
            score = 0.0
            suggestion_lower = suggestion.lower()
            
            # Word overlap with original query
            suggestion_words = set(suggestion_lower.split())
            word_overlap = len(query_words.intersection(suggestion_words))
            score += word_overlap * 0.3
            
            # Intent alignment
            if intent in suggestion_lower:
                score += 0.4
            
            # Entity relevance
            if isinstance(entities, dict):
                for entity_type, entity_list in entities.items():
                    if isinstance(entity_list, list):
                        for entity in entity_list:
                            if entity.lower() in suggestion_lower:
                                score += 0.2
            elif isinstance(entities, list):
                for entity in entities:
                    if entity.lower() in suggestion_lower:
                        score += 0.2
            
            # Length preference (not too short, not too long)
            word_count = len(suggestion_words)
            if 5 <= word_count <= 15:
                score += 0.1
            
            return score
        
        # Sort by score (descending)
        return sorted(suggestions, key=score_suggestion, reverse=True)
    
    def _get_fallback_suggestions(self, query: str) -> List[str]:
        """Get fallback suggestions when generation fails."""
        return [
            "Can you provide more details about this?",
            "What are the next steps?",
            "Who should I contact for more information?",
            "Are there any related resources?",
            "What should I know about this topic?"
        ]
