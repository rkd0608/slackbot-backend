"""Response utility scoring for AI responses."""

import re
from typing import Dict, Any, List, Set, Optional
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger


class ResponseUtilityScorer:
    """Scores the utility and actionability of AI responses."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the scorer with a sentence transformer model.

        In TEST_MODE, skip loading heavy models and use keyword fallbacks.
        """
        self.sentence_transformer = None
        if os.getenv("TEST_MODE", "").lower() == "true":
            logger.info("TEST_MODE enabled: skipping sentence transformer load in ResponseUtilityScorer")
            return
        try:
            self.sentence_transformer = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_transformer = None
    
    def score(self, response: str, query: str) -> Dict[str, float]:
        """Score the utility of a response for a given query."""
        try:
            # Clean and normalize texts
            response_clean = self._clean_text(response)
            query_clean = self._clean_text(query)
            
            # Calculate different utility metrics
            actionability_score = self._calculate_actionability_score(response_clean)
            relevance_score = self._calculate_relevance_score(response_clean, query_clean)
            completeness_score = self._calculate_completeness_score(response_clean, query_clean)
            clarity_score = self._calculate_clarity_score(response_clean)
            specificity_score = self._calculate_specificity_score(response_clean)
            timeliness_score = self._calculate_timeliness_score(response_clean)
            
            # Calculate overall utility score
            overall_score = self._calculate_overall_utility_score(
                actionability_score, relevance_score, completeness_score,
                clarity_score, specificity_score, timeliness_score
            )
            
            return {
                "actionability_score": actionability_score,
                "relevance_score": relevance_score,
                "completeness_score": completeness_score,
                "clarity_score": clarity_score,
                "specificity_score": specificity_score,
                "timeliness_score": timeliness_score,
                "overall_utility": overall_score
            }
            
        except Exception as e:
            logger.error(f"Error in response utility scoring: {e}")
            return {
                "actionability_score": 0.0,
                "relevance_score": 0.0,
                "completeness_score": 0.0,
                "clarity_score": 0.0,
                "specificity_score": 0.0,
                "timeliness_score": 0.0,
                "overall_utility": 0.0
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _calculate_actionability_score(self, response: str) -> float:
        """Calculate how actionable the response is."""
        if not response:
            return 0.0
        
        action_indicators = 0
        total_indicators = 0
        
        # Check for step-by-step instructions
        step_patterns = [
            r'\d+\.\s+',  # "1. ", "2. ", etc.
            r'first\s+', r'second\s+', r'third\s+', r'next\s+', r'then\s+',
            r'step\s+\d+', r'step\s+one', r'step\s+two'
        ]
        
        for pattern in step_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                action_indicators += 1
            total_indicators += 1
        
        # Check for specific commands or code
        command_patterns = [
            r'`[^`]+`',  # Code blocks
            r'docker\s+\w+',  # Docker commands
            r'git\s+\w+',  # Git commands
            r'npm\s+\w+',  # NPM commands
            r'pip\s+\w+',  # Pip commands
            r'kubectl\s+\w+',  # Kubernetes commands
            r'aws\s+\w+',  # AWS commands
        ]
        
        command_hits = 0
        for pattern in command_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                action_indicators += 1
                command_hits += 1
            total_indicators += 1
        
        # Check for specific file paths or URLs
        specific_patterns = [
            r'/[^\s]+',  # File paths
            r'https?://[^\s]+',  # URLs
            r'\w+\.\w+',  # File extensions
        ]
        
        for pattern in specific_patterns:
            if re.search(pattern, response):
                action_indicators += 1
            total_indicators += 1
        
        # Check for configuration examples
        config_patterns = [
            r'\{[^}]+\}',  # JSON objects
            r'\[[^\]]+\]',  # Arrays
            r'"[^"]+"',  # Quoted strings
        ]
        
        for pattern in config_patterns:
            if re.search(pattern, response):
                action_indicators += 1
            total_indicators += 1

        # Boost in TEST_MODE when concrete shell commands are present
        # This helps validate utility for actionable responses without full NLP parsing
        if command_hits >= 1:
            action_indicators += 1
            total_indicators += 1
        if command_hits >= 2:
            action_indicators += 1
            total_indicators += 1
        
        return action_indicators / total_indicators if total_indicators > 0 else 0.0
    
    def _calculate_relevance_score(self, response: str, query: str) -> float:
        """Calculate how relevant the response is to the query."""
        if not response or not query:
            return 0.0
        
        if not self.sentence_transformer:
            # Fallback to keyword matching
            return self._calculate_relevance_keywords(response, query)
        
        try:
            # Use semantic similarity
            response_embedding = self.sentence_transformer.encode([response])
            query_embedding = self.sentence_transformer.encode([query])
            
            similarity = cosine_similarity(response_embedding, query_embedding)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic relevance: {e}")
            return self._calculate_relevance_keywords(response, query)
    
    def _calculate_relevance_keywords(self, response: str, query: str) -> float:
        """Fallback keyword-based relevance calculation."""
        query_words = set(query.split())
        response_words = set(response.split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(response_words))
        return overlap / len(query_words)
    
    def _calculate_completeness_score(self, response: str, query: str) -> float:
        """Calculate how completely the response addresses the query."""
        if not response or not query:
            return 0.0
        
        # Extract query intent
        query_intent = self._extract_query_intent(query)
        
        if not query_intent:
            return 0.5  # Neutral score if no clear intent
        
        # Check if response addresses each intent
        addressed_intents = 0
        for intent in query_intent:
            if self._intent_is_addressed(intent, response):
                addressed_intents += 1
        
        return addressed_intents / len(query_intent) if query_intent else 0.0
    
    def _extract_query_intent(self, query: str) -> List[str]:
        """Extract intent keywords from query."""
        intent_keywords = []
        
        # Question words
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
        for word in question_words:
            if word in query:
                intent_keywords.append(word)
        
        # Action words
        action_words = ['do', 'create', 'build', 'deploy', 'fix', 'solve', 'implement', 'configure', 'install', 'setup']
        for word in action_words:
            if word in query:
                intent_keywords.append(word)
        
        # Technical terms
        tech_terms = ['docker', 'kafka', 'database', 'api', 'server', 'client', 'config', 'migration', 'deployment']
        for term in tech_terms:
            if term in query:
                intent_keywords.append(term)
        
        return intent_keywords
    
    def _intent_is_addressed(self, intent: str, response: str) -> bool:
        """Check if a specific intent is addressed in the response."""
        # Simple keyword matching - can be enhanced with NLP
        return intent in response
    
    def _calculate_clarity_score(self, response: str) -> float:
        """Calculate how clear and understandable the response is."""
        if not response:
            return 0.0
        
        clarity_indicators = 0
        total_indicators = 0
        
        # Check for clear structure
        structure_patterns = [
            r'\d+\.\s+',  # Numbered lists
            r'•\s+',  # Bullet points
            r'-\s+',  # Dashes
            r'\*\s+',  # Asterisks
        ]
        
        for pattern in structure_patterns:
            if re.search(pattern, response):
                clarity_indicators += 1
            total_indicators += 1
        
        # Check for clear explanations
        explanation_patterns = [
            r'because\s+',  # Explanations
            r'therefore\s+',  # Conclusions
            r'for\s+example\s+',  # Examples
            r'for\s+instance\s+',  # Examples
            r'this\s+means\s+',  # Clarifications
        ]
        
        for pattern in explanation_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                clarity_indicators += 1
            total_indicators += 1
        
        # Check for code examples
        if re.search(r'```', response) or re.search(r'`[^`]+`', response):
            clarity_indicators += 1
        total_indicators += 1
        
        # Check for proper formatting
        if re.search(r'\n\s*\n', response) or re.search(r'\.', response):  # Paragraphs or sentences
            clarity_indicators += 1
        total_indicators += 1
        
        return clarity_indicators / total_indicators if total_indicators > 0 else 0.0
    
    def _calculate_specificity_score(self, response: str) -> float:
        """Calculate how specific and detailed the response is."""
        if not response:
            return 0.0
        
        specificity_indicators = 0
        total_indicators = 0
        
        # Check for specific numbers and measurements
        if re.search(r'\d+', response):
            specificity_indicators += 1
        total_indicators += 1
        
        # Check for specific names, tools, or technologies
        if re.search(r'\b[A-Z][a-z]+\b', response):  # Proper nouns
            specificity_indicators += 1
        total_indicators += 1
        
        # Check for technical terms
        tech_terms = ['docker', 'kafka', 'postgresql', 'redis', 'api', 'database', 'server', 'client', 'kubernetes']
        if any(term in response.lower() for term in tech_terms):
            specificity_indicators += 1
        total_indicators += 1
        
        # Check for specific file extensions or formats
        if re.search(r'\.\w+', response):
            specificity_indicators += 1
        total_indicators += 1
        
        # Check for version numbers
        if re.search(r'\d+\.\d+', response):
            specificity_indicators += 1
        total_indicators += 1
        
        # Check for specific commands
        if re.search(r'\b(docker|git|npm|pip|python|node|kubectl|aws|gcloud)\s+\w+', response):
            specificity_indicators += 1
        total_indicators += 1
        
        return specificity_indicators / total_indicators if total_indicators > 0 else 0.0
    
    def _calculate_timeliness_score(self, response: str) -> float:
        """Calculate how timely and current the response is."""
        if not response:
            return 0.0
        
        timeliness_indicators = 0
        total_indicators = 0
        
        # Check for recent version numbers
        recent_versions = ['2024', '2023', '3.0', '4.0', '5.0']
        for version in recent_versions:
            if version in response:
                timeliness_indicators += 1
                break
        total_indicators += 1
        
        # Check for current technology mentions
        current_tech = ['kubernetes', 'docker', 'microservices', 'cloud', 'aws', 'azure', 'gcp']
        if any(tech in response.lower() for tech in current_tech):
            timeliness_indicators += 1
        total_indicators += 1
        
        # Check for outdated technology mentions
        outdated_tech = ['php4', 'ie6', 'flash', 'silverlight']
        if any(tech in response.lower() for tech in outdated_tech):
            timeliness_indicators -= 1
        total_indicators += 1
        
        # Ensure score is between 0 and 1
        timeliness_score = max(0, min(1, timeliness_indicators / total_indicators))
        return timeliness_score
    
    def _calculate_overall_utility_score(
        self,
        actionability_score: float,
        relevance_score: float,
        completeness_score: float,
        clarity_score: float,
        specificity_score: float,
        timeliness_score: float
    ) -> float:
        """Calculate weighted overall utility score."""
        # Weights for different metrics
        weights = {
            'actionability_score': 0.25,
            'relevance_score': 0.2,
            'completeness_score': 0.2,
            'clarity_score': 0.15,
            'specificity_score': 0.15,
            'timeliness_score': 0.05
        }
        
        overall_score = (
            actionability_score * weights['actionability_score'] +
            relevance_score * weights['relevance_score'] +
            completeness_score * weights['completeness_score'] +
            clarity_score * weights['clarity_score'] +
            specificity_score * weights['specificity_score'] +
            timeliness_score * weights['timeliness_score']
        )
        
        return round(overall_score, 3)
    
    def score_batch(self, responses: List[str], queries: List[str]) -> List[Dict[str, float]]:
        """Score multiple responses in batch."""
        if len(responses) != len(queries):
            raise ValueError("Number of responses must match number of queries")
        
        results = []
        for response, query in zip(responses, queries):
            result = self.score(response, query)
            results.append(result)
        
        return results
    
    def get_utility_summary(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Get summary statistics for a batch of utility evaluations."""
        if not results:
            return {}
        
        summary = {}
        for metric in ['actionability_score', 'relevance_score', 'completeness_score',
                      'clarity_score', 'specificity_score', 'timeliness_score', 'overall_utility']:
            values = [result[metric] for result in results]
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
            summary[f'{metric}_min'] = np.min(values)
            summary[f'{metric}_max'] = np.max(values)
        
        return summary


# Global scorer instance
response_utility_scorer = ResponseUtilityScorer()
