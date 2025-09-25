"""Completeness evaluation for AI responses."""

import re
from typing import Dict, Any, List, Set, Optional
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger


class CompletenessEvaluator:
    """Evaluates how completely AI responses address user queries."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the evaluator with a sentence transformer model.

        In TEST_MODE, skip loading heavy models and use keyword fallbacks.
        """
        self.sentence_transformer = None
        if os.getenv("TEST_MODE", "").lower() == "true":
            logger.info("TEST_MODE enabled: skipping sentence transformer load in CompletenessEvaluator")
            return
        try:
            self.sentence_transformer = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_transformer = None
    
    def evaluate(self, response: str, query: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate response completeness using multiple metrics."""
        try:
            # Clean and normalize texts
            response_clean = self._clean_text(response)
            query_clean = self._clean_text(query)
            ground_truth_clean = self._clean_text(ground_truth)
            
            # Calculate different completeness metrics
            information_coverage = self._calculate_information_coverage(response_clean, ground_truth_clean)
            query_intent_addressed = self._check_query_intent_addressed(response_clean, query_clean)
            actionability_score = self._calculate_actionability_score(response_clean)
            specificity_score = self._calculate_specificity_score(response_clean)
            completeness_ratio = self._calculate_completeness_ratio(response_clean, ground_truth_clean)
            
            # Calculate overall completeness score
            overall_score = self._calculate_overall_completeness_score(
                information_coverage, query_intent_addressed, actionability_score, 
                specificity_score, completeness_ratio
            )
            
            return {
                "information_coverage": information_coverage,
                "query_intent_addressed": query_intent_addressed,
                "actionability_score": actionability_score,
                "specificity_score": specificity_score,
                "completeness_ratio": completeness_ratio,
                "overall_completeness": overall_score
            }
            
        except Exception as e:
            logger.error(f"Error in completeness evaluation: {e}")
            return {
                "information_coverage": 0.0,
                "query_intent_addressed": 0.0,
                "actionability_score": 0.0,
                "specificity_score": 0.0,
                "completeness_ratio": 0.0,
                "overall_completeness": 0.0
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
    
    def _calculate_information_coverage(self, response: str, ground_truth: str) -> float:
        """Calculate how much of the ground truth information is covered in the response."""
        if not response or not ground_truth:
            return 0.0
        
        # Split into sentences for comparison
        response_sentences = self._split_into_sentences(response)
        ground_truth_sentences = self._split_into_sentences(ground_truth)
        
        if not ground_truth_sentences:
            return 1.0  # No ground truth to cover
        
        # Calculate coverage for each ground truth sentence
        covered_sentences = 0
        for gt_sentence in ground_truth_sentences:
            if self._sentence_is_covered(gt_sentence, response_sentences):
                covered_sentences += 1
        
        coverage_ratio = covered_sentences / len(ground_truth_sentences)
        return coverage_ratio
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _sentence_is_covered(self, gt_sentence: str, response_sentences: List[str]) -> bool:
        """Check if a ground truth sentence is covered in the response."""
        if not self.sentence_transformer:
            # Fallback to simple keyword matching
            gt_words = set(gt_sentence.split())
            for resp_sentence in response_sentences:
                resp_words = set(resp_sentence.split())
                if len(gt_words.intersection(resp_words)) / len(gt_words) > 0.6:
                    return True
            return False
        
        try:
            # Use semantic similarity
            gt_embedding = self.sentence_transformer.encode([gt_sentence])
            resp_embeddings = self.sentence_transformer.encode(response_sentences)
            
            similarities = cosine_similarity(gt_embedding, resp_embeddings)[0]
            max_similarity = np.max(similarities)
            
            return max_similarity > 0.7  # Threshold for coverage
            
        except Exception as e:
            logger.error(f"Error in semantic coverage check: {e}")
            return False
    
    def _check_query_intent_addressed(self, response: str, query: str) -> float:
        """Check if the response addresses the query intent."""
        if not response or not query:
            return 0.0
        
        # Extract query intent keywords
        query_intent = self._extract_query_intent(query)
        
        if not query_intent:
            return 0.5  # Neutral score if no clear intent
        
        # Check if response addresses the intent
        intent_addressed = 0
        for intent_keyword in query_intent:
            if intent_keyword in response:
                intent_addressed += 1
        
        return intent_addressed / len(query_intent) if query_intent else 0.0
    
    def _extract_query_intent(self, query: str) -> List[str]:
        """Extract intent keywords from query."""
        intent_keywords = []
        
        # Question words
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
        for word in question_words:
            if word in query:
                intent_keywords.append(word)
        
        # Action words
        action_words = ['do', 'create', 'build', 'deploy', 'fix', 'solve', 'implement', 'configure']
        for word in action_words:
            if word in query:
                intent_keywords.append(word)
        
        # Technical terms
        tech_terms = ['docker', 'kafka', 'database', 'api', 'server', 'client', 'config', 'migration']
        for term in tech_terms:
            if term in query:
                intent_keywords.append(term)
        
        return intent_keywords
    
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
        ]
        
        for pattern in command_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                action_indicators += 1
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
        
        return action_indicators / total_indicators if total_indicators > 0 else 0.0
    
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
        tech_terms = ['docker', 'kafka', 'postgresql', 'redis', 'api', 'database', 'server', 'client']
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
        
        return specificity_indicators / total_indicators if total_indicators > 0 else 0.0
    
    def _calculate_completeness_ratio(self, response: str, ground_truth: str) -> float:
        """Calculate the ratio of response length to ground truth length."""
        if not response or not ground_truth:
            return 0.0
        
        response_length = len(response.split())
        ground_truth_length = len(ground_truth.split())
        
        if ground_truth_length == 0:
            return 1.0
        
        ratio = response_length / ground_truth_length
        
        # Normalize ratio (too short or too long is not ideal)
        if ratio < 0.5:
            return ratio * 2  # Penalize very short responses
        elif ratio > 2.0:
            return 1.0 - (ratio - 2.0) * 0.1  # Slightly penalize very long responses
        else:
            return 1.0
    
    def _calculate_overall_completeness_score(
        self, 
        information_coverage: float,
        query_intent_addressed: float,
        actionability_score: float,
        specificity_score: float,
        completeness_ratio: float
    ) -> float:
        """Calculate weighted overall completeness score."""
        # Weights for different metrics
        weights = {
            'information_coverage': 0.3,
            'query_intent_addressed': 0.25,
            'actionability_score': 0.2,
            'specificity_score': 0.15,
            'completeness_ratio': 0.1
        }
        
        overall_score = (
            information_coverage * weights['information_coverage'] +
            query_intent_addressed * weights['query_intent_addressed'] +
            actionability_score * weights['actionability_score'] +
            specificity_score * weights['specificity_score'] +
            completeness_ratio * weights['completeness_ratio']
        )
        
        return round(overall_score, 3)
    
    def evaluate_batch(self, responses: List[str], queries: List[str], ground_truths: List[str]) -> List[Dict[str, float]]:
        """Evaluate multiple responses in batch."""
        if len(responses) != len(queries) or len(responses) != len(ground_truths):
            raise ValueError("Number of responses, queries, and ground truths must match")
        
        results = []
        for response, query, ground_truth in zip(responses, queries, ground_truths):
            result = self.evaluate(response, query, ground_truth)
            results.append(result)
        
        return results
    
    def get_completeness_summary(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Get summary statistics for a batch of completeness evaluations."""
        if not results:
            return {}
        
        summary = {}
        for metric in ['information_coverage', 'query_intent_addressed', 'actionability_score', 
                      'specificity_score', 'completeness_ratio', 'overall_completeness']:
            values = [result[metric] for result in results]
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
            summary[f'{metric}_min'] = np.min(values)
            summary[f'{metric}_max'] = np.max(values)
        
        return summary


# Global evaluator instance
completeness_evaluator = CompletenessEvaluator()
