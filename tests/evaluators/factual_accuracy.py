"""Factual accuracy evaluation for AI responses."""

import re
import difflib
from typing import Dict, Any, List, Tuple, Optional
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger


class FactualAccuracyEvaluator:
    """Evaluates factual accuracy of AI responses using multiple metrics."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the evaluator with a sentence transformer model.

        In TEST_MODE, skip loading heavy models and rely on keyword-based fallbacks.
        """
        self.sentence_transformer = None
        if os.getenv("TEST_MODE", "").lower() == "true":
            logger.info("TEST_MODE enabled: skipping sentence transformer load in FactualAccuracyEvaluator")
            return
        try:
            self.sentence_transformer = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_transformer = None
    
    def evaluate(self, response: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate factual accuracy using multiple metrics."""
        try:
            # Clean and normalize texts
            response_clean = self._clean_text(response)
            ground_truth_clean = self._clean_text(ground_truth)
            
            # Calculate different accuracy metrics
            exact_match = self._exact_match_score(response_clean, ground_truth_clean)
            # Use model similarity when available; otherwise, keyword-based similarity
            if self.sentence_transformer is None:
                semantic_similarity = self._keyword_similarity(response_clean, ground_truth_clean)
            else:
                semantic_similarity = self._semantic_similarity_score(response_clean, ground_truth_clean)
            named_entity_accuracy = self._named_entity_accuracy(response_clean, ground_truth_clean)
            partial_match = self._partial_match_score(response_clean, ground_truth_clean)
            
            # Calculate overall score (weighted average)
            overall_score = self._calculate_overall_score(
                exact_match, semantic_similarity, named_entity_accuracy, partial_match
            )
            
            return {
                "exact_match": exact_match,
                "semantic_similarity": semantic_similarity,
                "named_entity_accuracy": named_entity_accuracy,
                "partial_match": partial_match,
                "overall_score": overall_score
            }
            
        except Exception as e:
            logger.error(f"Error in factual accuracy evaluation: {e}")
            return {
                "exact_match": 0.0,
                "semantic_similarity": 0.0,
                "named_entity_accuracy": 0.0,
                "partial_match": 0.0,
                "overall_score": 0.0
            }

    def _keyword_similarity(self, response: str, ground_truth: str) -> float:
        """Approximate semantic similarity via Jaccard similarity of keyword sets."""
        if not response or not ground_truth:
            return 0.0
        # Remove very common stop words to focus on content terms
        stop = {
            'the','a','an','and','or','but','if','then','else','for','on','in','at','to','of','with','by','is','are','was','were','be','been','do','does','did','that','this','it','as','from','we','you','they','our','your'
        }
        resp_words = {w for w in response.split() if w not in stop}
        gt_words = {w for w in ground_truth.split() if w not in stop}
        if not resp_words or not gt_words:
            return 0.0
        inter = len(resp_words & gt_words)
        union = len(resp_words | gt_words)
        return inter / union
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for comparison."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation for exact matching
        text = re.sub(r'[^\w\s]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _exact_match_score(self, response: str, ground_truth: str) -> float:
        """Calculate exact match score (0 or 1)."""
        if not response or not ground_truth:
            return 0.0
        
        return 1.0 if response == ground_truth else 0.0
    
    def _semantic_similarity_score(self, response: str, ground_truth: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        if not self.sentence_transformer or not response or not ground_truth:
            return 0.0
        
        try:
            # Generate embeddings
            response_embedding = self.sentence_transformer.encode([response])
            ground_truth_embedding = self.sentence_transformer.encode([ground_truth])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(response_embedding, ground_truth_embedding)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _named_entity_accuracy(self, response: str, ground_truth: str) -> float:
        """Calculate accuracy of named entities (dates, names, numbers, commands)."""
        if not response or not ground_truth:
            return 0.0
        
        # Extract named entities using regex patterns
        response_entities = self._extract_entities(response)
        ground_truth_entities = self._extract_entities(ground_truth)
        
        if not ground_truth_entities:
            return 1.0  # No entities to match
        
        # Calculate precision and recall
        matches = 0
        for entity in ground_truth_entities:
            if any(self._entities_match(entity, resp_entity) for resp_entity in response_entities):
                matches += 1
        
        precision = matches / len(response_entities) if response_entities else 0.0
        recall = matches / len(ground_truth_entities)
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        entities = []
        
        # Extract dates
        date_pattern = r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b|\b\w+ \d{1,2}, \d{4}\b'
        for match in re.finditer(date_pattern, text):
            entities.append({
                'type': 'date',
                'value': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract commands (docker, git, etc.)
        command_pattern = r'\b(docker|git|npm|pip|python|node|kubectl|aws|gcloud)\s+\w+'
        for match in re.finditer(command_pattern, text):
            entities.append({
                'type': 'command',
                'value': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract version numbers
        version_pattern = r'\b\d+\.\d+(\.\d+)?\b'
        for match in re.finditer(version_pattern, text):
            entities.append({
                'type': 'version',
                'value': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract URLs
        url_pattern = r'https?://[^\s]+'
        for match in re.finditer(url_pattern, text):
            entities.append({
                'type': 'url',
                'value': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract file paths
        path_pattern = r'/[^\s]+|\./[^\s]+|\.\./[^\s]+'
        for match in re.finditer(path_pattern, text):
            entities.append({
                'type': 'path',
                'value': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        return entities
    
    def _entities_match(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        """Check if two entities match."""
        if entity1['type'] != entity2['type']:
            return False
        
        # For commands, check if the base command matches
        if entity1['type'] == 'command':
            cmd1 = entity1['value'].split()[0]
            cmd2 = entity2['value'].split()[0]
            return cmd1 == cmd2
        
        # For other entities, check exact match or similarity
        if entity1['type'] in ['date', 'version', 'url', 'path']:
            return entity1['value'] == entity2['value']
        
        # For general text, use fuzzy matching
        return difflib.SequenceMatcher(None, entity1['value'], entity2['value']).ratio() > 0.8
    
    def _partial_match_score(self, response: str, ground_truth: str) -> float:
        """Calculate partial match score using sequence matching."""
        if not response or not ground_truth:
            return 0.0
        
        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, response, ground_truth)
        return matcher.ratio()
    
    def _calculate_overall_score(
        self, 
        exact_match: float, 
        semantic_similarity: float, 
        named_entity_accuracy: float, 
        partial_match: float
    ) -> float:
        """Calculate weighted overall score."""
        # Weights for different metrics
        weights = {
            'exact_match': 0.2,
            'semantic_similarity': 0.5,
            'named_entity_accuracy': 0.2,
            'partial_match': 0.1
        }
        
        overall_score = (
            exact_match * weights['exact_match'] +
            semantic_similarity * weights['semantic_similarity'] +
            named_entity_accuracy * weights['named_entity_accuracy'] +
            partial_match * weights['partial_match']
        )
        
        return round(overall_score, 3)
    
    def evaluate_batch(self, responses: List[str], ground_truths: List[str]) -> List[Dict[str, float]]:
        """Evaluate multiple responses in batch."""
        if len(responses) != len(ground_truths):
            raise ValueError("Number of responses must match number of ground truths")
        
        results = []
        for response, ground_truth in zip(responses, ground_truths):
            result = self.evaluate(response, ground_truth)
            results.append(result)
        
        return results
    
    def get_accuracy_summary(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Get summary statistics for a batch of evaluations."""
        if not results:
            return {}
        
        summary = {}
        for metric in ['exact_match', 'semantic_similarity', 'named_entity_accuracy', 'partial_match', 'overall_score']:
            values = [result[metric] for result in results]
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
            summary[f'{metric}_min'] = np.min(values)
            summary[f'{metric}_max'] = np.max(values)
        
        return summary


# Global evaluator instance
factual_accuracy_evaluator = FactualAccuracyEvaluator()
