"""Hallucination detection for AI responses."""

import re
from typing import Dict, Any, List, Set, Optional, Tuple
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger


class HallucinationDetector:
    """Detects hallucinations and fabricated information in AI responses."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the detector with a sentence transformer model.

        In TEST_MODE, skip loading heavy models and use keyword fallbacks.
        """
        self.sentence_transformer = None
        if os.getenv("TEST_MODE", "").lower() == "true":
            logger.info("TEST_MODE enabled: skipping sentence transformer load in HallucinationDetector")
            return
        try:
            self.sentence_transformer = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_transformer = None
    
    def detect(self, response: str, source_material: List[str]) -> Dict[str, Any]:
        """Detect hallucinations in the response based on source material."""
        try:
            # Clean and normalize texts
            response_clean = self._clean_text(response)
            source_clean = [self._clean_text(source) for source in source_material]
            
            # Perform different types of hallucination detection
            source_verification = self._verify_source_attribution(response_clean, source_clean)
            contradiction_analysis = self._analyze_contradictions(response_clean, source_clean)
            fabrication_patterns = self._detect_fabrication_patterns(response_clean)
            confidence_calibration = self._calibrate_confidence(response_clean, source_clean)
            
            # Calculate overall hallucination score
            overall_score = self._calculate_overall_hallucination_score(
                source_verification, contradiction_analysis, fabrication_patterns, confidence_calibration
            )
            
            # Determine if response contains hallucinations
            has_hallucinations = overall_score > 0.5
            
            return {
                "source_verification": source_verification,
                "contradiction_analysis": contradiction_analysis,
                "fabrication_patterns": fabrication_patterns,
                "confidence_calibration": confidence_calibration,
                "overall_hallucination_score": overall_score,
                "has_hallucinations": has_hallucinations,
                "hallucination_details": self._get_hallucination_details(
                    response_clean, source_clean, source_verification, contradiction_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Error in hallucination detection: {e}")
            return {
                "source_verification": {"score": 0.0, "details": []},
                "contradiction_analysis": {"score": 0.0, "details": []},
                "fabrication_patterns": {"score": 0.0, "details": []},
                "confidence_calibration": {"score": 0.0, "details": []},
                "overall_hallucination_score": 0.0,
                "has_hallucinations": False,
                "hallucination_details": []
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
    
    def _verify_source_attribution(self, response: str, source_material: List[str]) -> Dict[str, Any]:
        """Verify that claims in the response can be traced to source material."""
        if not response or not source_material:
            return {"score": 0.0, "details": []}
        
        # Extract claims from response
        claims = self._extract_claims(response)
        
        verified_claims = 0
        unverified_claims = []
        
        for claim in claims:
            if self._claim_is_verified(claim, source_material):
                verified_claims += 1
            else:
                unverified_claims.append(claim)
        
        verification_score = verified_claims / len(claims) if claims else 1.0
        
        return {
            "score": verification_score,
            "total_claims": len(claims),
            "verified_claims": verified_claims,
            "unverified_claims": unverified_claims
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        claims = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for sentence in sentences:
            # Skip questions and commands
            if sentence.startswith(('what', 'how', 'when', 'where', 'why', 'who', 'which')):
                continue
            if sentence.startswith(('run', 'execute', 'type', 'enter')):
                continue
            
            # Extract factual statements
            if self._is_factual_statement(sentence):
                claims.append(sentence)
        
        return claims
    
    def _is_factual_statement(self, sentence: str) -> bool:
        """Check if a sentence is a factual statement."""
        # Skip very short sentences
        if len(sentence.split()) < 3:
            return False
        
        # Skip questions
        if sentence.endswith('?'):
            return False
        
        # Skip commands
        command_indicators = ['run', 'execute', 'type', 'enter', 'click', 'select']
        if any(indicator in sentence for indicator in command_indicators):
            return False
        
        # Look for factual indicators
        factual_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'will', 'can', 'should', 'must']
        if any(indicator in sentence for indicator in factual_indicators):
            return True
        
        # Look for specific information
        specific_patterns = [
            r'\d+',  # Numbers
            r'\b[A-Z][a-z]+\b',  # Proper nouns
            r'\.\w+',  # File extensions
            r'https?://',  # URLs
        ]
        
        if any(re.search(pattern, sentence) for pattern in specific_patterns):
            return True
        
        return False
    
    def _claim_is_verified(self, claim: str, source_material: List[str]) -> bool:
        """Check if a claim can be verified against source material."""
        if not self.sentence_transformer:
            # Fallback to keyword matching
            return self._claim_is_verified_keywords(claim, source_material)
        
        try:
            # Use semantic similarity
            claim_embedding = self.sentence_transformer.encode([claim])
            source_embeddings = self.sentence_transformer.encode(source_material)
            
            similarities = cosine_similarity(claim_embedding, source_embeddings)[0]
            max_similarity = np.max(similarities)
            
            return max_similarity > 0.7  # Threshold for verification
            
        except Exception as e:
            logger.error(f"Error in semantic verification: {e}")
            return self._claim_is_verified_keywords(claim, source_material)
    
    def _claim_is_verified_keywords(self, claim: str, source_material: List[str]) -> bool:
        """Fallback keyword-based verification."""
        claim_words = set(claim.split())
        
        for source in source_material:
            source_words = set(source.split())
            overlap = len(claim_words.intersection(source_words))
            overlap_ratio = overlap / len(claim_words) if claim_words else 0
            
            if overlap_ratio > 0.6:  # 60% word overlap
                return True
        
        return False
    
    def _analyze_contradictions(self, response: str, source_material: List[str]) -> Dict[str, Any]:
        """Analyze contradictions between response and source material."""
        if not response or not source_material:
            return {"score": 0.0, "details": []}
        
        contradictions = []
        
        # Extract key facts from response
        response_facts = self._extract_key_facts(response)
        
        # Check each fact against source material
        for fact in response_facts:
            contradiction = self._find_contradiction(fact, source_material)
            if contradiction:
                contradictions.append({
                    "fact": fact,
                    "contradiction": contradiction
                })
        
        # If no facts were extracted, but we found alternative-pair contradictions, score as 1.0
        contradiction_score = len(contradictions) / len(response_facts) if response_facts else (1.0 if contradictions else 0.0)
        
        return {
            "score": contradiction_score,
            "total_facts": len(response_facts),
            "contradictions": contradictions
        }
    
    def _extract_key_facts(self, text: str) -> List[str]:
        """Extract key facts from text."""
        facts = []
        
        # Look for specific patterns that indicate facts
        fact_patterns = [
            r'\d+\.\d+',  # Version numbers
            r'\d{4}-\d{2}-\d{2}',  # Dates
            r'https?://[^\s]+',  # URLs
            r'/[^\s]+',  # File paths
            r'\b[A-Z][a-z]+\s+\d+',  # Names with numbers
        ]
        
        for pattern in fact_patterns:
            matches = re.findall(pattern, text)
            facts.extend(matches)
        
        return facts
    
    def _find_contradiction(self, fact: str, source_material: List[str]) -> Optional[str]:
        """Find if a fact contradicts source material."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated NLP techniques
        
        for source in source_material:
            if fact in source:
                continue  # Fact is supported
            
            # Check for direct contradictions
            if self._is_direct_contradiction(fact, source):
                return source
        
        # Additional heuristic: contradictory alternatives within the same domain
        # e.g., "rolling updates" vs "blue-green deployment"
        alt_pairs = [
            ("rolling updates", "blue-green"),
            ("enable", "disable"),
            ("on", "off"),
        ]
        fact_lower = fact.lower()
        for a, b in alt_pairs:
            if a in fact_lower:
                for source in source_material:
                    if b in source.lower():
                        return source
            if b in fact_lower:
                for source in source_material:
                    if a in source.lower():
                        return source
        
        return None
    
    def _is_direct_contradiction(self, fact: str, source: str) -> bool:
        """Check if fact directly contradicts source."""
        # Simple contradiction detection
        # This could be enhanced with more sophisticated NLP
        
        # Check for opposite numbers
        if re.search(r'\d+', fact) and re.search(r'\d+', source):
            fact_numbers = re.findall(r'\d+', fact)
            source_numbers = re.findall(r'\d+', source)
            
            if fact_numbers and source_numbers:
                if fact_numbers[0] != source_numbers[0]:
                    return True
        
        # Check for opposite statements
        opposite_pairs = [
            ('true', 'false'), ('yes', 'no'), ('enabled', 'disabled'),
            ('on', 'off'), ('up', 'down'), ('start', 'stop')
        ]
        
        for positive, negative in opposite_pairs:
            if positive in fact and negative in source:
                return True
            if negative in fact and positive in source:
                return True
        
        return False
    
    def _detect_fabrication_patterns(self, response: str) -> Dict[str, Any]:
        """Detect common fabrication patterns."""
        if not response:
            return {"score": 0.0, "details": []}
        
        fabrication_indicators = 0
        total_indicators = 0
        details = []
        
        # Check for overly specific details without context
        specific_patterns = [
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # Specific timestamps
            r'version\s+\d+\.\d+\.\d+',  # Specific version numbers
            r'commit\s+[a-f0-9]{40}',  # Specific commit hashes
        ]
        
        for pattern in specific_patterns:
            if re.search(pattern, response):
                fabrication_indicators += 1
                details.append(f"Overly specific detail: {pattern}")
            total_indicators += 1
        
        # Check for vague language that might indicate uncertainty
        vague_patterns = [
            r'\b(probably|maybe|might|could|possibly)\b',
            r'\b(i think|i believe|i assume)\b',
            r'\b(not sure|unclear|unknown)\b',
        ]
        
        for pattern in vague_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                fabrication_indicators += 1
                details.append(f"Vague language: {pattern}")
            total_indicators += 1
        
        # Check for made-up technical terms
        made_up_terms = [
            r'\b(hyperconfig|megadatabase|ultraserver)\b',
            r'\b(superapi|megafile|ultraprocess)\b',
        ]
        
        for pattern in made_up_terms:
            if re.search(pattern, response, re.IGNORECASE):
                fabrication_indicators += 1
                details.append(f"Made-up term: {pattern}")
            total_indicators += 1
        
        fabrication_score = fabrication_indicators / total_indicators if total_indicators > 0 else 0.0
        
        return {
            "score": fabrication_score,
            "indicators": fabrication_indicators,
            "total_indicators": total_indicators,
            "details": details
        }
    
    def _calibrate_confidence(self, response: str, source_material: List[str]) -> Dict[str, Any]:
        """Calibrate confidence based on source material support."""
        if not response or not source_material:
            return {"score": 0.0, "details": []}
        
        # Extract confidence indicators from response
        confidence_indicators = self._extract_confidence_indicators(response)
        
        # Check if confidence is justified by source material
        justified_confidence = 0
        total_confidence = 0
        
        for indicator in confidence_indicators:
            total_confidence += 1
            if self._confidence_is_justified(indicator, source_material):
                justified_confidence += 1
        
        calibration_score = justified_confidence / total_confidence if total_confidence > 0 else 1.0
        
        return {
            "score": calibration_score,
            "justified_confidence": justified_confidence,
            "total_confidence": total_confidence
        }
    
    def _extract_confidence_indicators(self, text: str) -> List[str]:
        """Extract confidence indicators from text."""
        indicators = []
        
        # High confidence indicators
        high_confidence = [
            r'\b(definitely|certainly|absolutely|always|never)\b',
            r'\b(100%|always|never|all|none)\b',
            r'\b(proven|confirmed|verified|established)\b',
        ]
        
        for pattern in high_confidence:
            matches = re.findall(pattern, text, re.IGNORECASE)
            indicators.extend(matches)
        
        return indicators
    
    def _confidence_is_justified(self, indicator: str, source_material: List[str]) -> bool:
        """Check if confidence indicator is justified by source material."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated analysis
        
        for source in source_material:
            if indicator.lower() in source.lower():
                return True
        
        return False
    
    def _calculate_overall_hallucination_score(
        self,
        source_verification: Dict[str, Any],
        contradiction_analysis: Dict[str, Any],
        fabrication_patterns: Dict[str, Any],
        confidence_calibration: Dict[str, Any]
    ) -> float:
        """Calculate overall hallucination score."""
        # Weights for different metrics
        weights = {
            'source_verification': 0.4,
            'contradiction_analysis': 0.3,
            'fabrication_patterns': 0.2,
            'confidence_calibration': 0.1
        }
        
        overall_score = (
            (1.0 - source_verification['score']) * weights['source_verification'] +
            contradiction_analysis['score'] * weights['contradiction_analysis'] +
            fabrication_patterns['score'] * weights['fabrication_patterns'] +
            (1.0 - confidence_calibration['score']) * weights['confidence_calibration']
        )
        
        return round(overall_score, 3)
    
    def _get_hallucination_details(
        self,
        response: str,
        source_material: List[str],
        source_verification: Dict[str, Any],
        contradiction_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get detailed information about detected hallucinations."""
        details = []
        
        # Add unverified claims
        for claim in source_verification.get('unverified_claims', []):
            details.append({
                'type': 'unverified_claim',
                'content': claim,
                'severity': 'medium'
            })
        
        # Add contradictions
        for contradiction in contradiction_analysis.get('contradictions', []):
            details.append({
                'type': 'contradiction',
                'content': contradiction['fact'],
                'contradiction': contradiction['contradiction'],
                'severity': 'high'
            })
        
        return details
    
    def detect_batch(self, responses: List[str], source_materials: List[List[str]]) -> List[Dict[str, Any]]:
        """Detect hallucinations in multiple responses."""
        if len(responses) != len(source_materials):
            raise ValueError("Number of responses must match number of source material lists")
        
        results = []
        for response, source_material in zip(responses, source_materials):
            result = self.detect(response, source_material)
            results.append(result)
        
        return results


# Global detector instance
hallucination_detector = HallucinationDetector()
