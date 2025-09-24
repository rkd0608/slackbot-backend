"""
Hallucination Prevention Service

This service implements multiple layers of fact-checking to prevent AI hallucinations
in knowledge extraction and response generation. It ensures that only explicitly
stated information is preserved and prevents fabricated details.

Key capabilities:
- Source verification: Ensures all claims can be traced to specific messages
- Fact extraction validation: Prevents inference of unstated information
- Cross-referencing: Validates consistency across multiple sources
- Confidence calibration: Adjusts confidence based on evidence strength
- Fabrication detection: Identifies common hallucination patterns
- Quote verification: Ensures quoted information is accurate
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
from datetime import datetime

from ..services.openai_service import OpenAIService


class HallucinationCheck:
    """Represents a hallucination check result."""
    
    def __init__(
        self,
        check_type: str,
        is_safe: bool,
        confidence: float,
        issues_found: List[str] = None,
        evidence: List[str] = None,
        corrected_content: Optional[str] = None
    ):
        self.check_type = check_type
        self.is_safe = is_safe
        self.confidence = confidence
        self.issues_found = issues_found or []
        self.evidence = evidence or []
        self.corrected_content = corrected_content


class HallucinationPreventer:
    """Prevents AI hallucinations through multi-layered fact-checking."""
    
    def __init__(self):
        self.openai_service = OpenAIService()
        
        # Common hallucination patterns to detect
        self.hallucination_indicators = [
            r"(obviously|clearly|of course|naturally)",  # Assumption indicators
            r"(should|would|might|could|probably|likely)",  # Speculation
            r"(typically|usually|generally|often|commonly)",  # Generalization
            r"(as we know|it's well known|everyone knows)",  # Assumed knowledge
            r"(based on experience|in my experience)",  # Personal inference
            r"(this means|this implies|this suggests)",  # Inference indicators
            r"(therefore|thus|hence|consequently)",  # Causal inference
        ]
        
        # Technical details that are often fabricated
        self.fabrication_patterns = [
            r"\d+\.\d+\.\d+",  # Version numbers (unless explicitly mentioned)
            r"port \d+",       # Port numbers
            r"--\w+",          # Command line flags
            r"http[s]?://",    # URLs (unless explicitly shared)
            r"\$\w+",          # Environment variables
            r"/[a-z/]+",       # File paths (unless explicitly mentioned)
        ]

    async def validate_extracted_knowledge(
        self,
        extracted_content: str,
        source_messages: List[Dict[str, Any]],
        original_conversation: str
    ) -> HallucinationCheck:
        """
        Validate extracted knowledge against source messages to prevent hallucinations.
        
        This is the main validation function that runs multiple checks.
        """
        try:
            logger.info("Running comprehensive hallucination validation")
            
            # Run multiple validation checks
            checks = await self._run_validation_checks(
                extracted_content, source_messages, original_conversation
            )
            
            # Combine check results
            overall_result = self._combine_check_results(checks)
            
            logger.info(f"Hallucination validation result: {overall_result.is_safe} (confidence: {overall_result.confidence:.2f})")
            
            return overall_result
            
        except Exception as e:
            logger.error(f"Error in hallucination validation: {e}")
            return HallucinationCheck(
                check_type="error",
                is_safe=False,
                confidence=0.0,
                issues_found=[f"Validation error: {e}"]
            )

    async def _run_validation_checks(
        self,
        extracted_content: str,
        source_messages: List[Dict[str, Any]],
        original_conversation: str
    ) -> List[HallucinationCheck]:
        """Run multiple validation checks in parallel."""
        checks = []
        
        # 1. Source verification check
        source_check = await self._verify_source_attribution(
            extracted_content, source_messages
        )
        checks.append(source_check)
        
        # 2. Pattern-based fabrication detection
        pattern_check = self._detect_fabrication_patterns(extracted_content)
        checks.append(pattern_check)
        
        # 3. AI-powered fact verification
        ai_check = await self._ai_fact_verification(
            extracted_content, original_conversation
        )
        checks.append(ai_check)
        
        # 4. Quote accuracy verification
        quote_check = await self._verify_quotes_accuracy(
            extracted_content, source_messages
        )
        checks.append(quote_check)
        
        # 5. Technical detail validation
        technical_check = await self._validate_technical_details(
            extracted_content, original_conversation
        )
        checks.append(technical_check)
        
        return checks

    async def _verify_source_attribution(
        self,
        extracted_content: str,
        source_messages: List[Dict[str, Any]]
    ) -> HallucinationCheck:
        """Verify that all claims in extracted content can be attributed to source messages."""
        try:
            # Combine all source message content
            source_text = '\n'.join([
                f"{msg.get('user_id', 'unknown')}: {msg.get('content', '')}"
                for msg in source_messages
            ])
            
            system_prompt = """You are a fact-checker verifying that extracted knowledge only contains information explicitly stated in source messages.

Your job is to identify any claims, details, or statements in the extracted content that are NOT directly supported by the source messages.

Look for:
- Added details not mentioned in sources
- Inferred information not explicitly stated
- Assumptions or generalizations
- Technical details that weren't specified
- Causal relationships not explicitly mentioned
- Fabricated quotes or paraphrases

Return JSON:
{
    "is_safe": true|false,
    "confidence": 0.0-1.0,
    "unsupported_claims": ["claim1", "claim2"],
    "evidence_for_claims": ["evidence1", "evidence2"],
    "corrected_version": "version with only supported claims"
}"""

            response = await self.openai_service._make_request(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Source Messages:\n{source_text}\n\nExtracted Content:\n{extracted_content}"}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            import json
            result = json.loads(response['choices'][0]['message']['content'])
            
            return HallucinationCheck(
                check_type="source_attribution",
                is_safe=result.get('is_safe', False),
                confidence=result.get('confidence', 0.0),
                issues_found=result.get('unsupported_claims', []),
                evidence=result.get('evidence_for_claims', []),
                corrected_content=result.get('corrected_version')
            )
            
        except Exception as e:
            logger.error(f"Source attribution check failed: {e}")
            return HallucinationCheck(
                check_type="source_attribution",
                is_safe=False,
                confidence=0.0,
                issues_found=[f"Check failed: {e}"]
            )

    def _detect_fabrication_patterns(self, extracted_content: str) -> HallucinationCheck:
        """Detect common patterns that indicate potential fabrication."""
        try:
            issues_found = []
            content_lower = extracted_content.lower()
            
            # Check for hallucination indicators
            for pattern in self.hallucination_indicators:
                matches = re.findall(pattern, content_lower)
                if matches:
                    issues_found.append(f"Assumption indicator detected: {matches}")
            
            # Check for fabricated technical details
            for pattern in self.fabrication_patterns:
                matches = re.findall(pattern, extracted_content)
                if matches:
                    issues_found.append(f"Potential fabricated technical detail: {matches}")
            
            # Check for overly specific claims
            specific_patterns = [
                r"\d{4}-\d{2}-\d{2}",  # Specific dates
                r"\d+:\d{2}",          # Specific times
                r"\d+\.\d+ (seconds|minutes|hours)",  # Specific durations
                r"\d+ (GB|MB|KB)",     # Specific sizes
            ]
            
            for pattern in specific_patterns:
                matches = re.findall(pattern, extracted_content)
                if matches:
                    issues_found.append(f"Overly specific claim (verify if stated): {matches}")
            
            is_safe = len(issues_found) == 0
            confidence = 1.0 - (len(issues_found) * 0.2)  # Reduce confidence for each issue
            
            return HallucinationCheck(
                check_type="pattern_detection",
                is_safe=is_safe,
                confidence=max(confidence, 0.0),
                issues_found=issues_found
            )
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return HallucinationCheck(
                check_type="pattern_detection",
                is_safe=False,
                confidence=0.0,
                issues_found=[f"Pattern check failed: {e}"]
            )

    async def _ai_fact_verification(
        self,
        extracted_content: str,
        original_conversation: str
    ) -> HallucinationCheck:
        """Use AI to verify facts against the original conversation."""
        try:
            system_prompt = """You are a meticulous fact-checker. Compare the extracted content against the original conversation to identify any fabricated or inferred information.

Focus on detecting:
1. Information that was IMPLIED but not EXPLICITLY stated
2. Technical details that were assumed rather than mentioned
3. Causal relationships that were inferred
4. Specific values, numbers, or names that weren't provided
5. Steps or procedures that were summarized incorrectly
6. Quotes that don't match the original text

Be extremely strict - if something wasn't explicitly mentioned, it should be flagged.

Return JSON:
{
    "is_accurate": true|false,
    "confidence": 0.0-1.0,
    "fabricated_elements": ["element1", "element2"],
    "severity": "low|medium|high",
    "corrected_content": "content with only verified facts"
}"""

            response = await self.openai_service._make_request(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original Conversation:\n{original_conversation}\n\nExtracted Content:\n{extracted_content}"}
                ],
                temperature=0.0,  # Zero temperature for consistent fact-checking
                max_tokens=800
            )
            
            import json
            result = json.loads(response['choices'][0]['message']['content'])
            
            return HallucinationCheck(
                check_type="ai_fact_verification",
                is_safe=result.get('is_accurate', False),
                confidence=result.get('confidence', 0.0),
                issues_found=result.get('fabricated_elements', []),
                corrected_content=result.get('corrected_content')
            )
            
        except Exception as e:
            logger.error(f"AI fact verification failed: {e}")
            return HallucinationCheck(
                check_type="ai_fact_verification",
                is_safe=False,
                confidence=0.0,
                issues_found=[f"AI verification failed: {e}"]
            )

    async def _verify_quotes_accuracy(
        self,
        extracted_content: str,
        source_messages: List[Dict[str, Any]]
    ) -> HallucinationCheck:
        """Verify that any quoted text matches the original messages exactly."""
        try:
            # Find potential quotes in extracted content
            quote_patterns = [
                r'"([^"]+)"',      # Double quotes
                r"'([^']+)'",      # Single quotes
                r"said: (.+)",     # "User said: ..."
                r"mentioned: (.+)", # "User mentioned: ..."
            ]
            
            potential_quotes = []
            for pattern in quote_patterns:
                matches = re.findall(pattern, extracted_content)
                potential_quotes.extend(matches)
            
            if not potential_quotes:
                return HallucinationCheck(
                    check_type="quote_verification",
                    is_safe=True,
                    confidence=1.0
                )
            
            # Check each quote against source messages
            source_text = ' '.join([msg.get('content', '') for msg in source_messages])
            
            issues_found = []
            for quote in potential_quotes:
                quote_clean = quote.strip().lower()
                if quote_clean not in source_text.lower():
                    issues_found.append(f"Unverified quote: '{quote}'")
            
            is_safe = len(issues_found) == 0
            confidence = 1.0 - (len(issues_found) * 0.3)
            
            return HallucinationCheck(
                check_type="quote_verification",
                is_safe=is_safe,
                confidence=max(confidence, 0.0),
                issues_found=issues_found
            )
            
        except Exception as e:
            logger.error(f"Quote verification failed: {e}")
            return HallucinationCheck(
                check_type="quote_verification",
                is_safe=False,
                confidence=0.0,
                issues_found=[f"Quote verification failed: {e}"]
            )

    async def _validate_technical_details(
        self,
        extracted_content: str,
        original_conversation: str
    ) -> HallucinationCheck:
        """Validate technical details against the conversation to prevent fabrication."""
        try:
            system_prompt = """You are validating technical details in extracted content against the original conversation.

Look for technical details that might have been fabricated or assumed:
- Command syntax and parameters
- Configuration values and settings
- File paths and directory structures
- Port numbers and network settings
- Version numbers and dependencies
- Error messages and codes
- API endpoints and parameters

Flag any technical detail that:
1. Wasn't explicitly mentioned in the conversation
2. Was generalized from specific examples
3. Was assumed based on common practices
4. Contains placeholder or example values

Return JSON:
{
    "technical_accuracy": "accurate|questionable|fabricated",
    "confidence": 0.0-1.0,
    "questionable_details": ["detail1", "detail2"],
    "verified_details": ["detail1", "detail2"],
    "corrected_version": "version with only verified technical details"
}"""

            response = await self.openai_service._make_request(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original Conversation:\n{original_conversation}\n\nExtracted Content:\n{extracted_content}"}
                ],
                temperature=0.0,
                max_tokens=600
            )
            
            import json
            result = json.loads(response['choices'][0]['message']['content'])
            
            accuracy = result.get('technical_accuracy', 'questionable')
            is_safe = accuracy == 'accurate'
            
            return HallucinationCheck(
                check_type="technical_validation",
                is_safe=is_safe,
                confidence=result.get('confidence', 0.0),
                issues_found=result.get('questionable_details', []),
                evidence=result.get('verified_details', []),
                corrected_content=result.get('corrected_version')
            )
            
        except Exception as e:
            logger.error(f"Technical validation failed: {e}")
            return HallucinationCheck(
                check_type="technical_validation",
                is_safe=False,
                confidence=0.0,
                issues_found=[f"Technical validation failed: {e}"]
            )

    def _combine_check_results(self, checks: List[HallucinationCheck]) -> HallucinationCheck:
        """Combine multiple check results into a final assessment."""
        try:
            if not checks:
                return HallucinationCheck(
                    check_type="combined",
                    is_safe=False,
                    confidence=0.0,
                    issues_found=["No checks performed"]
                )
            
            # Calculate overall safety
            safe_checks = sum(1 for check in checks if check.is_safe)
            total_checks = len(checks)
            safety_ratio = safe_checks / total_checks
            
            # Overall is safe only if ALL checks pass
            overall_safe = safety_ratio == 1.0
            
            # Calculate weighted confidence
            total_confidence = sum(check.confidence for check in checks)
            average_confidence = total_confidence / total_checks
            
            # Reduce confidence if any checks failed
            if not overall_safe:
                average_confidence *= safety_ratio
            
            # Combine all issues
            all_issues = []
            all_evidence = []
            corrected_versions = []
            
            for check in checks:
                all_issues.extend([f"[{check.check_type}] {issue}" for issue in check.issues_found])
                all_evidence.extend(check.evidence)
                if check.corrected_content:
                    corrected_versions.append(check.corrected_content)
            
            # Use the most conservative corrected version
            final_corrected = corrected_versions[-1] if corrected_versions else None
            
            return HallucinationCheck(
                check_type="combined",
                is_safe=overall_safe,
                confidence=average_confidence,
                issues_found=all_issues,
                evidence=all_evidence,
                corrected_content=final_corrected
            )
            
        except Exception as e:
            logger.error(f"Error combining check results: {e}")
            return HallucinationCheck(
                check_type="combined",
                is_safe=False,
                confidence=0.0,
                issues_found=[f"Error combining results: {e}"]
            )

    async def validate_ai_response(
        self,
        ai_response: str,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> HallucinationCheck:
        """Validate AI response to prevent hallucinations in user-facing answers."""
        try:
            # Prepare search results context
            context = '\n'.join([
                f"Source: {result.get('title', 'Unknown')}\nContent: {result.get('content', '')}"
                for result in search_results
            ])
            
            system_prompt = """You are validating an AI response to ensure it only contains information supported by the provided search results.

The AI should NOT:
- Add information not present in the search results
- Make assumptions or inferences
- Provide generic advice not specific to the context
- Include fabricated examples or details
- Assume causation without explicit evidence

Flag any response that:
1. Contains unsupported claims
2. Adds details not in the sources
3. Makes assumptions about user's environment
4. Provides generic information not relevant to the query

Return JSON:
{
    "response_accuracy": "accurate|contains_hallucinations|mostly_fabricated",
    "confidence": 0.0-1.0,
    "hallucination_issues": ["issue1", "issue2"],
    "supported_claims": ["claim1", "claim2"],
    "recommended_action": "approve|revise|reject"
}"""

            response = await self.openai_service._make_request(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nSearch Results:\n{context}\n\nAI Response:\n{ai_response}"}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            import json
            result = json.loads(response['choices'][0]['message']['content'])
            
            accuracy = result.get('response_accuracy', 'contains_hallucinations')
            is_safe = accuracy == 'accurate'
            
            return HallucinationCheck(
                check_type="response_validation",
                is_safe=is_safe,
                confidence=result.get('confidence', 0.0),
                issues_found=result.get('hallucination_issues', []),
                evidence=result.get('supported_claims', [])
            )
            
        except Exception as e:
            logger.error(f"AI response validation failed: {e}")
            return HallucinationCheck(
                check_type="response_validation",
                is_safe=False,
                confidence=0.0,
                issues_found=[f"Response validation failed: {e}"]
            )

    def adjust_confidence_for_hallucination_risk(
        self,
        original_confidence: float,
        hallucination_check: HallucinationCheck
    ) -> float:
        """Adjust confidence score based on hallucination risk assessment."""
        try:
            if not hallucination_check.is_safe:
                # Significantly reduce confidence for unsafe content
                risk_penalty = 0.7  # Reduce by up to 70%
                adjusted_confidence = original_confidence * (1 - risk_penalty)
                
                # Further reduce based on number of issues
                issue_count = len(hallucination_check.issues_found)
                issue_penalty = min(issue_count * 0.1, 0.5)  # Max 50% penalty
                adjusted_confidence *= (1 - issue_penalty)
                
                logger.info(f"Confidence reduced from {original_confidence:.2f} to {adjusted_confidence:.2f} due to hallucination risk")
                
                return max(adjusted_confidence, 0.1)  # Minimum 10% confidence
            
            else:
                # Slightly boost confidence for verified safe content
                boost = min(hallucination_check.confidence * 0.1, 0.1)
                return min(original_confidence + boost, 1.0)
                
        except Exception as e:
            logger.error(f"Error adjusting confidence: {e}")
            return original_confidence * 0.5  # Conservative fallback
