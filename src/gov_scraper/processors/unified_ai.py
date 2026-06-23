"""Unified AI processor for Israeli Government Decisions - Single consolidated AI call."""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from google import genai
from google.genai import types

from ..config import GEMINI_API_KEY, GEMINI_MODEL, MAX_RETRIES, RETRY_DELAY
from .ai_prompts import (
    UNIFIED_PROCESSING_PROMPT,
    OPERATIVITY_EXAMPLES,
    POLICY_TAG_EXAMPLES,
    validate_confidence_scores
)
from .ai_validator import AIResponseValidator
from .alignment_validator import create_alignment_validator

# Set up logging
logger = logging.getLogger(__name__)

# Try to import committee mappings if available
try:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    from config.committee_mappings import normalize_committee_name
except ImportError:
    logger.warning("Committee mappings not found, using direct mapping")
    def normalize_committee_name(name):
        return name


@dataclass
class AIProcessingResult:
    """Structured result from unified AI processing."""
    summary: str
    operativity: str
    policy_areas: List[str]
    government_bodies: List[str]
    locations: List[str]
    special_categories: List[str]

    # New alignment fields
    core_theme: str  # The main theme/essence of the decision
    alignment_check: str  # AI's self-assessment of alignment
    alignment_score: float  # Quantitative alignment measure

    # Confidence scores (0.0-1.0)
    summary_confidence: float
    operativity_confidence: float
    tags_confidence: float
    alignment_confidence: float  # New alignment confidence

    # Evidence tracking
    summary_evidence: str  # Quote from source
    operativity_evidence: str
    tags_evidence: List[str]  # List of supporting quotes

    # Processing metadata
    processing_time: float
    api_calls_used: int
    fallback_used: bool = False


class UnifiedAIProcessor:
    """
    Unified AI processor that extracts all decision fields in a single API call.

    Features:
    - Single consolidated prompt for all extractions
    - Structured JSON output with confidence scores
    - Evidence tracking with source quotes
    - Smart fallback to individual calls if needed
    - Performance optimizations (caching, batching)
    """

    def __init__(self, policy_areas: List[str], government_bodies: List[str]):
        """Initialize with authorized tag lists."""
        self.policy_areas = policy_areas
        self.government_bodies = government_bodies
        self.validator = AIResponseValidator(policy_areas, government_bodies)
        self.alignment_validator = create_alignment_validator(policy_areas)
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.response_cache = {}  # Simple cache for retries

    def _get_smart_content(self, content: str, max_length: int = 4000) -> str:
        """
        Extract content intelligently for long decisions.
        Takes 70% from beginning and 30% from end for better context.
        """
        if len(content) <= max_length:
            return content

        head_size = int(max_length * 0.7)
        tail_size = max_length - head_size

        return f"{content[:head_size]}\n\n[...תוכן מקוצץ...]\n\n{content[-tail_size:]}"

    def _make_unified_request(self, prompt: str, max_tokens: int = 1500) -> str:
        """Make unified API request with retry logic and caching."""
        cache_key = hash(prompt)
        if cache_key in self.response_cache:
            logger.info("Using cached response for unified request")
            return self.response_cache[cache_key]

        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Making unified Gemini request (attempt {attempt + 1}/{MAX_RETRIES})")

                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction="אתה מנתח מקצועי של החלטות ממשלה ישראליות. החזר תמיד JSON מובנה ומדויק.",
                        max_output_tokens=max_tokens,
                        temperature=0.1,  # Low temperature for consistency
                        thinking_config=types.ThinkingConfig(thinking_budget=0),  # Disable thinking — saves token budget for JSON output
                    ),
                )

                if not response.text:
                    raise Exception("Gemini returned empty response")

                result = response.text.strip()
                self.response_cache[cache_key] = result
                logger.info(f"Unified request successful (attempt {attempt + 1})")
                return result

            except Exception as e:
                error_str = str(e)
                logger.warning(f"Unified request failed (attempt {attempt + 1}): {e}")

                # Hard-quota detection: "limit: 0" or PerDayPerProject quota means daily
                # quota is exhausted/disabled and will NOT recover within minutes. Bail
                # immediately instead of burning ~15min in pointless backoff per decision.
                # User must wait for daily reset OR upgrade Gemini tier.
                hard_quota = (
                    "limit: 0" in error_str
                    or "PerDayPerProject" in error_str and "exceeded" in error_str.lower()
                )
                if hard_quota:
                    raise Exception(
                        f"Gemini DAILY quota exhausted (limit:0 or PerDay exceeded). "
                        f"Will not retry — daily reset required. Detail: {e}"
                    )

                # Check for transient rate limit error (429 minute/burst quota)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # Exponential backoff for rate limits: 30s, 60s, 120s, 240s, 480s
                    wait_time = min(30 * (2 ** attempt), 480)  # Cap at 8 minutes
                    logger.warning(f"Rate limit hit! Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                elif attempt < MAX_RETRIES - 1:
                    # Regular linear backoff for other errors
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise Exception(f"Unified AI request failed after {MAX_RETRIES} attempts: {e}")

    def _parse_unified_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from unified prompt."""
        try:
            # Clean up common AI response patterns
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            response = response.strip()

            parsed = json.loads(response)

            # Validate required fields
            required_fields = [
                'summary', 'operativity', 'policy_areas', 'government_bodies',
                'locations', 'special_categories', 'confidence_scores', 'evidence'
            ]
            # New optional fields for enhanced alignment
            optional_fields = ['core_theme', 'alignment_check']

            for field in required_fields:
                if field not in parsed:
                    raise ValueError(f"Missing required field: {field}")

            # Fix truncated summaries - ensure they end properly
            if 'summary' in parsed and parsed['summary']:
                summary = parsed['summary'].strip()
                # Check if summary ends mid-word or without proper punctuation
                if summary and not summary[-1] in '.!?׃:':
                    # If it looks truncated, add ellipsis
                    if len(summary) > 50 and not summary.endswith('...'):
                        # Check if last word might be incomplete (no space before it within last 10 chars)
                        last_space = summary.rfind(' ')
                        if last_space > len(summary) - 15:
                            # Remove potential incomplete word
                            summary = summary[:last_space].rstrip()
                        # Add proper ending
                        summary = summary + '...'
                        logger.debug(f"Fixed truncated summary: {summary[-50:]}")
                    parsed['summary'] = summary

            # Normalize committee names in government bodies
            if 'government_bodies' in parsed and parsed['government_bodies']:
                normalized_bodies = []
                for body in parsed['government_bodies']:
                    normalized = normalize_committee_name(body)
                    if normalized != body:
                        logger.debug(f"Normalized committee: '{body}' -> '{normalized}'")
                    normalized_bodies.append(normalized)
                parsed['government_bodies'] = normalized_bodies

            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse unified JSON response: {e}")
            logger.error(f"Response was: {response[:500]}")
            raise ValueError(f"Invalid JSON response from AI: {e}")

    def _extract_confidence_scores(self, parsed: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """Extract and validate confidence scores including new alignment confidence."""
        confidence = parsed.get('confidence_scores', {})
        if not isinstance(confidence, dict):
            confidence = {}

        def _safe_float(val, default=0.5):
            """Coerce value to float, handling dicts/lists/strings from AI."""
            if isinstance(val, (int, float)):
                return max(0.0, min(1.0, float(val)))
            if isinstance(val, str):
                try:
                    return max(0.0, min(1.0, float(val)))
                except ValueError:
                    return default
            return default

        summary_conf = _safe_float(confidence.get('summary', 0.5))
        operativity_conf = _safe_float(confidence.get('operativity', 0.5))
        tags_conf = _safe_float(confidence.get('tags', 0.5))
        alignment_conf = _safe_float(confidence.get('alignment', 0.5))

        return summary_conf, operativity_conf, tags_conf, alignment_conf

    def _calculate_alignment_score(self, parsed: Dict[str, Any], alignment_check: str) -> float:
        """Calculate quantitative alignment score based on AI assessment and validation."""
        # Base score from AI assessment
        if "כן" in alignment_check:
            base_score = 0.8
        elif "חלקי" in alignment_check or "ברוב" in alignment_check:
            base_score = 0.6
        elif "לא" in alignment_check:
            base_score = 0.3
        else:
            base_score = 0.5  # Unknown/unclear

        # Cross-validate by checking semantic overlap
        try:
            summary_words = set(parsed.get('summary', '').lower().split())
            tag_words = set()

            # Extract words from policy tags
            for tag in parsed.get('policy_areas', []):
                tag_words.update(tag.lower().split())

            # Calculate word overlap (simplified semantic alignment)
            if summary_words and tag_words:
                overlap = len(summary_words.intersection(tag_words))
                total = len(summary_words.union(tag_words))
                semantic_overlap = overlap / total if total > 0 else 0

                # Adjust base score based on semantic overlap
                if semantic_overlap > 0.2:  # Good overlap
                    base_score = min(1.0, base_score + 0.1)
                elif semantic_overlap < 0.1:  # Poor overlap
                    base_score = max(0.0, base_score - 0.2)

        except Exception as e:
            logger.debug(f"Could not calculate semantic overlap: {e}")

        return base_score

    def _create_processing_result(self, parsed: Dict[str, Any], processing_time: float) -> AIProcessingResult:
        """Create structured result from parsed response."""

        # Extract confidence scores including alignment
        summary_conf, operativity_conf, tags_conf, alignment_conf = self._extract_confidence_scores(parsed)

        # Extract evidence
        evidence = parsed.get('evidence', {})

        # Extract alignment fields (optional)
        core_theme = parsed.get('core_theme', '')
        alignment_check = parsed.get('alignment_check', '')

        # Calculate alignment score based on AI assessment
        alignment_score = self._calculate_alignment_score(parsed, alignment_check)

        # Import validation function from ai.py
        from .ai import validate_operativity_classification

        # Apply operativity validation
        validated_operativity = validate_operativity_classification(
            parsed['operativity'],
            self.decision_content,
            self.decision_title
        )

        return AIProcessingResult(
            summary=parsed['summary'],
            operativity=validated_operativity,
            policy_areas=parsed['policy_areas'],
            government_bodies=parsed['government_bodies'],
            locations=parsed['locations'],
            special_categories=parsed['special_categories'],
            core_theme=core_theme,
            alignment_check=alignment_check,
            alignment_score=alignment_score,
            summary_confidence=summary_conf,
            operativity_confidence=operativity_conf,
            tags_confidence=tags_conf,
            alignment_confidence=alignment_conf,
            summary_evidence=evidence.get('summary_quote', ''),
            operativity_evidence=evidence.get('operativity_quote', ''),
            tags_evidence=evidence.get('tags_quotes', []),
            processing_time=processing_time,
            api_calls_used=1,
            fallback_used=False
        )

    def process_decision_unified(
        self,
        decision_content: str,
        decision_title: str,
        decision_date: str = None
    ) -> AIProcessingResult:
        # Store for validation use
        self.decision_content = decision_content
        self.decision_title = decision_title
        """
        Process decision with unified AI call.

        Args:
            decision_content: Full decision text
            decision_title: Decision title
            decision_date: Optional decision date for context

        Returns:
            AIProcessingResult with all extracted fields and metadata
        """
        start_time = time.time()

        logger.info("Starting unified AI processing")

        try:
            # Send full content — Gemini 2.0 Flash supports 1M tokens,
            # max decision is ~32K chars (~15K tokens). No truncation needed.
            smart_content = decision_content

            # Calculate dynamic summary parameters based on content length
            from .ai import calculate_dynamic_summary_params
            summary_instructions, max_tokens_for_summary = calculate_dynamic_summary_params(len(decision_content))

            # Build unified prompt with dynamic summary instructions
            prompt = UNIFIED_PROCESSING_PROMPT.format(
                policy_areas=" | ".join(self.policy_areas),
                government_bodies=" | ".join(self.government_bodies),
                operativity_examples=OPERATIVITY_EXAMPLES,
                policy_examples=POLICY_TAG_EXAMPLES,
                decision_title=decision_title,
                decision_content=smart_content,
                decision_date=f"תאריך: {decision_date}" if decision_date else "",
                summary_instructions=summary_instructions
            )

            # Calculate total max tokens (summary + other fields)
            # Other fields need ~500 tokens, so add that to summary tokens
            total_max_tokens = max_tokens_for_summary + 500

            logger.debug(f"Content length: {len(decision_content)} -> summary: {summary_instructions}, tokens: {total_max_tokens}")

            # Make unified API call with dynamic token limit
            response = self._make_unified_request(prompt, max_tokens=total_max_tokens)

            # Parse response
            parsed = self._parse_unified_response(response)

            # Create result
            processing_time = time.time() - start_time
            result = self._create_processing_result(parsed, processing_time)

            # Enhanced validation with tag-content relevance checking
            validation_result = self.validator.validate_unified_result(
                result, decision_content, decision_title
            )

            # Additional policy tag validation using detection profiles
            tag_validation = self.validator.validate_policy_tags_with_profiles(
                result.policy_areas, decision_content, decision_title
            )

            # NEW: Summary-Tag Alignment Validation
            alignment_validation = self.alignment_validator.validate_alignment(
                result.summary, result.policy_areas, decision_title, decision_content
            )

            if not validation_result.is_valid:
                logger.warning(f"General validation failed: {validation_result.errors}")

            # Auto-correct policy tags when profile validation rejects some
            if tag_validation.errors:
                validated_tags, rejected_tags = self.validator._validate_tag_content_relevance(
                    result.policy_areas, decision_content, decision_title
                )
                if validated_tags:
                    logger.info(f"Auto-correcting policy tags: keeping {validated_tags}, rejected {rejected_tags}")
                    result.policy_areas = validated_tags
                else:
                    # All tags rejected — keep original (post-processor whitelist will handle)
                    logger.warning(f"All tags rejected by profile validation, keeping original: {result.policy_areas}")

            # Log alignment info (warn-only, no auto-correction — concept map
            # only covers 30% of tags, so auto-correction destroys good tags)
            if not alignment_validation.is_aligned:
                logger.debug(f"Alignment score: {alignment_validation.alignment_score:.2f}, issues: {alignment_validation.issues}")

            logger.info(f"Unified processing completed in {processing_time:.2f}s (alignment score: {result.alignment_score:.2f})")
            return result

        except Exception as e:
            logger.error(f"Unified processing failed: {e}")
            # Fallback to individual calls
            return self._fallback_to_individual_calls(
                decision_content, decision_title, decision_date, start_time
            )

    def _fallback_to_individual_calls(
        self,
        decision_content: str,
        decision_title: str,
        decision_date: str,
        start_time: float
    ) -> AIProcessingResult:
        """Fallback to individual AI calls if unified call fails."""
        logger.info("Falling back to individual AI calls")

        try:
            # Import individual functions from existing ai.py
            from .ai import (
                generate_summary,
                generate_operativity,
                generate_policy_area_tags_strict,
                generate_government_body_tags_validated,
                generate_location_tags,
                generate_special_category_tags
            )

            # Make individual calls (5-6 API calls)
            summary = generate_summary(decision_content, decision_title)
            operativity = generate_operativity(decision_content)
            # Apply operativity validation
            from .ai import validate_operativity_classification
            operativity = validate_operativity_classification(operativity, decision_content, decision_title)
            policy_areas = generate_policy_area_tags_strict(decision_content, decision_title, summary)
            government_bodies = generate_government_body_tags_validated(decision_content, decision_title, summary)
            locations = generate_location_tags(decision_content, decision_title)
            special_categories = generate_special_category_tags(decision_content, decision_title, summary, decision_date)

            processing_time = time.time() - start_time

            # Convert to structured format
            return AIProcessingResult(
                summary=summary,
                operativity=operativity,
                policy_areas=policy_areas.split(';') if policy_areas else [],
                government_bodies=government_bodies.split(';') if government_bodies else [],
                locations=locations.split(',') if locations else [],
                special_categories=special_categories,
                core_theme="",  # Not available in fallback
                alignment_check="Fallback processing - no alignment check",
                alignment_score=0.5,  # Default for fallback
                summary_confidence=0.7,  # Default confidence for fallback
                operativity_confidence=0.7,
                tags_confidence=0.7,
                alignment_confidence=0.5,  # Default for fallback
                summary_evidence="",  # No evidence tracking for fallback
                operativity_evidence="",
                tags_evidence=[],
                processing_time=processing_time,
                api_calls_used=6,  # Individual calls
                fallback_used=True
            )

        except Exception as e:
            logger.error(f"Fallback processing also failed: {e}")
            raise Exception(f"Both unified and fallback processing failed: {e}")

    def process_decision_batch(self, decisions: List[Dict[str, str]]) -> List[AIProcessingResult]:
        """
        Process multiple decisions efficiently.

        Note: Current implementation processes sequentially.
        Future versions could implement true batching.
        """
        results = []

        for i, decision in enumerate(decisions):
            logger.info(f"Processing decision {i+1}/{len(decisions)}")

            try:
                result = self.process_decision_unified(
                    decision['decision_content'],
                    decision['decision_title'],
                    decision.get('decision_date')
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process decision {i+1}: {e}")
                # Could add empty result or skip
                continue

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics."""
        return {
            "cache_size": len(self.response_cache),
            "cache_hit_rate": "N/A",  # Would need hit tracking
            "average_processing_time": "N/A",  # Would need history
        }


def create_unified_processor(policy_areas: List[str], government_bodies: List[str]) -> UnifiedAIProcessor:
    """Factory function to create unified processor."""
    return UnifiedAIProcessor(policy_areas, government_bodies)