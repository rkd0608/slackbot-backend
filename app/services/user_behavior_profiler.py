"""User behavior profiler for understanding individual communication patterns."""

import asyncio
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from dataclasses import dataclass
from loguru import logger

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc

from ..models.base import Message, Conversation, User, Query
from ..models.intent_classification import UserCommunicationProfile, IntentClassificationHistory


@dataclass
class UserProfile:
    """User communication profile with learned patterns."""
    user_id: str
    formality_level: float
    verbosity_preference: float
    emoji_usage_frequency: float
    question_asking_frequency: float
    common_greetings: List[str]
    common_phrases: List[str]
    preferred_response_style: str
    communication_culture: str
    timezone_preference: Optional[str]
    language_preference: str
    learning_confidence: float
    interaction_count: int
    last_updated: datetime


class UserBehaviorProfiler:
    """Profiles user communication patterns for personalized responses."""
    
    def __init__(self):
        self.logger = logger
        self.emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U0001F018-\U0001F0F5\U0001F200-\U0001F2FF]')
        self.question_pattern = re.compile(r'\?+')
        self.greeting_patterns = [
            r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
            r'\b(what\'s up|how are you|how\'s it going)\b',
            r'\b(yo|sup|wassup)\b'
        ]
    
    async def get_or_create_user_profile(
        self,
        db: AsyncSession,
        user_id: str,
        workspace_id: int
    ) -> UserProfile:
        """Get or create user communication profile."""
        try:
            # First, get the database user ID from the Slack user ID
            from ..models.base import User
            user_result = await db.execute(
                select(User).where(
                    and_(
                        User.slack_id == user_id,
                        User.workspace_id == workspace_id
                    )
                )
            )
            user = user_result.scalar_one_or_none()
            
            if not user:
                # User doesn't exist, return default profile
                return UserProfile(
                    user_id=user_id,
                    formality_level=0.5,
                    verbosity_preference=0.5,
                    emoji_usage_frequency=0.0,
                    question_asking_frequency=0.0,
                    common_greetings=[],
                    common_phrases=[],
                    preferred_response_style="balanced",
                    communication_culture="general",
                    timezone_preference=None,
                    language_preference="en",
                    learning_confidence=0.0,
                    interaction_count=0
                )
            
            # Try to get existing profile using the database user ID
            result = await db.execute(
                select(UserCommunicationProfile)
                .where(
                    and_(
                        UserCommunicationProfile.user_id == user.id,  # Use integer user ID
                        UserCommunicationProfile.workspace_id == workspace_id
                    )
                )
            )
            profile = result.scalar_one_or_none()
            
            if profile:
                return self._convert_to_user_profile(profile)
            
            # Create new profile
            new_profile = UserCommunicationProfile(
                user_id=user.id,  # Use integer user ID
                workspace_id=workspace_id,
                formality_level=0.5,
                verbosity_preference=0.5,
                emoji_usage_frequency=0.0,
                question_asking_frequency=0.0,
                common_greetings=[],
                common_phrases=[],
                preferred_response_style="balanced",
                communication_culture="general",
                timezone_preference=None,
                language_preference="en",
                learning_confidence=0.0,
                interaction_count=0
            )
            
            db.add(new_profile)
            await db.commit()
            
            return self._convert_to_user_profile(new_profile)
            
        except Exception as e:
            self.logger.error(f"Error getting/creating user profile: {e}")
            # Return default profile
            return UserProfile(
                user_id=user_id,
                formality_level=0.5,
                verbosity_preference=0.5,
                emoji_usage_frequency=0.0,
                question_asking_frequency=0.0,
                common_greetings=[],
                common_phrases=[],
                preferred_response_style="balanced",
                communication_culture="general",
                timezone_preference=None,
                language_preference="en",
                learning_confidence=0.0,
                interaction_count=0,
                last_updated=datetime.utcnow()
            )
    
    async def update_user_profile(
        self,
        db: AsyncSession,
        user_id: str,
        workspace_id: int,
        message_text: str,
        message_metadata: Optional[Dict[str, Any]] = None
    ) -> UserProfile:
        """Update user profile based on new message."""
        try:
            # Get current profile
            profile = await self.get_or_create_user_profile(db, user_id, workspace_id)
            
            # Analyze the new message
            message_analysis = await self._analyze_message(message_text, message_metadata)
            
            # Update profile based on analysis
            updated_profile = await self._update_profile_with_analysis(
                db, profile, message_analysis, user_id, workspace_id
            )
            
            return updated_profile
            
        except Exception as e:
            self.logger.error(f"Error updating user profile: {e}")
            return await self.get_or_create_user_profile(db, user_id, workspace_id)
    
    async def _analyze_message(
        self,
        message_text: str,
        message_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze a single message for communication patterns."""
        try:
            analysis = {
                "formality_indicators": 0,
                "verbosity_score": 0,
                "emoji_count": 0,
                "question_count": 0,
                "greeting_detected": False,
                "common_phrases": [],
                "language_indicators": {},
                "timezone_hints": []
            }
            
            text_lower = message_text.lower()
            text_length = len(message_text)
            word_count = len(message_text.split())
            
            # Analyze formality
            formal_indicators = [
                "please", "thank you", "would you", "could you", "may I",
                "sir", "madam", "regards", "sincerely", "respectfully"
            ]
            casual_indicators = [
                "yo", "hey", "sup", "wassup", "lol", "haha", "omg",
                "btw", "fyi", "tbh", "imo", "ttyl", "brb"
            ]
            
            formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
            casual_count = sum(1 for indicator in casual_indicators if indicator in text_lower)
            
            analysis["formality_indicators"] = (formal_count - casual_count) / max(word_count, 1)
            
            # Analyze verbosity
            analysis["verbosity_score"] = word_count / 50.0  # Normalize to 0-1 scale
            
            # Count emojis
            emoji_matches = self.emoji_pattern.findall(message_text)
            analysis["emoji_count"] = len(emoji_matches)
            
            # Count questions
            question_matches = self.question_pattern.findall(message_text)
            analysis["question_count"] = len(question_matches)
            
            # Detect greetings
            for pattern in self.greeting_patterns:
                if re.search(pattern, text_lower):
                    analysis["greeting_detected"] = True
                    break
            
            # Extract common phrases (simple n-gram analysis)
            words = text_lower.split()
            if len(words) >= 2:
                bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
                analysis["common_phrases"] = bigrams[:5]  # Top 5 bigrams
            
            # Language detection hints
            if any(word in text_lower for word in ["the", "and", "or", "but", "in", "on", "at"]):
                analysis["language_indicators"]["english"] = 1.0
            
            # Timezone hints from message metadata
            if message_metadata and "timestamp" in message_metadata:
                try:
                    timestamp = datetime.fromisoformat(message_metadata["timestamp"].replace("Z", "+00:00"))
                    hour = timestamp.hour
                    if 0 <= hour < 6:
                        analysis["timezone_hints"].append("late_night")
                    elif 6 <= hour < 12:
                        analysis["timezone_hints"].append("morning")
                    elif 12 <= hour < 18:
                        analysis["timezone_hints"].append("afternoon")
                    else:
                        analysis["timezone_hints"].append("evening")
                except:
                    pass
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing message: {e}")
            return {}
    
    async def _update_profile_with_analysis(
        self,
        db: AsyncSession,
        current_profile: UserProfile,
        message_analysis: Dict[str, Any],
        user_id: str,
        workspace_id: int
    ) -> UserProfile:
        """Update profile based on message analysis."""
        try:
            # Get the database profile
            result = await db.execute(
                select(UserCommunicationProfile)
                .where(
                    and_(
                        UserCommunicationProfile.user_id == user_id,
                        UserCommunicationProfile.workspace_id == workspace_id
                    )
                )
            )
            db_profile = result.scalar_one_or_none()
            
            if not db_profile:
                return current_profile
            
            # Update formality level (exponential moving average)
            formality_delta = message_analysis.get("formality_indicators", 0)
            alpha = 0.1  # Learning rate
            db_profile.formality_level = (1 - alpha) * db_profile.formality_level + alpha * (0.5 + formality_delta)
            db_profile.formality_level = max(0.0, min(1.0, db_profile.formality_level))
            
            # Update verbosity preference
            verbosity_score = message_analysis.get("verbosity_score", 0.5)
            db_profile.verbosity_preference = (1 - alpha) * db_profile.verbosity_preference + alpha * verbosity_score
            db_profile.verbosity_preference = max(0.0, min(1.0, db_profile.verbosity_preference))
            
            # Update emoji usage frequency
            emoji_count = message_analysis.get("emoji_count", 0)
            if emoji_count > 0:
                db_profile.emoji_usage_frequency = min(1.0, db_profile.emoji_usage_frequency + 0.1)
            else:
                db_profile.emoji_usage_frequency = max(0.0, db_profile.emoji_usage_frequency - 0.05)
            
            # Update question asking frequency
            question_count = message_analysis.get("question_count", 0)
            if question_count > 0:
                db_profile.question_asking_frequency = min(1.0, db_profile.question_asking_frequency + 0.1)
            else:
                db_profile.question_asking_frequency = max(0.0, db_profile.question_asking_frequency - 0.02)
            
            # Update common greetings
            if message_analysis.get("greeting_detected", False):
                # Extract greeting from message
                greeting = self._extract_greeting(message_analysis.get("common_phrases", []))
                if greeting and greeting not in (db_profile.common_greetings or []):
                    current_greetings = list(db_profile.common_greetings or [])
                    current_greetings.append(greeting)
                    db_profile.common_greetings = current_greetings[-10:]  # Keep last 10
            
            # Update common phrases
            new_phrases = message_analysis.get("common_phrases", [])
            if new_phrases:
                current_phrases = list(db_profile.common_phrases or [])
                current_phrases.extend(new_phrases)
                # Keep most frequent phrases
                phrase_counter = Counter(current_phrases)
                db_profile.common_phrases = [phrase for phrase, count in phrase_counter.most_common(20)]
            
            # Update preferred response style based on user behavior
            if db_profile.formality_level > 0.7:
                db_profile.preferred_response_style = "formal"
            elif db_profile.formality_level < 0.3:
                db_profile.preferred_response_style = "casual"
            else:
                db_profile.preferred_response_style = "balanced"
            
            # Update communication culture
            if message_analysis.get("language_indicators", {}).get("english", 0) > 0.5:
                db_profile.communication_culture = "english"
            
            # Update timezone preference
            timezone_hints = message_analysis.get("timezone_hints", [])
            if timezone_hints:
                # Simple timezone detection based on message timing
                if "morning" in timezone_hints:
                    db_profile.timezone_preference = "morning_person"
                elif "evening" in timezone_hints:
                    db_profile.timezone_preference = "evening_person"
            
            # Update learning confidence
            db_profile.interaction_count += 1
            if db_profile.interaction_count > 10:
                db_profile.learning_confidence = min(1.0, db_profile.interaction_count / 100.0)
            
            # Update last updated timestamp
            db_profile.last_updated = datetime.utcnow()
            
            await db.commit()
            
            return self._convert_to_user_profile(db_profile)
            
        except Exception as e:
            self.logger.error(f"Error updating profile with analysis: {e}")
            return current_profile
    
    def _extract_greeting(self, phrases: List[str]) -> Optional[str]:
        """Extract greeting from common phrases."""
        greeting_keywords = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "yo", "sup"]
        
        for phrase in phrases:
            for keyword in greeting_keywords:
                if keyword in phrase.lower():
                    return phrase
        
        return None
    
    def _convert_to_user_profile(self, db_profile: UserCommunicationProfile) -> UserProfile:
        """Convert database profile to UserProfile dataclass."""
        return UserProfile(
            user_id=db_profile.user_id,
            formality_level=db_profile.formality_level,
            verbosity_preference=db_profile.verbosity_preference,
            emoji_usage_frequency=db_profile.emoji_usage_frequency,
            question_asking_frequency=db_profile.question_asking_frequency,
            common_greetings=db_profile.common_greetings or [],
            common_phrases=db_profile.common_phrases or [],
            preferred_response_style=db_profile.preferred_response_style,
            communication_culture=db_profile.communication_culture,
            timezone_preference=db_profile.timezone_preference,
            language_preference=db_profile.language_preference,
            learning_confidence=db_profile.learning_confidence,
            interaction_count=db_profile.interaction_count,
            last_updated=db_profile.last_updated
        )
    
    async def get_user_communication_style(
        self,
        db: AsyncSession,
        user_id: str,
        workspace_id: int
    ) -> Dict[str, Any]:
        """Get user's communication style for response generation."""
        try:
            profile = await self.get_or_create_user_profile(db, user_id, workspace_id)
            
            return {
                "formality_level": profile.formality_level,
                "verbosity_preference": profile.verbosity_preference,
                "emoji_usage_frequency": profile.emoji_usage_frequency,
                "preferred_response_style": profile.preferred_response_style,
                "common_greetings": profile.common_greetings,
                "communication_culture": profile.communication_culture,
                "learning_confidence": profile.learning_confidence,
                "interaction_count": profile.interaction_count
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user communication style: {e}")
            return {
                "formality_level": 0.5,
                "verbosity_preference": 0.5,
                "emoji_usage_frequency": 0.0,
                "preferred_response_style": "balanced",
                "common_greetings": [],
                "communication_culture": "general",
                "learning_confidence": 0.0,
                "interaction_count": 0
            }
    
    async def analyze_user_patterns(
        self,
        db: AsyncSession,
        user_id: str,
        workspace_id: int,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Analyze user communication patterns over time."""
        try:
            # Get user's recent messages
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            result = await db.execute(
                select(Message)
                .join(Conversation)
                .where(
                    and_(
                        Message.slack_user_id == user_id,
                        Conversation.workspace_id == workspace_id,
                        Message.created_at >= cutoff_date
                    )
                )
                .order_by(Message.created_at.desc())
                .limit(100)
            )
            messages = result.scalars().all()
            
            if not messages:
                return {"message_count": 0, "patterns": {}}
            
            # Analyze patterns
            patterns = {
                "message_count": len(messages),
                "avg_message_length": sum(len(msg.content) for msg in messages) / len(messages),
                "emoji_usage_rate": sum(1 for msg in messages if self.emoji_pattern.search(msg.content)) / len(messages),
                "question_rate": sum(1 for msg in messages if "?" in msg.content) / len(messages),
                "greeting_rate": sum(1 for msg in messages if any(re.search(pattern, msg.content.lower()) for pattern in self.greeting_patterns)) / len(messages),
                "most_active_hours": self._analyze_active_hours(messages),
                "most_active_days": self._analyze_active_days(messages),
                "common_words": self._extract_common_words(messages),
                "communication_evolution": self._analyze_communication_evolution(messages)
            }
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing user patterns: {e}")
            return {"message_count": 0, "patterns": {}}
    
    def _analyze_active_hours(self, messages: List[Message]) -> List[int]:
        """Analyze most active hours of the day."""
        hour_counts = Counter(msg.created_at.hour for msg in messages)
        return [hour for hour, count in hour_counts.most_common(3)]
    
    def _analyze_active_days(self, messages: List[Message]) -> List[str]:
        """Analyze most active days of the week."""
        day_counts = Counter(msg.created_at.strftime("%A") for msg in messages)
        return [day for day, count in day_counts.most_common(3)]
    
    def _extract_common_words(self, messages: List[Message]) -> List[str]:
        """Extract most common words from user's messages."""
        all_words = []
        for msg in messages:
            words = msg.content.lower().split()
            # Filter out common stop words
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            all_words.extend(filtered_words)
        
        word_counts = Counter(all_words)
        return [word for word, count in word_counts.most_common(10)]
    
    def _analyze_communication_evolution(self, messages: List[Message]) -> Dict[str, Any]:
        """Analyze how user's communication style has evolved over time."""
        if len(messages) < 10:
            return {"trend": "insufficient_data"}
        
        # Split messages into early and recent periods
        mid_point = len(messages) // 2
        early_messages = messages[mid_point:]
        recent_messages = messages[:mid_point]
        
        # Compare formality, verbosity, etc.
        early_formality = self._calculate_formality_score(early_messages)
        recent_formality = self._calculate_formality_score(recent_messages)
        
        early_verbosity = sum(len(msg.content.split()) for msg in early_messages) / len(early_messages)
        recent_verbosity = sum(len(msg.content.split()) for msg in recent_messages) / len(recent_messages)
        
        return {
            "formality_trend": "increasing" if recent_formality > early_formality else "decreasing",
            "verbosity_trend": "increasing" if recent_verbosity > early_verbosity else "decreasing",
            "formality_change": recent_formality - early_formality,
            "verbosity_change": recent_verbosity - early_verbosity
        }
    
    def _calculate_formality_score(self, messages: List[Message]) -> float:
        """Calculate formality score for a set of messages."""
        if not messages:
            return 0.5
        
        formal_indicators = ["please", "thank you", "would you", "could you", "may I"]
        casual_indicators = ["yo", "hey", "sup", "lol", "haha", "omg", "btw", "fyi"]
        
        total_words = sum(len(msg.content.split()) for msg in messages)
        if total_words == 0:
            return 0.5
        
        formal_count = sum(sum(1 for word in msg.content.lower().split() if word in formal_indicators) for msg in messages)
        casual_count = sum(sum(1 for word in msg.content.lower().split() if word in casual_indicators) for msg in messages)
        
        return (formal_count - casual_count) / total_words + 0.5
