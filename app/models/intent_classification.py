"""Enhanced intent classification models for real-world conversation understanding."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON, Float, Boolean, ARRAY
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func

from .base import BaseModel


class UserCommunicationProfile(BaseModel):
    """User-specific communication patterns and preferences."""
    __tablename__ = "user_communication_profile"
    __table_args__ = {"schema": "public"}
    
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.user.id"), nullable=False, index=True, unique=True)
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.workspace.id"), nullable=False, index=True)
    
    # Communication style analysis
    formality_level: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)  # 0.0 = very casual, 1.0 = very formal
    verbosity_preference: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)  # 0.0 = concise, 1.0 = detailed
    emoji_usage_frequency: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # 0.0 = never, 1.0 = always
    question_asking_frequency: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # 0.0 = never, 1.0 = always
    
    # Common communication patterns
    common_greetings: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=True, default=list)
    common_phrases: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=True, default=list)
    preferred_response_style: Mapped[str] = mapped_column(String(50), nullable=True, default="balanced")
    
    # Cultural and regional context
    communication_culture: Mapped[str] = mapped_column(String(100), nullable=True, default="general")
    timezone_preference: Mapped[str] = mapped_column(String(50), nullable=True)
    language_preference: Mapped[str] = mapped_column(String(10), nullable=True, default="en")
    
    # Learning metadata
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    learning_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # How confident we are in this profile
    interaction_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Relationships
    user: Mapped["User"] = relationship("User")
    workspace: Mapped["Workspace"] = relationship("Workspace")


class ChannelCulture(BaseModel):
    """Channel-specific communication norms and culture."""
    __tablename__ = "channel_culture"
    __table_args__ = {"schema": "public"}
    
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.workspace.id"), nullable=False, index=True)
    channel_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    channel_name: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Channel communication style
    formality_level: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    topic_focus: Mapped[str] = mapped_column(String(100), nullable=True)  # "general", "technical", "social", etc.
    response_expectations: Mapped[str] = mapped_column(String(50), nullable=True, default="balanced")  # "quick", "detailed", "balanced"
    
    # Common patterns in this channel
    common_topics: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=True, default=list)
    common_phrases: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=True, default=list)
    active_participants: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=True, default=list)
    
    # Channel-specific metadata
    channel_purpose: Mapped[str] = mapped_column(String(255), nullable=True)
    communication_norms: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True, default=dict)
    
    # Learning metadata
    last_analyzed: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    analysis_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    message_count_analyzed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace")


class ConversationContext(BaseModel):
    """Current conversation state and context."""
    __tablename__ = "conversation_context"
    __table_args__ = {"schema": "public"}
    
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.workspace.id"), nullable=False, index=True)
    channel_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    thread_ts: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    
    # Current conversation state
    active_topic: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    conversation_stage: Mapped[str] = mapped_column(String(50), nullable=False, default="starting")  # "starting", "developing", "concluding"
    participant_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Context metadata
    last_bot_response: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_user_message: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    conversation_sentiment: Mapped[float] = mapped_column(Float, nullable=True)  # -1.0 to 1.0
    
    # Contextual information
    recent_messages: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, nullable=True, default=list)
    key_participants: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=True, default=list)
    conversation_metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True, default=dict)
    
    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace")


class IntentClassificationHistory(BaseModel):
    """History of intent classifications for learning and analysis."""
    __tablename__ = "intent_classification_history"
    __table_args__ = {"schema": "public"}
    
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.workspace.id"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.user.id"), nullable=False, index=True)
    channel_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    query_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("public.query.id"), nullable=True, index=True)
    
    # Classification details
    original_message: Mapped[str] = mapped_column(Text, nullable=False)
    classified_intent: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    classification_method: Mapped[str] = mapped_column(String(50), nullable=False)  # "rule_based", "contextual", "ai_powered"
    
    # Context at time of classification
    conversation_context: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True, default=dict)
    user_context: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True, default=dict)
    channel_context: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True, default=dict)
    
    # Outcome tracking
    response_generated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    user_satisfaction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 1.0 to 5.0
    response_effectiveness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0.0 to 1.0
    follow_up_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    # Learning metadata
    was_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)  # Was this classification correct?
    correction_applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    learning_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace")
    user: Mapped["User"] = relationship("User")
    query: Mapped[Optional["Query"]] = relationship("Query")


class ResponseEffectiveness(BaseModel):
    """Track effectiveness of different response strategies."""
    __tablename__ = "response_effectiveness"
    __table_args__ = {"schema": "public"}
    
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.workspace.id"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.user.id"), nullable=False, index=True)
    query_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.query.id"), nullable=False, index=True)
    
    # Response details
    response_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "social", "knowledge", "clarification", etc.
    response_style: Mapped[str] = mapped_column(String(50), nullable=False)  # "formal", "casual", "detailed", "concise"
    response_length: Mapped[int] = mapped_column(Integer, nullable=False)  # Character count
    
    # Effectiveness metrics
    user_rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 1.0 to 5.0
    response_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Seconds to respond
    follow_up_questions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    resolution_achieved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    # Context factors
    intent_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    conversation_stage: Mapped[str] = mapped_column(String(50), nullable=False)
    time_of_day: Mapped[str] = mapped_column(String(20), nullable=True)  # "morning", "afternoon", "evening"
    
    # Learning data
    was_helpful: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    improvement_suggestions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace")
    user: Mapped["User"] = relationship("User")
    query: Mapped["Query"] = relationship("Query")
