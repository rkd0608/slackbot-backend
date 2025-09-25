"""Base model configuration and all database models."""

from datetime import datetime
from typing import Optional, List, Any
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON, Float, TypeDecorator, UniqueConstraint, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declared_attr

from ..core.database import Base


class Vector(TypeDecorator):
    """Custom type for pgvector that properly handles vector data."""
    
    impl = String
    cache_ok = True
    
    def __init__(self, dimensions: Optional[int] = None):
        super().__init__()
        self.dimensions = dimensions
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            # Use the actual vector type from pgvector
            from sqlalchemy.dialects.postgresql import ARRAY
            from sqlalchemy import Float
            return ARRAY(Float)
        return self.impl
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == 'postgresql':
            if isinstance(value, list):
                # Convert list to proper vector format for pgvector
                return value  # Let SQLAlchemy handle the array conversion
            return value
        return value
    
    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value


class BaseModel(Base):
    """Base model with common fields and methods."""
    
    __abstract__ = True
    
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        return cls.__name__.lower()
    
    # Common fields for all models
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now(), 
        nullable=False
    )
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"<{self.__class__.__name__}(id={self.id})>"


class Workspace(BaseModel):
    """Workspace model representing a Slack workspace."""
    __tablename__ = "workspace"
    __table_args__ = {"schema": "public"}
    
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slack_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    tokens: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    
    # Relationships
    users: Mapped[List["User"]] = relationship("User", back_populates="workspace", cascade="all, delete-orphan")
    conversations: Mapped[List["Conversation"]] = relationship("Conversation", back_populates="workspace", cascade="all, delete-orphan")
    knowledge_items: Mapped[List["KnowledgeItem"]] = relationship("KnowledgeItem", back_populates="workspace", cascade="all, delete-orphan")
    queries: Mapped[List["Query"]] = relationship("Query", back_populates="workspace", cascade="all, delete-orphan")
    feedback: Mapped[List["QueryFeedback"]] = relationship("QueryFeedback", back_populates="workspace")
    interactions: Mapped[List["InteractionEvent"]] = relationship("InteractionEvent", back_populates="workspace")
    knowledge_quality: Mapped[List["KnowledgeQuality"]] = relationship("KnowledgeQuality", back_populates="workspace")


class User(BaseModel):
    """User model representing a Slack user within a workspace."""
    __tablename__ = "user"
    __table_args__ = {"schema": "public"}
    
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.workspace.id"), nullable=False, index=True)
    slack_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="user")
    
    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="users")
    queries: Mapped[List["Query"]] = relationship("Query", back_populates="user", cascade="all, delete-orphan")
    feedback: Mapped[List["QueryFeedback"]] = relationship("QueryFeedback", back_populates="user")
    interactions: Mapped[List["InteractionEvent"]] = relationship("InteractionEvent", back_populates="user")


class Conversation(BaseModel):
    """Conversation model representing logical conversation units."""
    __tablename__ = "conversation"
    __table_args__ = (
        UniqueConstraint("workspace_id", "slack_channel_id", "thread_timestamp", name="uq_conversation_workspace_channel_thread"),
        {"schema": "public"}
    )
    
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.workspace.id"), nullable=False, index=True)
    slack_channel_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    slack_channel_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    thread_timestamp: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)  # For threaded conversations
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    topic: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Main topic/subject
    
    # Conversation state management
    state: Mapped[str] = mapped_column(String(50), nullable=False, default="developing")  # ConversationState enum
    state_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    state_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    
    # Boundary detection metadata
    participant_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    resolution_indicators: Mapped[Optional[List[str]]] = mapped_column(JSONB, nullable=True, default=list)
    
    # Processing flags
    is_ready_for_extraction: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    extraction_attempted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    extraction_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    conversation_metadata: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    
    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    knowledge_items: Mapped[List["KnowledgeItem"]] = relationship("KnowledgeItem", back_populates="conversation")


class Message(BaseModel):
    """Message model representing Slack messages."""
    __tablename__ = "message"
    __table_args__ = (
        UniqueConstraint("conversation_id", "slack_message_id", name="uq_message_conversation_slack_id"),
        {"schema": "public"}
    )
    
    conversation_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.conversation.id"), nullable=False, index=True)
    slack_message_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    slack_user_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")


class KnowledgeItem(BaseModel):
    """Knowledge item model for extracted information."""
    __tablename__ = "knowledgeitem"
    __table_args__ = {"schema": "public"}
    
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.workspace.id"), nullable=False, index=True)
    conversation_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("public.conversation.id"), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    knowledge_type: Mapped[str] = mapped_column(String(100), nullable=False, default='general')
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    source_messages: Mapped[Optional[List[int]]] = mapped_column(JSONB, nullable=True, default=list)
    participants: Mapped[Optional[List[str]]] = mapped_column(JSONB, nullable=True, default=list)
    # Use proper vector type for pgvector - store as text for now
    embedding: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    item_metadata: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True, default=dict)
    
    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="knowledge_items")
    conversation: Mapped[Optional["Conversation"]] = relationship("Conversation", back_populates="knowledge_items")


class Query(BaseModel):
    """Query model for user questions and AI responses."""
    __tablename__ = "query"
    __table_args__ = {"schema": "public"}
    
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.workspace.id"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.user.id"), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[dict] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="queries")
    user: Mapped["User"] = relationship("User", back_populates="queries")
    feedback: Mapped[List["QueryFeedback"]] = relationship("QueryFeedback", back_populates="query")


class QueryFeedback(BaseModel):
    """Feedback model for user responses to queries."""
    __tablename__ = "query_feedback"
    __table_args__ = {"schema": "public"}
    
    query_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.query.id"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.user.id"), nullable=False, index=True)
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.workspace.id"), nullable=False, index=True)
    
    # Feedback data
    feedback_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'thumbs_up', 'thumbs_down', 'detailed', 'report_issue'
    rating: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1-5 scale
    is_helpful: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    
    # Detailed feedback
    feedback_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    issue_category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # 'inaccurate', 'incomplete', 'irrelevant'
    suggested_correction: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Metadata
    interaction_metadata: Mapped[dict] = mapped_column(JSONB, nullable=True, default=dict)
    
    # Relationships
    query: Mapped["Query"] = relationship("Query", back_populates="feedback")
    user: Mapped["User"] = relationship("User", back_populates="feedback")
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="feedback")


class InteractionEvent(BaseModel):
    """Model for tracking all user interactions with the bot."""
    __tablename__ = "interaction_event"
    __table_args__ = {"schema": "public"}
    
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.user.id"), nullable=False, index=True)
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.workspace.id"), nullable=False, index=True)
    query_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("public.query.id"), nullable=True, index=True)
    
    # Interaction details
    interaction_type: Mapped[str] = mapped_column(String(100), nullable=False)  # 'button_click', 'modal_submit', 'view_sources', etc.
    component_type: Mapped[str] = mapped_column(String(100), nullable=False)  # 'button', 'select', 'modal', etc.
    action_id: Mapped[str] = mapped_column(String(200), nullable=False)
    
    # Slack context
    slack_user_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    slack_channel_id: Mapped[str] = mapped_column(String(50), nullable=False)
    slack_message_ts: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    slack_trigger_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    
    # Event data
    payload: Mapped[dict] = mapped_column(JSONB, nullable=True, default=dict)
    response_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="interactions")
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="interactions")
    query: Mapped[Optional["Query"]] = relationship("Query")


class KnowledgeQuality(BaseModel):
    """Model for tracking knowledge item quality based on user feedback."""
    __tablename__ = "knowledge_quality"
    __table_args__ = {"schema": "public"}
    
    knowledge_item_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.knowledgeitem.id"), nullable=False, index=True)
    workspace_id: Mapped[int] = mapped_column(Integer, ForeignKey("public.workspace.id"), nullable=False, index=True)
    
    # Quality metrics
    positive_feedback_count: Mapped[int] = mapped_column(Integer, default=0)
    negative_feedback_count: Mapped[int] = mapped_column(Integer, default=0)
    total_usage_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Calculated scores
    quality_score: Mapped[float] = mapped_column(Float, default=0.5)  # 0.0 - 1.0
    confidence_adjustment: Mapped[float] = mapped_column(Float, default=0.0)  # -0.5 to +0.5
    
    # Flags
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    is_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timestamps
    last_feedback_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_quality_update: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    # Relationships
    knowledge_item: Mapped["KnowledgeItem"] = relationship("KnowledgeItem")
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="knowledge_quality")
