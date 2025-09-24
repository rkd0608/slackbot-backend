"""Enhanced security and validation service for query processing."""

import re
import html
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from ..models.base import Workspace, User, Conversation

class SecurityService:
    """Service for enhanced security validation and query sanitization."""
    
    def __init__(self):
        # Malicious patterns to detect
        self.malicious_patterns = [
            r"<script[^>]*>.*?</script>",  # Script tags
            r"javascript:",  # JavaScript protocol
            r"on\w+\s*=",  # Event handlers
            r"data:text/html",  # Data URLs
            r"vbscript:",  # VBScript protocol
            r"<iframe[^>]*>",  # Iframe tags
            r"<object[^>]*>",  # Object tags
            r"<embed[^>]*>",  # Embed tags
            r"<link[^>]*>",  # Link tags
            r"<meta[^>]*>",  # Meta tags
            r"eval\s*\(",  # Eval function
            r"expression\s*\(",  # CSS expression
            r"url\s*\(",  # CSS url function
            r"@import",  # CSS import
            r"\\x[0-9a-fA-F]{2}",  # Hex encoding
            r"\\u[0-9a-fA-F]{4}",  # Unicode encoding
            r"\\[0-7]{1,3}",  # Octal encoding
        ]
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+'.*'\s*=\s*'.*')",
            r"(\b(OR|AND)\s+\".*\"\s*=\s*\".*\")",
            r"(\b(OR|AND)\s+1\s*=\s*1)",
            r"(\b(OR|AND)\s+true)",
            r"(\b(OR|AND)\s+false)",
            r"(--|\#|\/\*|\*\/)",  # SQL comments
            r"(\b(OR|AND)\s+.*\s+LIKE\s+.*)",
            r"(\b(OR|AND)\s+.*\s+IN\s+\(.*\))",
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            r"[;&|`$(){}[\]\\]",  # Command separators
            r"\b(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig)\b",  # Common commands
            r"\b(rm|del|format|fdisk|mkfs)\b",  # Destructive commands
            r"\b(wget|curl|nc|telnet|ssh|ftp)\b",  # Network commands
            r"\b(sudo|su|chmod|chown)\b",  # Privilege escalation
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r"\.\./",  # Directory traversal
            r"\.\.\\",  # Windows directory traversal
            r"%2e%2e%2f",  # URL encoded
            r"%2e%2e%5c",  # URL encoded Windows
            r"\.\.%2f",  # Mixed encoding
            r"\.\.%5c",  # Mixed encoding Windows
        ]
    
    async def validate_query_security(
        self, 
        query: str, 
        user_id: str, 
        workspace_id: int, 
        channel_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Comprehensive security validation for user queries."""
        try:
            validation_results = {
                "is_safe": True,
                "sanitized_query": query,
                "warnings": [],
                "blocked": False,
                "block_reason": None,
                "validation_details": {}
            }
            
            # 1. Basic sanitization
            sanitized_query = self._sanitize_query(query)
            validation_results["sanitized_query"] = sanitized_query
            
            # 2. Malicious content detection
            malicious_check = self._check_malicious_content(sanitized_query)
            validation_results["validation_details"]["malicious_check"] = malicious_check
            
            if malicious_check["is_malicious"]:
                validation_results["is_safe"] = False
                validation_results["blocked"] = True
                validation_results["block_reason"] = "Malicious content detected"
                return validation_results
            
            # 3. SQL injection detection
            sql_check = self._check_sql_injection(sanitized_query)
            validation_results["validation_details"]["sql_check"] = sql_check
            
            if sql_check["is_sql_injection"]:
                validation_results["is_safe"] = False
                validation_results["blocked"] = True
                validation_results["block_reason"] = "SQL injection attempt detected"
                return validation_results
            
            # 4. Command injection detection
            command_check = self._check_command_injection(sanitized_query)
            validation_results["validation_details"]["command_check"] = command_check
            
            if command_check["is_command_injection"]:
                validation_results["is_safe"] = False
                validation_results["blocked"] = True
                validation_results["block_reason"] = "Command injection attempt detected"
                return validation_results
            
            # 5. Path traversal detection
            path_check = self._check_path_traversal(sanitized_query)
            validation_results["validation_details"]["path_check"] = path_check
            
            if path_check["is_path_traversal"]:
                validation_results["is_safe"] = False
                validation_results["blocked"] = True
                validation_results["block_reason"] = "Path traversal attempt detected"
                return validation_results
            
            # 6. Subscription validation
            subscription_check = await self._validate_subscription(workspace_id, db)
            validation_results["validation_details"]["subscription_check"] = subscription_check
            
            if not subscription_check["is_valid"]:
                validation_results["is_safe"] = False
                validation_results["blocked"] = True
                validation_results["block_reason"] = "Invalid subscription"
                return validation_results
            
            # 7. Channel permissions validation
            permission_check = await self._validate_channel_permissions(
                user_id, workspace_id, channel_id, db
            )
            validation_results["validation_details"]["permission_check"] = permission_check
            
            if not permission_check["has_permission"]:
                validation_results["is_safe"] = False
                validation_results["blocked"] = True
                validation_results["block_reason"] = "Insufficient channel permissions"
                return validation_results
            
            # 8. Query length and complexity validation
            complexity_check = self._validate_query_complexity(sanitized_query)
            validation_results["validation_details"]["complexity_check"] = complexity_check
            
            if complexity_check["is_too_complex"]:
                validation_results["warnings"].append("Query is very complex and may take longer to process")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in security validation: {e}")
            return {
                "is_safe": False,
                "sanitized_query": query,
                "warnings": ["Security validation failed"],
                "blocked": True,
                "block_reason": "Security validation error",
                "validation_details": {"error": str(e)}
            }
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize the query by removing potentially dangerous content."""
        # HTML escape
        sanitized = html.escape(query)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Remove control characters except newlines and tabs
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def _check_malicious_content(self, query: str) -> Dict[str, Any]:
        """Check for malicious content patterns."""
        for pattern in self.malicious_patterns:
            if re.search(pattern, query, re.IGNORECASE | re.DOTALL):
                return {
                    "is_malicious": True,
                    "matched_pattern": pattern,
                    "severity": "high"
                }
        
        return {"is_malicious": False, "severity": "none"}
    
    def _check_sql_injection(self, query: str) -> Dict[str, Any]:
        """Check for SQL injection patterns."""
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return {
                    "is_sql_injection": True,
                    "matched_pattern": pattern,
                    "severity": "high"
                }
        
        return {"is_sql_injection": False, "severity": "none"}
    
    def _check_command_injection(self, query: str) -> Dict[str, Any]:
        """Check for command injection patterns."""
        for pattern in self.command_injection_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return {
                    "is_command_injection": True,
                    "matched_pattern": pattern,
                    "severity": "high"
                }
        
        return {"is_command_injection": False, "severity": "none"}
    
    def _check_path_traversal(self, query: str) -> Dict[str, Any]:
        """Check for path traversal patterns."""
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return {
                    "is_path_traversal": True,
                    "matched_pattern": pattern,
                    "severity": "high"
                }
        
        return {"is_path_traversal": False, "severity": "none"}
    
    async def _validate_subscription(self, workspace_id: int, db: AsyncSession) -> Dict[str, Any]:
        """Validate workspace subscription status."""
        try:
            result = await db.execute(
                select(Workspace).where(Workspace.id == workspace_id)
            )
            workspace = result.scalar_one_or_none()
            
            if not workspace:
                return {
                    "is_valid": False,
                    "reason": "Workspace not found"
                }
            
            # Check if workspace has active subscription
            # For now, assume all workspaces are active (implement subscription logic later)
            subscription_active = True
            
            # Check subscription limits
            subscription_limits = {
                "max_queries_per_day": 1000,
                "max_queries_per_hour": 100,
                "max_queries_per_minute": 10
            }
            
            return {
                "is_valid": subscription_active,
                "subscription_limits": subscription_limits,
                "workspace_name": workspace.name
            }
            
        except Exception as e:
            logger.error(f"Error validating subscription: {e}")
            return {
                "is_valid": False,
                "reason": f"Validation error: {str(e)}"
            }
    
    async def _validate_channel_permissions(
        self, 
        user_id: str, 
        workspace_id: int, 
        channel_id: str, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Validate user permissions for the channel."""
        try:
            # Check if user exists in workspace
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
                return {
                    "has_permission": False,
                    "reason": "User not found in workspace"
                }
            
            # Check if conversation exists for this channel
            conversation_result = await db.execute(
                select(Conversation).where(
                    and_(
                        Conversation.slack_channel_id == channel_id,
                        Conversation.workspace_id == workspace_id
                    )
                )
            )
            conversation = conversation_result.scalar_one_or_none()
            
            if not conversation:
                return {
                    "has_permission": False,
                    "reason": "Channel not found in workspace"
                }
            
            # For now, assume all users have access to all channels in their workspace
            # Implement more granular permissions later
            return {
                "has_permission": True,
                "user_id": user.id,
                "channel_name": conversation.slack_channel_name,
                "permission_level": "read_write"
            }
            
        except Exception as e:
            logger.error(f"Error validating channel permissions: {e}")
            return {
                "has_permission": False,
                "reason": f"Permission validation error: {str(e)}"
            }
    
    def _validate_query_complexity(self, query: str) -> Dict[str, Any]:
        """Validate query complexity and length."""
        word_count = len(query.split())
        char_count = len(query)
        
        # Check for very long queries
        if char_count > 2000:
            return {
                "is_too_complex": True,
                "reason": "Query too long",
                "char_count": char_count,
                "word_count": word_count
            }
        
        # Check for very short queries
        if word_count < 2:
            return {
                "is_too_complex": False,
                "reason": "Query too short",
                "char_count": char_count,
                "word_count": word_count
            }
        
        # Check for excessive special characters
        special_char_count = len(re.findall(r'[^\w\s]', query))
        if special_char_count > word_count * 0.5:  # More than 50% special chars
            return {
                "is_too_complex": True,
                "reason": "Too many special characters",
                "char_count": char_count,
                "word_count": word_count,
                "special_char_count": special_char_count
            }
        
        return {
            "is_too_complex": False,
            "char_count": char_count,
            "word_count": word_count,
            "special_char_count": special_char_count
        }
