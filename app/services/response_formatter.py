"""Professional response formatting system with Slack Block Kit integration.

This module provides structured response templates and formatting for different
content types including decisions, processes, technical solutions, and more.
Implements consistent visual hierarchy and interactive elements.
"""

import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from loguru import logger

from .intent_classifier import IntentClassificationResult


@dataclass
class ResponseTemplate:
    """Template structure for different response types."""
    template_type: str
    blocks: List[Dict[str, Any]]
    text_fallback: str
    metadata: Dict[str, Any]


@dataclass
class ContentData:
    """Structured content data for response generation."""
    main_content: str
    sources: List[Dict[str, Any]]
    next_steps: List[str]
    related_info: List[str]
    decision_info: Optional[Dict[str, Any]] = None
    process_steps: Optional[List[Dict[str, Any]]] = None
    technical_details: Optional[Dict[str, Any]] = None
    verification_links: List[str] = None


class ResponseFormatter:
    """
    Professional response formatting with Slack Block Kit integration.
    
    Provides structured templates for different content types with:
    - Consistent visual hierarchy
    - Interactive elements
    - Professional formatting standards
    - Source attribution
    - Action buttons
    """
    
    def __init__(self):
        # Template configurations
        self.template_configs = {
            'decision_response': {
                'title': 'Decision Information',
                'icon': '',  # Removed emoji
                'color': '#2E8B57'
            },
            'process_response': {
                'title': 'Process Guide',
                'icon': '',  # Removed emoji
                'color': '#4169E1'
            },
            'technical_solution': {
                'title': 'Technical Solution',
                'icon': '🔧',
                'color': '#FF6347'
            },
            'general_info': {
                'title': 'Information',
                'icon': '',  # Removed emoji
                'color': '#4682B4'
            },
            'no_information': {
                'title': 'No Information Found',
                'icon': '',  # Removed emoji
                'color': '#DC143C'
            },
            'social_response': {
                'title': 'Response',
                'icon': '👋',
                'color': '#32CD32'
            }
        }
        
        # Formatting standards
        self.formatting_standards = {
            'max_section_length': 3000,
            'max_source_count': 5,
            'max_next_steps': 5,
            'max_related_info': 3
        }

    async def format_response(
        self,
        content_data: ContentData,
        intent_result: IntentClassificationResult,
        query_text: str,
        query_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Format response based on content type and intent.
        
        Returns structured Slack Block Kit response with appropriate template.
        """
        try:
            # Determine template type based on intent and content
            template_type = self._determine_template_type(content_data, intent_result)
            
            # Generate template
            template = await self._generate_template(
                template_type, content_data, intent_result, query_text, query_id
            )
            
            return {
                "response_type": "in_channel",
                "text": template.text_fallback,
                "blocks": template.blocks,
                "metadata": template.metadata
            }
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return self._create_error_response(query_text, str(e))

    def _determine_template_type(
        self, 
        content_data: ContentData, 
        intent_result: IntentClassificationResult
    ) -> str:
        """Determine appropriate template type based on content and intent."""
        
        # Check for specific content types first
        if content_data.decision_info:
            return 'decision_response'
        elif content_data.process_steps:
            return 'process_response'
        elif content_data.technical_details:
            return 'technical_solution'
        elif intent_result.intent == 'social_interaction':
            return 'social_response'
        elif not content_data.sources and not content_data.main_content.strip():
            return 'no_information'
        else:
            return 'general_info'

    async def _generate_template(
        self,
        template_type: str,
        content_data: ContentData,
        intent_result: IntentClassificationResult,
        query_text: str,
        query_id: Optional[int]
    ) -> ResponseTemplate:
        """Generate specific template based on type."""
        
        if template_type == 'decision_response':
            return self._create_decision_template(content_data, intent_result, query_text, query_id)
        elif template_type == 'process_response':
            return self._create_process_template(content_data, intent_result, query_text, query_id)
        elif template_type == 'technical_solution':
            return self._create_technical_template(content_data, intent_result, query_text, query_id)
        elif template_type == 'social_response':
            return self._create_social_template(content_data, intent_result, query_text, query_id)
        elif template_type == 'no_information':
            return self._create_no_info_template(content_data, intent_result, query_text, query_id)
        else:
            return self._create_general_template(content_data, intent_result, query_text, query_id)

    def _create_decision_template(
        self,
        content_data: ContentData,
        intent_result: IntentClassificationResult,
        query_text: str,
        query_id: Optional[int]
    ) -> ResponseTemplate:
        """Create decision response template."""
        config = self.template_configs['decision_response']
        decision_info = content_data.decision_info or {}
        
        blocks = []
        
        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{config['icon']} {config['title']}"
            }
        })
        
        # Decision outcome
        if decision_info.get('outcome'):
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Decision:* {decision_info['outcome']}"
                }
            })
        
        # Decision maker and timing
        if decision_info.get('decision_maker') or decision_info.get('date'):
            info_text = ""
            if decision_info.get('decision_maker'):
                info_text += f"*Decision Maker:* {decision_info['decision_maker']}"
            if decision_info.get('date'):
                if info_text:
                    info_text += " | "
                info_text += f"*Date:* {decision_info['date']}"
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": info_text
                }
            })
        
        # Rationale
        if decision_info.get('rationale'):
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Rationale:*\n{decision_info['rationale']}"
                }
            })
        
        # Main content
        if content_data.main_content:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Details:*\n{content_data.main_content}"
                }
            })
        
        # Next steps
        if content_data.next_steps:
            steps_text = "\n".join([f"• {step}" for step in content_data.next_steps[:5]])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Next Steps:*\n{steps_text}"
                }
            })
        
        # Sources
        if content_data.sources:
            self._add_sources_section(blocks, content_data.sources)
        
        # Action buttons
        self._add_action_buttons(blocks, query_id, config['color'])
        
        # Text fallback
        text_fallback = self._create_text_fallback(content_data, intent_result, query_text)
        
        return ResponseTemplate(
            template_type='decision_response',
            blocks=blocks,
            text_fallback=text_fallback,
            metadata={'template_type': 'decision_response', 'intent': intent_result.intent}
        )

    def _create_process_template(
        self,
        content_data: ContentData,
        intent_result: IntentClassificationResult,
        query_text: str,
        query_id: Optional[int]
    ) -> ResponseTemplate:
        """Create process response template."""
        config = self.template_configs['process_response']
        process_steps = content_data.process_steps or []
        
        blocks = []
        
        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{config['icon']} {config['title']}"
            }
        })
        
        # Process overview
        if content_data.main_content:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Overview:*\n{content_data.main_content}"
                }
            })
        
        # Process steps
        if process_steps:
            for i, step in enumerate(process_steps[:10], 1):  # Limit to 10 steps
                step_text = f"*{i}.* {step.get('description', '')}"
                if step.get('responsible'):
                    step_text += f"\n   _Responsible: {step['responsible']}_"
                if step.get('tools'):
                    step_text += f"\n   _Tools: {step['tools']}_"
                
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": step_text
                    }
                })
        
        # Prerequisites
        if content_data.related_info:
            prereq_text = "\n".join([f"• {info}" for info in content_data.related_info[:5]])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Prerequisites:*\n{prereq_text}"
                }
            })
        
        # Sources
        if content_data.sources:
            self._add_sources_section(blocks, content_data.sources)
        
        # Action buttons
        self._add_action_buttons(blocks, query_id, config['color'])
        
        # Text fallback
        text_fallback = self._create_text_fallback(content_data, intent_result, query_text)
        
        return ResponseTemplate(
            template_type='process_response',
            blocks=blocks,
            text_fallback=text_fallback,
            metadata={'template_type': 'process_response', 'intent': intent_result.intent}
        )

    def _create_technical_template(
        self,
        content_data: ContentData,
        intent_result: IntentClassificationResult,
        query_text: str,
        query_id: Optional[int]
    ) -> ResponseTemplate:
        """Create technical solution template."""
        config = self.template_configs['technical_solution']
        tech_details = content_data.technical_details or {}
        
        blocks = []
        
        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{config['icon']} {config['title']}"
            }
        })
        
        # Problem context
        if tech_details.get('problem'):
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Problem:*\n{tech_details['problem']}"
                }
            })
        
        # Solution approach
        if tech_details.get('solution'):
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Solution:*\n{tech_details['solution']}"
                }
            })
        
        # Implementation steps
        if tech_details.get('implementation'):
            impl_text = "\n".join([f"• {step}" for step in tech_details['implementation'][:8]])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Implementation Steps:*\n{impl_text}"
                }
            })
        
        # Tools and resources
        if tech_details.get('tools'):
            tools_text = "\n".join([f"• {tool}" for tool in tech_details['tools'][:5]])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Required Tools:*\n{tools_text}"
                }
            })
        
        # Verification
        if tech_details.get('verification'):
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Verification:*\n{tech_details['verification']}"
                }
            })
        
        # Main content
        if content_data.main_content:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Additional Details:*\n{content_data.main_content}"
                }
            })
        
        # Sources
        if content_data.sources:
            self._add_sources_section(blocks, content_data.sources)
        
        # Action buttons
        self._add_action_buttons(blocks, query_id, config['color'])
        
        # Text fallback
        text_fallback = self._create_text_fallback(content_data, intent_result, query_text)
        
        return ResponseTemplate(
            template_type='technical_solution',
            blocks=blocks,
            text_fallback=text_fallback,
            metadata={'template_type': 'technical_solution', 'intent': intent_result.intent}
        )

    def _create_social_template(
        self,
        content_data: ContentData,
        intent_result: IntentClassificationResult,
        query_text: str,
        query_id: Optional[int]
    ) -> ResponseTemplate:
        """Create social interaction template."""
        config = self.template_configs['social_response']
        
        blocks = []
        
        # Simple response
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{config['icon']} {content_data.main_content or 'Hello! How can I help you today?'}"
            }
        })
        
        # Text fallback
        text_fallback = content_data.main_content or "Hello! How can I help you today?"
        
        return ResponseTemplate(
            template_type='social_response',
            blocks=blocks,
            text_fallback=text_fallback,
            metadata={'template_type': 'social_response', 'intent': intent_result.intent}
        )

    def _create_no_info_template(
        self,
        content_data: ContentData,
        intent_result: IntentClassificationResult,
        query_text: str,
        query_id: Optional[int]
    ) -> ResponseTemplate:
        """Create no information found template."""
        config = self.template_configs['no_information']
        
        blocks = []
        
        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{config['icon']} {config['title']}"
            }
        })
        
        # Main message
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"I couldn't find specific information about: *{query_text}*\n\nThis might be because:\n• The information hasn't been discussed in team conversations yet\n• It's in a private channel I don't have access to\n• The topic uses different terminology than your search"
            }
        })
        
        # Suggestions
        suggestions = [
            "Try rephrasing your question with different keywords",
            "Ask in a relevant channel where the topic might have been discussed",
            "Check if there's documentation or external resources available",
            "Ask a team member who might know about this topic"
        ]
        
        suggestions_text = "\n".join([f"• {suggestion}" for suggestion in suggestions])
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Suggestions:*\n{suggestions_text}"
            }
        })
        
        # Action buttons
        self._add_action_buttons(blocks, query_id, config['color'])
        
        # Text fallback
        text_fallback = f"I couldn't find specific information about: {query_text}\n\nSuggestions:\n" + "\n".join([f"• {s}" for s in suggestions])
        
        return ResponseTemplate(
            template_type='no_information',
            blocks=blocks,
            text_fallback=text_fallback,
            metadata={'template_type': 'no_information', 'intent': intent_result.intent}
        )

    def _create_general_template(
        self,
        content_data: ContentData,
        intent_result: IntentClassificationResult,
        query_text: str,
        query_id: Optional[int]
    ) -> ResponseTemplate:
        """Create general information template."""
        config = self.template_configs['general_info']
        
        blocks = []
        
        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{config['icon']} {config['title']}"
            }
        })
        
        # Main content
        if content_data.main_content:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": content_data.main_content
                }
            })
        
        # Related information
        if content_data.related_info:
            related_text = "\n".join([f"• {info}" for info in content_data.related_info[:5]])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Related Information:*\n{related_text}"
                }
            })
        
        # Next steps
        if content_data.next_steps:
            steps_text = "\n".join([f"• {step}" for step in content_data.next_steps[:5]])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Next Steps:*\n{steps_text}"
                }
            })
        
        # Sources
        if content_data.sources:
            self._add_sources_section(blocks, content_data.sources)
        
        # Action buttons
        self._add_action_buttons(blocks, query_id, config['color'])
        
        # Text fallback
        text_fallback = self._create_text_fallback(content_data, intent_result, query_text)
        
        return ResponseTemplate(
            template_type='general_info',
            blocks=blocks,
            text_fallback=text_fallback,
            metadata={'template_type': 'general_info', 'intent': intent_result.intent}
        )

    def _add_sources_section(self, blocks: List[Dict[str, Any]], sources: List[Dict[str, Any]]):
        """Add sources section to blocks."""
        if not sources:
            return
        
        # Limit sources to avoid overwhelming the response
        limited_sources = sources[:self.formatting_standards['max_source_count']]
        
        source_text = ""
        for i, source in enumerate(limited_sources, 1):
            title = source.get('title', 'Conversation')
            date = source.get('date', 'Unknown date')
            channel = source.get('channel', 'Unknown channel')
            
            source_text += f"{i}. *{title}* ({date})\n"
            if channel != 'Unknown channel':
                source_text += f"   _Channel: {channel}_\n"
            if source.get('excerpt'):
                excerpt = source['excerpt'][:100] + "..." if len(source['excerpt']) > 100 else source['excerpt']
                source_text += f"   _{excerpt}_\n"
            source_text += "\n"
        
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Sources:*\n{source_text.strip()}"
            }
        })

    def _add_action_buttons(self, blocks: List[Dict[str, Any]], query_id: Optional[int], color: str):
        """Add action buttons to blocks."""
        buttons = [
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "👍 Helpful"
                },
                "style": "primary",
                "action_id": "feedback_helpful",
                "value": str(query_id) if query_id else "0"
            },
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "👎 Not Helpful"
                },
                "action_id": "feedback_not_helpful",
                "value": str(query_id) if query_id else "0"
            },
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "🔗 View Sources"
                },
                "action_id": "view_sources",
                "value": str(query_id) if query_id else "0"
            }
        ]
        
        blocks.append({
            "type": "actions",
            "elements": buttons
        })

    def _create_text_fallback(
        self,
        content_data: ContentData,
        intent_result: IntentClassificationResult,
        query_text: str
    ) -> str:
        """Create text fallback for Slack compatibility."""
        text_parts = []
        
        # Main content
        if content_data.main_content:
            text_parts.append(content_data.main_content)
        
        # Decision info
        if content_data.decision_info:
            decision = content_data.decision_info
            text_parts.append(f"Decision: {decision.get('outcome', 'N/A')}")
            if decision.get('decision_maker'):
                text_parts.append(f"Decision Maker: {decision['decision_maker']}")
            if decision.get('rationale'):
                text_parts.append(f"Rationale: {decision['rationale']}")
        
        # Process steps
        if content_data.process_steps:
            text_parts.append("Process Steps:")
            for i, step in enumerate(content_data.process_steps[:5], 1):
                text_parts.append(f"{i}. {step.get('description', '')}")
        
        # Next steps
        if content_data.next_steps:
            text_parts.append("Next Steps:")
            for step in content_data.next_steps[:5]:
                text_parts.append(f"• {step}")
        
        # Sources
        if content_data.sources:
            text_parts.append("Sources:")
            for i, source in enumerate(content_data.sources[:3], 1):
                title = source.get('title', 'Conversation')
                date = source.get('date', 'Unknown date')
                text_parts.append(f"{i}. {title} ({date})")
        
        return "\n\n".join(text_parts)

    def _create_error_response(self, query_text: str, error_message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "response_type": "ephemeral",
            "text": f"Error processing your request: {error_message}\n\nPlease try again or contact support if the issue persists.",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Error Processing Request*\n\nI encountered an error while processing: *{query_text}*\n\nError: {error_message}\n\nPlease try again or contact support if the issue persists."
                    }
                }
            ]
        }
