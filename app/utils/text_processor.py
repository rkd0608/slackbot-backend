"""Text processing utilities for message analysis and cleaning."""

import re
from typing import List, Set
from urllib.parse import urlparse
import html
import unicodedata

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove Slack-specific formatting
    text = remove_slack_formatting(text)
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def remove_slack_formatting(text: str) -> str:
    """Remove Slack-specific formatting from text."""
    if not text:
        return ""
    
    # Remove user mentions: <@U12345678>
    text = re.sub(r'<@[A-Z0-9]+>', '', text)
    
    # Remove channel mentions: <#C12345678>
    text = re.sub(r'<#[A-Z0-9]+>', '', text)
    
    # Remove URL formatting: <https://example.com|Link Text>
    text = re.sub(r'<(https?://[^|>]+)\|[^>]+>', r'\1', text)
    text = re.sub(r'<(https?://[^>]+)>', r'\1', text)
    
    # Remove bold formatting: *bold text*
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Remove italic formatting: _italic text_
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # Remove strikethrough: ~strikethrough text~
    text = re.sub(r'~(.*?)~', r'\1', text)
    
    # Remove code formatting: `code`
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove code blocks: ```code block```
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Remove inline code blocks: `code`
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove emoji: :emoji_name:
    text = re.sub(r':[a-zA-Z0-9_+-]+:', '', text)
    
    return text

def extract_mentions(text: str) -> List[str]:
    """Extract user and channel mentions from text."""
    mentions = []
    
    if not text:
        return mentions
    
    # Extract user mentions: <@U12345678>
    user_mentions = re.findall(r'<@([A-Z0-9]+)>', text)
    mentions.extend([f"user:{uid}" for uid in user_mentions])
    
    # Extract channel mentions: <#C12345678>
    channel_mentions = re.findall(r'<#([A-Z0-9]+)>', text)
    mentions.extend([f"channel:{cid}" for cid in channel_mentions])
    
    return mentions

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    if not text:
        return []
    
    # URL pattern that matches http/https URLs
    url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
    
    urls = re.findall(url_pattern, text)
    
    # Clean and validate URLs
    cleaned_urls = []
    for url in urls:
        try:
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc:
                cleaned_urls.append(url)
        except Exception:
            continue
    
    return cleaned_urls

def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text."""
    if not text:
        return []
    
    # Find hashtags: #hashtag
    hashtags = re.findall(r'#(\w+)', text)
    return hashtags

def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text."""
    if not text:
        return []
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails

def is_question(text: str) -> bool:
    """Check if text is a question."""
    if not text:
        return False
    
    # Question words
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose', 'whom']
    
    # Check for question mark
    if '?' in text:
        return True
    
    # Check for question words at the beginning
    text_lower = text.lower().strip()
    for word in question_words:
        if text_lower.startswith(word):
            return True
    
    return False

def get_sentiment_indicators(text: str) -> dict:
    """Extract basic sentiment indicators from text."""
    if not text:
        return {"positive": 0, "negative": 0, "neutral": 0}
    
    text_lower = text.lower()
    
    # Positive words
    positive_words = [
        'good', 'great', 'excellent', 'awesome', 'amazing', 'wonderful', 'fantastic',
        'perfect', 'brilliant', 'outstanding', 'superb', 'terrific', 'marvelous',
        'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'grateful'
    ]
    
    # Negative words
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'dreadful', 'terrible', 'awful',
        'hate', 'dislike', 'hate', 'angry', 'frustrated', 'disappointed', 'upset',
        'sad', 'unhappy', 'worried', 'concerned', 'annoyed', 'irritated'
    ]
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in word in negative_words if word in text_lower)
    
    return {
        "positive": positive_count,
        "negative": negative_count,
        "neutral": max(0, len(text.split()) - positive_count - negative_count)
    }

def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract potential keywords from text."""
    if not text:
        return []
    
    # Clean text
    cleaned_text = clean_text(text)
    
    # Split into words and filter
    words = cleaned_text.split()
    
    # Filter words by length and common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    keywords = []
    for word in words:
        word_lower = word.lower()
        if (len(word_lower) >= min_length and 
            word_lower not in stop_words and
            word_lower.isalpha()):
            keywords.append(word_lower)
    
    return list(set(keywords))  # Remove duplicates

def calculate_readability_score(text: str) -> float:
    """Calculate a simple readability score (lower = more complex)."""
    if not text:
        return 0.0
    
    # Clean text
    cleaned_text = clean_text(text)
    
    # Count sentences, words, and syllables
    sentences = len(re.split(r'[.!?]+', cleaned_text))
    words = len(cleaned_text.split())
    
    if sentences == 0 or words == 0:
        return 0.0
    
    # Simple Flesch Reading Ease approximation
    # Higher score = easier to read
    avg_sentence_length = words / sentences
    
    # Rough approximation (not exact Flesch score)
    if avg_sentence_length <= 8:
        return 90.0
    elif avg_sentence_length <= 12:
        return 80.0
    elif avg_sentence_length <= 16:
        return 70.0
    elif avg_sentence_length <= 20:
        return 60.0
    else:
        return 50.0

def normalize_text(text: str) -> str:
    """Normalize text for consistent processing."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation (keep spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def get_text_statistics(text: str) -> dict:
    """Get comprehensive text statistics."""
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0,
            "avg_word_length": 0.0,
            "avg_sentence_length": 0.0,
            "readability_score": 0.0
        }
    
    cleaned_text = clean_text(text)
    
    # Basic counts
    char_count = len(cleaned_text)
    word_count = len(cleaned_text.split())
    sentence_count = len(re.split(r'[.!?]+', cleaned_text))
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    
    # Averages
    avg_word_length = char_count / word_count if word_count > 0 else 0.0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0.0
    
    # Readability
    readability_score = calculate_readability_score(text)
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "avg_word_length": round(avg_word_length, 2),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "readability_score": round(readability_score, 2)
    }
