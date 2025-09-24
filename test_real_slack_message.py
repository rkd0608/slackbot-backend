#!/usr/bin/env python3
"""Test script to simulate a real Slack message event."""

import sys
import os
from pathlib import Path
import json
import hmac
import hashlib
import time

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_slack_signature(body: str, timestamp: str, secret: str) -> str:
    """Create a Slack signature for testing."""
    sig_basestring = f"v0:{timestamp}:{body}"
    signature = hmac.new(secret.encode(), sig_basestring.encode(), hashlib.sha256).hexdigest()
    return f"v0={signature}"

def test_slack_message_event():
    """Test sending a simulated Slack message event."""
    print("🧪 Testing Slack message event processing...")
    
    # Simulate a real Slack message event
    event_data = {
        "token": "verification_token",
        "team_id": "T0420EE1VQ8",  # Your workspace ID
        "api_app_id": "A09CHQSD953",
        "event": {
            "type": "message",
            "channel": "C1234567890",
            "user": "U1234567890",
            "text": "This is a test message from Slack to verify the bot is working!",
            "ts": "1234567890.123456",
            "event_ts": "1234567890.123456",
            "channel_type": "channel"
        },
        "type": "event_callback",
        "event_id": "Ev1234567890",
        "event_time": int(time.time())
    }
    
    body = json.dumps(event_data)
    timestamp = str(int(time.time()))
    
    # Get the signing secret from environment (this is what your app uses)
    signing_secret = "5a78a190810d4c4da6e40bc678f5f7ae"  # From your env
    
    # Create signature
    signature = create_slack_signature(body, timestamp, signing_secret)
    
    print(f"📝 Event data: {json.dumps(event_data, indent=2)}")
    print(f"🔑 Timestamp: {timestamp}")
    print(f"✍️  Signature: {signature}")
    
    # Test the endpoint
    import requests
    
    headers = {
        "Content-Type": "application/json",
        "X-Slack-Request-Timestamp": timestamp,
        "X-Slack-Signature": signature
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/slack/events",
            headers=headers,
            data=body
        )
        
        print(f"📡 Response Status: {response.status_code}")
        print(f"📡 Response Body: {response.text}")
        
        if response.status_code == 200:
            print("✅ Slack message event processed successfully!")
            return True
        else:
            print(f"❌ Failed to process Slack message event: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Slack message event: {e}")
        return False

if __name__ == "__main__":
    success = test_slack_message_event()
    sys.exit(0 if success else 1)
