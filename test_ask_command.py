#!/usr/bin/env python3
"""Test script for the /ask command endpoint."""

import requests
import json

def test_ask_command():
    """Test the /ask command endpoint."""
    print("🧪 Testing the /ask command endpoint...")
    
    # Test data for the /ask command
    command_data = {
        "user_id": "U9999999999",  # Test user from our previous tests
        "channel_id": "C041D76JMRA",  # Test channel
        "team_id": "T0420EE1VQ8",  # Real workspace ID from database
        "text": "What database should we use for JSON support?",
        "response_url": "https://hooks.slack.com/commands/T1234567890/1234567890/abcdefghijklmnop",
        "command": "/ask"
    }
    
    try:
        # Send POST request to the /ask command endpoint
        response = requests.post(
            "http://localhost:8000/api/v1/slack/commands/ask",
            data=command_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        print(f"✅ Response Status: {response.status_code}")
        print(f"✅ Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"✅ Response Data: {json.dumps(response_data, indent=2)}")
            
            # Check if the response has the expected structure
            if "text" in response_data and "response_type" in response_data:
                print("✅ Response structure is correct")
                
                # Check if it mentions the query
                if "database" in response_data["text"].lower():
                    print("✅ Response contains the query text")
                else:
                    print("⚠️ Response doesn't contain the query text")
                    
            else:
                print("❌ Response structure is incorrect")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"❌ Response text: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the app is running on localhost:8000")
    except Exception as e:
        print(f"❌ Error testing /ask command: {e}")

def test_ask_command_missing_params():
    """Test the /ask command with missing parameters."""
    print("\n🧪 Testing /ask command with missing parameters...")
    
    # Test with missing user_id
    command_data = {
        "channel_id": "C041D76JMRA",
        "team_id": "T1234567890",
        "text": "What database should we use?",
        "command": "/ask"
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/slack/commands/ask",
            data=command_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        print(f"✅ Response Status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"✅ Response Data: {json.dumps(response_data, indent=2)}")
            
            # Should return an error message
            if "Missing required command parameters" in response_data.get("text", ""):
                print("✅ Correctly handled missing parameters")
            else:
                print("⚠️ Didn't handle missing parameters as expected")
                
    except Exception as e:
        print(f"❌ Error testing missing parameters: {e}")

if __name__ == "__main__":
    print("🚀 Testing /ask Command Endpoint")
    print("=" * 50)
    
    # Test 1: Normal command
    test_ask_command()
    
    # Test 2: Missing parameters
    test_ask_command_missing_params()
    
    print("\n" + "=" * 50)
    print("🏁 Testing completed!")
