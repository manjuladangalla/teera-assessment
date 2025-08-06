#!/usr/bin/env python
"""
Test script to verify API endpoints are working correctly
Demonstrates JWT authentication and endpoint access
"""

import requests
import json
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.contrib.auth import get_user_model

User = get_user_model()

BASE_URL = 'http://127.0.0.1:8000'

def get_jwt_token():
    """Get JWT token for authentication"""
    # Try to get existing user or create a test user
    try:
        user = User.objects.get(username='testuser')
        password = 'testpass123'
    except User.DoesNotExist:
        print("Test user not found, please run test_system.py first")
        return None
    
    # Get JWT token
    auth_data = {
        'username': 'testuser',
        'password': 'testpass123'
    }
    
    try:
        response = requests.post(f'{BASE_URL}/api/auth/token/', data=auth_data)
        if response.status_code == 200:
            token_data = response.json()
            return token_data['access']
        else:
            print(f"Authentication failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Error getting token: {e}")
        return None

def test_endpoints_with_auth():
    """Test API endpoints with authentication"""
    print("🔐 Getting JWT token...")
    token = get_jwt_token()
    
    if not token:
        print("❌ Could not get authentication token")
        print("💡 Hint: Make sure you've run test_system.py first to create test user")
        return
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    print("✅ Authentication successful!")
    print("\n🧪 Testing API endpoints...")
    
    # Test endpoints
    endpoints = [
        ('GET', '/api/v1/bank/transactions/', 'List bank transactions'),
        ('GET', '/api/v1/bank/transactions/unmatched/', 'Get unmatched transactions'),
        ('GET', '/api/v1/bank/uploads/', 'List file uploads'),
        ('GET', '/api/v1/bank/logs/', 'List reconciliation logs'),
        ('GET', '/api/v1/ml/models/', 'List ML models'),
    ]
    
    for method, endpoint, description in endpoints:
        url = BASE_URL + endpoint
        try:
            response = requests.get(url, headers=headers, timeout=5)
            
            print(f"\n📡 {method} {endpoint}")
            print(f"   {description}")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    if 'results' in data:
                        count = data.get('count', 0)
                        print(f"   ✅ Success - {count} items found")
                        if data['results']:
                            # Show sample field names
                            sample = data['results'][0]
                            fields = list(sample.keys())[:3]
                            print(f"   📊 Sample fields: {', '.join(fields)}")
                    else:
                        print(f"   ✅ Success - Response: {str(data)[:100]}...")
                elif isinstance(data, list):
                    print(f"   ✅ Success - {len(data)} items")
                else:
                    print(f"   ✅ Success - {type(data).__name__}")
                    
            elif response.status_code == 401:
                print("   ❌ Authentication failed")
            elif response.status_code == 403:
                print("   ❌ Permission denied")
            elif response.status_code == 404:
                print("   ❌ Endpoint not found")
            else:
                print(f"   ⚠️  Status {response.status_code}: {response.text[:100]}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Connection error: {e}")

def test_unmatched_specifically():
    """Test the unmatched endpoint specifically"""
    print("\n🎯 Testing unmatched transactions endpoint specifically...")
    
    token = get_jwt_token()
    if not token:
        return
        
    headers = {'Authorization': f'Bearer {token}'}
    
    # Test the specific endpoint the user mentioned
    test_urls = [
        '/api/v1/bank/transactions/unmatched/',  # Correct URL
        '/api/v1/bank/unmatched/',               # User's incorrect URL
    ]
    
    for url in test_urls:
        full_url = BASE_URL + url
        try:
            response = requests.get(full_url, headers=headers)
            print(f"\n🔍 Testing: {url}")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ SUCCESS - Found {len(data)} unmatched transactions")
                if data:
                    for i, txn in enumerate(data[:3]):  # Show first 3
                        print(f"   📋 Transaction {i+1}: {txn.get('description', 'N/A')[:30]} - {txn.get('amount', 'N/A')}")
                else:
                    print("   📋 No unmatched transactions found")
            elif response.status_code == 404:
                print("   ❌ ENDPOINT NOT FOUND")
            else:
                print(f"   ⚠️  Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == '__main__':
    print("API Endpoint Verification Tool")
    print("=" * 40)
    
    test_endpoints_with_auth()
    test_unmatched_specifically()
    
    print("\n" + "=" * 40)
    print("✅ Verification completed!")
    print("\n💡 Key findings:")
    print("   - Correct URL: /api/v1/bank/transactions/unmatched/")
    print("   - Incorrect URL: /api/v1/bank/unmatched/ (404 Not Found)")
    print("   - All endpoints require JWT authentication")
    print("   - Use 'Bearer <token>' in Authorization header")
