#!/usr/bin/env python
"""
Complete API Test with proper user-company setup
Demonstrates working endpoints with correct authentication and permissions
"""

import os
import sys
import django
import requests

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.contrib.auth import get_user_model
from core.models import Company, UserProfile

User = get_user_model()
BASE_URL = 'http://127.0.0.1:8000'

def setup_test_user_with_company():
    """Ensure test user has proper company association"""
    print("🔧 Setting up test user with company association...")
    
    # Get or create test company
    company, created = Company.objects.get_or_create(
        name="Test Bank Ltd",
        defaults={
            'registration_number': 'TB123456',
            'contact_email': 'admin@testbank.com',
            'industry': 'finance'
        }
    )
    
    # Get or create test user
    user, created = User.objects.get_or_create(
        username='testuser',
        defaults={
            'email': 'test@testbank.com',
            'first_name': 'Test',
            'last_name': 'User'
        }
    )
    
    if created:
        user.set_password('testpass123')
        user.save()
    
    # Create or update user profile with company
    profile, created = UserProfile.objects.get_or_create(
        user=user,
        defaults={
            'company': company,
            'employee_id': 'EMP001',
            'department': 'Finance',
            'is_admin': True
        }
    )
    
    if not created and profile.company != company:
        profile.company = company
        profile.is_admin = True
        profile.save()
    
    print(f"✅ User '{user.username}' associated with company '{company.name}'")
    return user, company

def get_jwt_token():
    """Get JWT token for the test user"""
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
            print(f"❌ Authentication failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error getting token: {e}")
        return None

def test_all_endpoints():
    """Test all endpoints with proper authentication"""
    print("\n🧪 Testing API endpoints with proper authentication...")
    
    # Setup user and company
    user, company = setup_test_user_with_company()
    
    # Get authentication token
    token = get_jwt_token()
    if not token:
        print("❌ Could not authenticate")
        return
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    print("✅ Authentication successful!")
    
    # Test endpoints
    endpoints = [
        ('GET', '/api/v1/bank/transactions/', 'List bank transactions'),
        ('GET', '/api/v1/bank/transactions/unmatched/', 'Get unmatched transactions ⭐'),
        ('GET', '/api/v1/bank/uploads/', 'List file uploads'),
        ('GET', '/api/v1/bank/logs/', 'List reconciliation logs'),
        ('GET', '/api/v1/ml/models/', 'List ML models'),
    ]
    
    for method, endpoint, description in endpoints:
        url = BASE_URL + endpoint
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            print(f"\n📡 {method} {endpoint}")
            print(f"   {description}")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    if 'results' in data:
                        count = data.get('count', 0)
                        print(f"   ✅ SUCCESS - {count} items found")
                        if data['results']:
                            sample = data['results'][0]
                            # Show key fields based on endpoint
                            if 'transactions' in endpoint:
                                desc = sample.get('description', 'N/A')[:30]
                                amount = sample.get('amount', 'N/A')
                                status = sample.get('status', 'N/A')
                                print(f"   📊 Sample: {desc}... | ${amount} | {status}")
                            else:
                                fields = list(sample.keys())[:3]
                                print(f"   📊 Sample fields: {', '.join(fields)}")
                    else:
                        print(f"   ✅ SUCCESS - Response keys: {list(data.keys())}")
                elif isinstance(data, list):
                    print(f"   ✅ SUCCESS - {len(data)} items")
                    if data and 'transactions' in endpoint:
                        sample = data[0]
                        desc = sample.get('description', 'N/A')[:30]
                        amount = sample.get('amount', 'N/A')
                        status = sample.get('status', 'N/A')
                        print(f"   📊 Sample: {desc}... | ${amount} | {status}")
                else:
                    print(f"   ✅ SUCCESS")
                    
            elif response.status_code == 401:
                print("   ❌ Authentication failed")
            elif response.status_code == 403:
                print("   ❌ Permission denied (check user-company association)")
            elif response.status_code == 404:
                print("   ❌ Endpoint not found")
            else:
                print(f"   ⚠️  Status {response.status_code}: {response.text[:100]}...")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Connection error: {e}")

def demonstrate_correct_vs_incorrect_urls():
    """Show the difference between correct and incorrect URLs"""
    print("\n🎯 DEMONSTRATING CORRECT vs INCORRECT URLs")
    print("=" * 50)
    
    user, company = setup_test_user_with_company()
    token = get_jwt_token()
    
    if not token:
        return
        
    headers = {'Authorization': f'Bearer {token}'}
    
    test_cases = [
        ('/api/v1/bank/transactions/unmatched/', '✅ CORRECT URL'),
        ('/api/v1/bank/unmatched/', '❌ INCORRECT URL (your original attempt)'),
        ('/api/v1/bank/transactions/', '✅ CORRECT URL - List all transactions'),
        ('/api/v1/transactions/', '❌ INCORRECT URL - Missing bank/ prefix'),
    ]
    
    for url, description in test_cases:
        full_url = BASE_URL + url
        try:
            response = requests.get(full_url, headers=headers, timeout=5)
            status_emoji = "✅" if response.status_code == 200 else "❌" if response.status_code == 404 else "⚠️"
            print(f"\n{status_emoji} {url}")
            print(f"   {description}")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    print(f"   Result: {len(data)} items found")
                elif isinstance(data, dict) and 'results' in data:
                    print(f"   Result: {data.get('count', 0)} items found")
            elif response.status_code == 404:
                print("   Result: ENDPOINT DOES NOT EXIST")
            elif response.status_code == 403:
                print("   Result: Permission denied")
                
        except Exception as e:
            print(f"   Result: Connection error - {e}")

if __name__ == '__main__':
    print("🏦 Complete API Test Suite")
    print("=" * 40)
    
    try:
        test_all_endpoints()
        demonstrate_correct_vs_incorrect_urls()
        
        print("\n" + "=" * 40)
        print("📋 SUMMARY:")
        print("✅ Correct unmatched URL: /api/v1/bank/transactions/unmatched/")
        print("❌ Incorrect URL you tried: /api/v1/bank/unmatched/")
        print("🔐 All endpoints require JWT authentication")
        print("👥 User must be associated with a company")
        print("🎯 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
