#!/usr/bin/env python
"""
API Demo Script for Bank Reconciliation System
Demonstrates API usage and capabilities
"""

import requests
import json
from datetime import datetime

# Base URL for the API
BASE_URL = 'http://127.0.0.1:8000/api/v1'

def test_api_endpoints():
    """Test all available API endpoints"""
    print("Bank Reconciliation System - API Demo")
    print("=" * 50)
    
    # Test new simplified endpoints
    simplified_endpoints = [
        ('GET', '/bank/unmatched/', 'List unmatched bank transactions'),
        ('GET', '/bank/logs/', 'View reconciliation logs'),
        ('GET', '/bank/summary/', 'Download reconciliation summary'),
    ]
    
    # Test existing endpoints
    detailed_endpoints = [
        ('GET', '/bank/transactions/', 'List all bank transactions'),
        ('GET', '/bank/uploads/', 'List file uploads'),
        ('GET', '/ml/models/', 'List ML models'),
    ]
    
    print("\n🎯 TESTING NEW SIMPLIFIED ENDPOINTS:")
    print("-" * 40)
    
    for method, endpoint, description in simplified_endpoints:
        url = BASE_URL + endpoint
        try:
            response = requests.get(url, timeout=5)
            print(f"\n📡 {method} {endpoint}")
            print(f"   {description}")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'results' in data:
                    print(f"   ✅ SUCCESS - {data.get('count', 0)} items")
                elif isinstance(data, list):
                    print(f"   ✅ SUCCESS - {len(data)} items")
                else:
                    print(f"   ✅ SUCCESS - Response keys: {list(data.keys()) if isinstance(data, dict) else 'Non-dict response'}")
            elif response.status_code == 401:
                print("   🔐 Requires authentication (expected)")
            elif response.status_code == 404:
                print("   ❌ Endpoint not found")
            else:
                print(f"   ⚠️  Status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Connection error: {e}")
    
    print("\n🔧 TESTING DETAILED ENDPOINTS:")
    print("-" * 40)
    
    for method, endpoint, description in detailed_endpoints:
        url = BASE_URL + endpoint
        try:
            response = requests.get(url, timeout=5)
            print(f"\n📡 {method} {endpoint}")
            print(f"   {description}")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'results' in data:
                    print(f"   ✅ SUCCESS - {data.get('count', 0)} items")
                    if data['results']:
                        print("   📊 Sample data structure:")
                        sample = data['results'][0]
                        for key in list(sample.keys())[:3]:  # Show first 3 fields
                            print(f"     - {key}: {sample[key]}")
                elif isinstance(data, list):
                    print(f"   ✅ SUCCESS - {len(data)} items")
                    if data:
                        print("   📊 Sample fields:", list(data[0].keys())[:3])
                else:
                    print("   ✅ SUCCESS")
            elif response.status_code == 401:
                print("   🔐 Requires authentication (expected)")
            else:
                print(f"   ⚠️  Status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Connection error: {e}")
    
    print("\n" + "=" * 50)
    print("📋 API ENDPOINT SUMMARY:")
    print("✅ NEW SIMPLIFIED ENDPOINTS:")
    print("   - POST /api/v1/bank/upload/      - Upload bank statement file")
    print("   - GET  /api/v1/bank/unmatched/   - List unmatched transactions")
    print("   - POST /api/v1/bank/reconcile/   - Bulk reconcile transactions")
    print("   - GET  /api/v1/bank/logs/        - View reconciliation logs")
    print("   - GET  /api/v1/bank/summary/     - Download summary report")
    print("\n🔧 DETAILED ENDPOINTS STILL AVAILABLE:")
    print("   - All /api/v1/bank/transactions/* endpoints")
    print("   - All /api/v1/bank/uploads/* endpoints")
    print("   - All other detailed endpoints")
    
    print("\n🌐 To explore APIs interactively:")
    print("   - Open http://127.0.0.1:8000/api/v1/ in your browser")
    print("   - Use Django REST Framework's browsable API")
    print("   - Access admin at http://127.0.0.1:8000/admin/")

if __name__ == '__main__':
    test_api_endpoints()
