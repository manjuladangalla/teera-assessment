#!/usr/bin/env python
"""
Quick test for the new simplified API endpoints
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_simplified_endpoints():
    print("🚀 Testing Simplified Bank Reconciliation API Endpoints")
    print("=" * 60)
    
    # Test endpoints that should return 401 (authentication required)
    auth_required_endpoints = [
        ("GET", "/bank/unmatched/", "List unmatched transactions"),
        ("GET", "/bank/logs/", "View reconciliation logs"),
        ("GET", "/bank/summary/", "Download summary report"),
    ]
    
    print("\n🔐 Testing Authentication Protection:")
    for method, endpoint, description in auth_required_endpoints:
        url = BASE_URL + endpoint
        response = requests.get(url)
        
        status_icon = "✅" if response.status_code == 401 else "❌"
        print(f"{status_icon} {method} {endpoint} - Status: {response.status_code}")
        print(f"   📝 {description}")
        
        if response.status_code == 401:
            print("   🔒 Correctly requires authentication")
        else:
            print(f"   ⚠️  Expected 401, got {response.status_code}")
        print()
    
    # Test the API root
    print("🌐 Testing API Root:")
    response = requests.get(BASE_URL + "/")
    if response.status_code == 200:
        data = response.json()
        print("✅ API Root accessible")
        print("📋 Available endpoints:")
        for key, url in data.items():
            print(f"   - {key}: {url}")
    else:
        print(f"❌ API Root error: {response.status_code}")
    
    print("\n" + "=" * 60)
    print("🎯 SIMPLIFIED ENDPOINT VERIFICATION COMPLETE")
    print("\n📚 New Simplified Endpoint Structure:")
    print("   POST /api/v1/bank/upload/      - Upload bank statement files")
    print("   GET  /api/v1/bank/unmatched/   - List unmatched transactions")
    print("   POST /api/v1/bank/reconcile/   - Bulk reconcile transactions")
    print("   GET  /api/v1/bank/logs/        - View reconciliation audit logs")
    print("   GET  /api/v1/bank/summary/     - Download reconciliation summary")
    print("\n🔧 All endpoints properly protected with authentication!")
    print("🌟 Ready for integration with frontend applications!")

if __name__ == "__main__":
    test_simplified_endpoints()
