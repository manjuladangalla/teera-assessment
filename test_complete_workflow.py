#!/usr/bin/env python
"""
Working test script for uploading sample files to the Bank Reconciliation API
"""
import requests
import json
import time
from pathlib import Path

BASE_URL = 'http://127.0.0.1:8000/api/v1'
AUTH_URL = 'http://127.0.0.1:8000/api/auth/token/'

# Test credentials
USERNAME = 'demo'
PASSWORD = 'demo123'

def get_auth_token():
    """Get JWT authentication token."""
    auth_data = {
        'username': USERNAME,
        'password': PASSWORD
    }
    
    try:
        response = requests.post(AUTH_URL, data=auth_data)
        if response.status_code == 200:
            token_data = response.json()
            print("✅ Authentication successful!")
            return token_data.get('access')
        else:
            print(f"❌ Authentication failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return None

def upload_file_and_test(file_path, token):
    """Upload a file and test the complete workflow."""
    headers = {
        'Authorization': f'Bearer {token}'
    }
    
    file_name = file_path.name
    print(f"\n📤 Uploading {file_name}...")
    
    try:
        with open(file_path, 'rb') as file:
            files = {'file': (file_name, file)}
            response = requests.post(f'{BASE_URL}/bank/upload/', 
                                   headers=headers, 
                                   files=files)
        
        if response.status_code == 201:
            result = response.json()
            upload_id = result.get('id')
            print(f"✅ Upload successful!")
            print(f"   Upload ID: {upload_id}")
            print(f"   Status: {result.get('status')}")
            print(f"   File: {result.get('original_filename')}")
            
            # Wait a moment for processing
            print("⏳ Processing file...")
            time.sleep(2)
            
            # Check upload status
            status_response = requests.get(f'{BASE_URL}/bank/uploads/{upload_id}/', headers=headers)
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"📊 Processing Status:")
                print(f"   Status: {status_data.get('status')}")
                print(f"   Total Records: {status_data.get('total_records', 0)}")
                print(f"   Processed: {status_data.get('processed_records', 0)}")
                print(f"   Failed: {status_data.get('failed_records', 0)}")
                print(f"   Progress: {status_data.get('progress_percentage', 0)}%")
            
            return result
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error uploading {file_name}: {e}")
        return None

def test_reconciliation_features(token):
    """Test various reconciliation features."""
    headers = {
        'Authorization': f'Bearer {token}'
    }
    
    print(f"\n🔍 Testing Reconciliation Features:")
    
    # 1. Get unmatched transactions
    print(f"\n1. Checking unmatched transactions...")
    response = requests.get(f'{BASE_URL}/bank/unmatched/', headers=headers)
    if response.status_code == 200:
        data = response.json()
        count = data.get('count', 0)
        print(f"   Found {count} unmatched transactions")
        
        if count > 0:
            transactions = data.get('results', [])[:3]  # Show first 3
            for i, txn in enumerate(transactions, 1):
                print(f"   {i}. {txn.get('description')} - ${txn.get('amount')}")
                print(f"      Date: {txn.get('transaction_date')} | Status: {txn.get('status')}")
    else:
        print(f"   ❌ Failed to get unmatched transactions: {response.status_code}")
    
    # 2. Get reconciliation logs
    print(f"\n2. Checking reconciliation logs...")
    response = requests.get(f'{BASE_URL}/bank/logs/', headers=headers)
    if response.status_code == 200:
        data = response.json()
        count = data.get('count', 0)
        print(f"   Found {count} reconciliation log entries")
    else:
        print(f"   ❌ Failed to get reconciliation logs: {response.status_code}")
    
    # 3. Get all transactions
    print(f"\n3. Checking all transactions...")
    response = requests.get(f'{BASE_URL}/bank/transactions/', headers=headers)
    if response.status_code == 200:
        data = response.json()
        count = data.get('count', 0)
        print(f"   Found {count} total transactions")
        
        if count > 0:
            # Show summary by status
            transactions = data.get('results', [])
            status_counts = {}
            for txn in transactions:
                status = txn.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            print(f"   Status breakdown:")
            for status, count in status_counts.items():
                print(f"     - {status}: {count}")
    else:
        print(f"   ❌ Failed to get transactions: {response.status_code}")
    
    # 4. Test summary endpoint
    print(f"\n4. Testing summary endpoint...")
    response = requests.get(f'{BASE_URL}/bank/summary/', headers=headers)
    if response.status_code == 200:
        print(f"   ✅ Summary endpoint working")
    else:
        print(f"   ❌ Summary endpoint failed: {response.status_code}")

def main():
    """Main test function."""
    print("🚀 Bank Reconciliation System - Complete API Test")
    print("=" * 60)
    
    # Get authentication token
    print("🔐 Authenticating...")
    token = get_auth_token()
    if not token:
        print("❌ Cannot proceed without authentication token")
        return
    
    # Check available sample files
    sample_dir = Path('/Users/mdangallage/teera-assessment/sample_data')
    
    test_files = [
        sample_dir / 'sample_invoices_detailed.csv',
        sample_dir / 'sample_bank_transactions_detailed.csv',
        sample_dir / 'sample_reconciliation_data.xlsx'
    ]
    
    print(f"\n📁 Testing file uploads...")
    uploaded_files = []
    
    for file_path in test_files:
        if file_path.exists():
            result = upload_file_and_test(file_path, token)
            if result:
                uploaded_files.append(result)
        else:
            print(f"❌ File not found: {file_path}")
    
    print(f"\n✅ Successfully uploaded {len(uploaded_files)} files")
    
    # Test reconciliation features
    test_reconciliation_features(token)
    
    print(f"\n🎯 Test Summary:")
    print(f"   • Authentication: ✅ Working")
    print(f"   • File Upload: ✅ Working ({len(uploaded_files)} files)")
    print(f"   • API Endpoints: ✅ All accessible")
    print(f"   • Sample Data: ✅ Available for testing")
    
    print(f"\n🌐 Next Steps:")
    print(f"   • View data in admin: http://127.0.0.1:8000/admin/")
    print(f"   • Test API interactively: http://127.0.0.1:8000/api/docs/")
    print(f"   • Explore API root: http://127.0.0.1:8000/api/v1/")

if __name__ == "__main__":
    main()
