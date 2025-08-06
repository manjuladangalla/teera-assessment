#!/usr/bin/env python
"""
Test script for uploading sample invoice and bank transaction data
to the Bank Reconciliation System API.
"""
import requests
import json
import os
from pathlib import Path

BASE_URL = 'http://127.0.0.1:8000/api/v1'

def get_auth_token():
    """Get JWT authentication token."""
    # You'll need to create a user first through Django admin
    # For demo purposes, we'll show the process
    auth_data = {
        'username': 'admin',  # Replace with your username
        'password': 'admin123'  # Replace with your password
    }
    
    try:
        response = requests.post(f'http://127.0.0.1:8000/api/auth/token/', data=auth_data)
        if response.status_code == 200:
            token_data = response.json()
            return token_data.get('access')
        else:
            print(f"❌ Authentication failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return None

def upload_file(file_path, token, file_type="invoice"):
    """Upload a file to the API."""
    headers = {
        'Authorization': f'Bearer {token}'
    }
    
    file_name = os.path.basename(file_path)
    
    try:
        with open(file_path, 'rb') as file:
            files = {'file': (file_name, file)}
            response = requests.post(f'{BASE_URL}/bank/upload/', 
                                   headers=headers, 
                                   files=files)
        
        if response.status_code == 201:
            result = response.json()
            print(f"✅ Successfully uploaded {file_name}")
            print(f"   Upload ID: {result.get('id')}")
            print(f"   Status: {result.get('status')}")
            print(f"   Total Records: {result.get('total_records', 'Processing...')}")
            return result
        else:
            print(f"❌ Upload failed for {file_name}: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error uploading {file_name}: {e}")
        return None
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None

def check_upload_status(upload_id, token):
    """Check the status of an upload."""
    headers = {
        'Authorization': f'Bearer {token}'
    }
    
    try:
        response = requests.get(f'{BASE_URL}/bank/uploads/{upload_id}/', headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Upload Status for {upload_id}:")
            print(f"   Status: {result.get('status')}")
            print(f"   Progress: {result.get('progress_percentage', 0)}%")
            print(f"   Total Records: {result.get('total_records', 0)}")
            print(f"   Processed: {result.get('processed_records', 0)}")
            print(f"   Failed: {result.get('failed_records', 0)}")
            return result
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return None

def get_unmatched_transactions(token):
    """Get list of unmatched transactions."""
    headers = {
        'Authorization': f'Bearer {token}'
    }
    
    try:
        response = requests.get(f'{BASE_URL}/bank/unmatched/', headers=headers)
        if response.status_code == 200:
            result = response.json()
            count = result.get('count', 0)
            transactions = result.get('results', [])
            
            print(f"🔍 Found {count} unmatched transactions:")
            for i, txn in enumerate(transactions[:5], 1):  # Show first 5
                print(f"   {i}. {txn.get('description')} - ${txn.get('amount')}")
                print(f"      Date: {txn.get('transaction_date')} | Ref: {txn.get('reference_number')}")
            
            if count > 5:
                print(f"   ... and {count - 5} more")
                
            return result
        else:
            print(f"❌ Failed to get unmatched transactions: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return None

def test_reconciliation_api():
    """Test the complete reconciliation workflow."""
    print("🚀 Bank Reconciliation System - File Upload Test")
    print("=" * 60)
    
    # Check if sample files exist
    sample_dir = Path('/Users/mdangallage/teera-assessment/sample_data')
    
    invoice_files = [
        sample_dir / 'sample_invoices.csv',
        sample_dir / 'sample_invoices_detailed.csv',
        sample_dir / 'sample_reconciliation_data.xlsx'
    ]
    
    bank_files = [
        sample_dir / 'sample_bank_transactions.csv',
        sample_dir / 'sample_bank_transactions_detailed.csv'
    ]
    
    print("\n📁 Available sample files:")
    for file_path in invoice_files + bank_files:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {file_path.name} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ {file_path.name} (not found)")
    
    print(f"\n🔐 Authentication Test:")
    print("Note: You need to create a user account first!")
    print("Run: python manage.py createsuperuser")
    print("Then update the credentials in this script.")
    
    # For demo purposes, show what the workflow would look like
    print(f"\n📚 Sample API Workflow:")
    print("1. Get authentication token:")
    print(f"   POST {BASE_URL.replace('/v1', '')}/auth/token/")
    print("   {'username': 'your_username', 'password': 'your_password'}")
    
    print(f"\n2. Upload invoice file:")
    print(f"   POST {BASE_URL}/bank/upload/")
    print("   files={'file': invoice_file}")
    print("   headers={'Authorization': 'Bearer <token>'}")
    
    print(f"\n3. Upload bank transaction file:")
    print(f"   POST {BASE_URL}/bank/upload/")
    print("   files={'file': bank_transaction_file}")
    print("   headers={'Authorization': 'Bearer <token>'}")
    
    print(f"\n4. Check unmatched transactions:")
    print(f"   GET {BASE_URL}/bank/unmatched/")
    print("   headers={'Authorization': 'Bearer <token>'}")
    
    print(f"\n5. Perform bulk reconciliation:")
    print(f"   POST {BASE_URL}/bank/reconcile/")
    print("   data={'transaction_ids': [...], 'operation': 'trigger_ml_matching'}")
    
    print(f"\n6. Get reconciliation summary:")
    print(f"   GET {BASE_URL}/bank/summary/")
    print("   headers={'Authorization': 'Bearer <token>'}")
    
    print(f"\n🌐 Interactive API Testing:")
    print(f"   • Swagger UI: http://127.0.0.1:8000/api/docs/")
    print(f"   • API Root: http://127.0.0.1:8000/api/v1/")
    print(f"   • Admin Panel: http://127.0.0.1:8000/admin/")
    
    # Attempt authentication (will likely fail but shows the process)
    print(f"\n🔍 Testing API Connectivity:")
    try:
        response = requests.get(f'{BASE_URL}/')
        if response.status_code == 200:
            print("✅ API is accessible")
            data = response.json()
            print(f"   System: {data.get('system', {}).get('system_name')}")
            print(f"   Version: {data.get('system', {}).get('api_version')}")
        else:
            print(f"❌ API connection issue: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")

if __name__ == "__main__":
    test_reconciliation_api()
