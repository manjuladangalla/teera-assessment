#!/usr/bin/env python3
"""
Sample API usage script for the Bank Reconciliation System.

This script demonstrates how to interact with the API programmatically.
"""

import requests
import json
from datetime import date, timedelta
import csv
import io

class ReconciliationAPIClient:
    """Client for interacting with the Reconciliation API."""
    
    def __init__(self, base_url='http://localhost:8000', username=None, password=None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.access_token = None
        
        if username and password:
            self.authenticate(username, password)
    
    def authenticate(self, username, password):
        """Authenticate with the API and get JWT token."""
        url = f"{self.base_url}/api/auth/token/"
        data = {
            'username': username,
            'password': password
        }
        
        response = self.session.post(url, json=data)
        response.raise_for_status()
        
        tokens = response.json()
        self.access_token = tokens['access']
        
        # Set authorization header for future requests
        self.session.headers.update({
            'Authorization': f'Bearer {self.access_token}'
        })
        
        print(f"Successfully authenticated as {username}")
        return tokens
    
    def refresh_token(self, refresh_token):
        """Refresh the JWT token."""
        url = f"{self.base_url}/api/auth/token/refresh/"
        data = {'refresh': refresh_token}
        
        response = self.session.post(url, json=data)
        response.raise_for_status()
        
        tokens = response.json()
        self.access_token = tokens['access']
        
        self.session.headers.update({
            'Authorization': f'Bearer {self.access_token}'
        })
        
        return tokens
    
    def get_transactions(self, status=None, limit=50):
        """Get list of bank transactions."""
        url = f"{self.base_url}/api/v1/bank/transactions/"
        params = {'limit': limit}
        
        if status:
            params['status'] = status
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_unmatched_transactions(self):
        """Get unmatched transactions."""
        url = f"{self.base_url}/api/v1/bank/unmatched/"
        
        response = self.session.get(url)
        response.raise_for_status()
        
        return response.json()
    
    def upload_bank_statement(self, file_path):
        """Upload a bank statement file."""
        url = f"{self.base_url}/api/v1/bank/upload/"
        
        with open(file_path, 'rb') as file:
            files = {'file': file}
            response = self.session.post(url, files=files)
        
        response.raise_for_status()
        return response.json()
    
    def reconcile_transaction(self, transaction_id, invoice_ids, amounts, notes=None):
        """Manually reconcile a transaction with invoices."""
        url = f"{self.base_url}/api/v1/bank/reconcile/{transaction_id}/"
        
        data = {
            'transaction_id': transaction_id,
            'invoice_ids': invoice_ids,
            'amounts': amounts
        }
        
        if notes:
            data['notes'] = notes
        
        response = self.session.post(url, json=data)
        response.raise_for_status()
        
        return response.json()
    
    def get_reconciliation_logs(self, limit=50):
        """Get reconciliation logs."""
        url = f"{self.base_url}/api/v1/bank/logs/"
        params = {'limit': limit}
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_invoices(self, limit=50):
        """Get invoices."""
        url = f"{self.base_url}/api/v1/core/invoices/"
        params = {'limit': limit}
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        return response.json()


def create_sample_csv(filename='sample_bank_statement.csv'):
    """Create a sample CSV file for testing."""
    sample_data = [
        ['date', 'description', 'amount', 'reference', 'type'],
        ['2024-01-15', 'Payment from Customer ABC Ltd', '1500.00', 'INV001', 'credit'],
        ['2024-01-16', 'Payment from XYZ Corp', '2750.50', 'INV002', 'credit'],
        ['2024-01-17', 'Bank charges', '-25.00', 'FEE001', 'debit'],
        ['2024-01-18', 'Customer refund', '-150.00', 'REF001', 'debit'],
        ['2024-01-19', 'Online payment Customer DEF', '890.75', 'INV003', 'credit'],
    ]
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sample_data)
    
    print(f"Created sample CSV file: {filename}")
    return filename


def demo_api_usage():
    """Demonstrate API usage."""
    print("Bank Reconciliation System - API Demo")
    print("=" * 40)
    
    # Initialize client
    client = ReconciliationAPIClient()
    
    # Authenticate (you'll need to replace with actual credentials)
    try:
        client.authenticate('admin1', 'password123')
    except requests.exceptions.RequestException as e:
        print(f"Authentication failed: {e}")
        print("Please make sure:")
        print("1. The server is running (python manage.py runserver)")
        print("2. You have created sample data (python manage.py create_sample_data)")
        return
    
    # Get transactions
    print("\n1. Getting transactions...")
    transactions = client.get_transactions(limit=10)
    print(f"Found {transactions['count']} transactions")
    
    if transactions['results']:
        print(f"First transaction: {transactions['results'][0]['description']}")
    
    # Get unmatched transactions
    print("\n2. Getting unmatched transactions...")
    unmatched = client.get_unmatched_transactions()
    print(f"Found {len(unmatched['results'])} unmatched transactions")
    
    # Get invoices
    print("\n3. Getting invoices...")
    invoices = client.get_invoices(limit=10)
    print(f"Found {invoices['count']} invoices")
    
    # Demo file upload
    print("\n4. Demonstrating file upload...")
    sample_file = create_sample_csv()
    
    try:
        upload_result = client.upload_bank_statement(sample_file)
        print(f"Upload successful: {upload_result}")
    except requests.exceptions.RequestException as e:
        print(f"Upload failed: {e}")
    
    # Demo manual reconciliation (if we have unmatched transactions and invoices)
    if unmatched['results'] and invoices['results']:
        print("\n5. Demonstrating manual reconciliation...")
        transaction = unmatched['results'][0]
        invoice = invoices['results'][0]
        
        try:
            reconcile_result = client.reconcile_transaction(
                transaction_id=transaction['id'],
                invoice_ids=[invoice['id']],
                amounts=[str(min(float(transaction['amount']), float(invoice['total_amount'])))],
                notes="API demo reconciliation"
            )
            print(f"Reconciliation successful: {reconcile_result}")
        except requests.exceptions.RequestException as e:
            print(f"Reconciliation failed: {e}")
    
    # Get reconciliation logs
    print("\n6. Getting reconciliation logs...")
    logs = client.get_reconciliation_logs(limit=5)
    print(f"Found {logs['count']} reconciliation logs")
    
    if logs['results']:
        latest_log = logs['results'][0]
        print(f"Latest reconciliation: {latest_log['matched_by']} match with confidence {latest_log['confidence_score']}")
    
    print("\nAPI demo completed!")


if __name__ == '__main__':
    demo_api_usage()
