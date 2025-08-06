"""
Django management command to load sample data compatible with the bank reconciliation models.
"""
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.db import transaction
from core.models import Company, Customer, Invoice, UserProfile
from reconciliation.models import BankTransaction, FileUploadStatus
import pandas as pd
import os
from datetime import datetime
import uuid

class Command(BaseCommand):
    help = 'Load sample data for testing bank reconciliation system'

    def add_arguments(self, parser):
        parser.add_argument(
            '--company-name',
            type=str,
            default='Test Company Ltd',
            help='Name of the company to create/use for sample data'
        )
        parser.add_argument(
            '--user-email',
            type=str,
            default='admin@testcompany.com',
            help='Email of the user to associate with the data'
        )

    def handle(self, *args, **options):
        self.stdout.write('Loading sample data for bank reconciliation...')
        
        # Get or create company
        company, created = Company.objects.get_or_create(
            name=options['company_name'],
            defaults={
                'industry': 'tech',
                'contact_email': options['user_email'],
                'is_active': True
            }
        )
        
        if created:
            self.stdout.write(f'✅ Created company: {company.name}')
        else:
            self.stdout.write(f'📊 Using existing company: {company.name}')
        
        # Get or create user and profile
        try:
            user = User.objects.get(email=options['user_email'])
            self.stdout.write(f'📊 Using existing user: {user.email}')
        except User.DoesNotExist:
            # Generate unique username if needed
            username_base = options['user_email'].split('@')[0]
            username = username_base
            counter = 1
            while User.objects.filter(username=username).exists():
                username = f"{username_base}{counter}"
                counter += 1
            
            user = User.objects.create_user(
                username=username,
                email=options['user_email'],
                password='admin123',
                first_name='Test',
                last_name='User'
            )
            self.stdout.write(f'✅ Created user: {user.email} (username: {username})')
        
        # Get or create user profile
        profile, created = UserProfile.objects.get_or_create(
            user=user,
            defaults={
                'company': company,
                'employee_id': 'EMP001',
                'department': 'Finance',
                'is_admin': True
            }
        )
        
        if created:
            self.stdout.write(f'✅ Created user profile for: {user.email}')
        else:
            # Update company if different
            if profile.company != company:
                profile.company = company
                profile.save()
                self.stdout.write(f'📊 Updated user profile company to: {company.name}')
        
        # Load sample data files
        data_dir = '/Users/mdangallage/teera-assessment/sample_data'
        
        if not os.path.exists(data_dir):
            self.stdout.write(self.style.ERROR('Sample data directory not found. Run generate_sample_data.py first.'))
            return
        
        try:
            with transaction.atomic():
                # Load customers
                self.load_customers(company, data_dir)
                
                # Load invoices
                self.load_invoices(company, data_dir)
                
                # Load bank transactions
                self.load_bank_transactions(company, user, data_dir)
                
            self.stdout.write(self.style.SUCCESS('✅ Successfully loaded all sample data!'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error loading data: {str(e)}'))
            raise

    def load_customers(self, company, data_dir):
        """Load customer data."""
        customers_file = os.path.join(data_dir, 'sample_customers.csv')
        if not os.path.exists(customers_file):
            self.stdout.write('⚠️  Customer data file not found, skipping...')
            return
            
        df = pd.read_csv(customers_file)
        customers_created = 0
        
        for _, row in df.iterrows():
            customer, created = Customer.objects.get_or_create(
                company=company,
                customer_code=row['customer_code'],
                defaults={
                    'name': row['name'],
                    'email': row['email'],
                    'phone': row['phone'],
                    'address': row['address'],
                    'is_active': row['is_active']
                }
            )
            if created:
                customers_created += 1
        
        self.stdout.write(f'📋 Loaded {customers_created} customers')

    def load_invoices(self, company, data_dir):
        """Load invoice data."""
        invoices_file = os.path.join(data_dir, 'sample_invoices.csv')
        if not os.path.exists(invoices_file):
            self.stdout.write('⚠️  Invoice data file not found, skipping...')
            return
            
        df = pd.read_csv(invoices_file)
        invoices_created = 0
        
        for _, row in df.iterrows():
            # Find the customer
            try:
                customer = Customer.objects.get(
                    company=company,
                    customer_code=row['customer_code']
                )
            except Customer.DoesNotExist:
                self.stdout.write(f'⚠️  Customer {row["customer_code"]} not found, skipping invoice {row["invoice_number"]}')
                continue
            
            invoice, created = Invoice.objects.get_or_create(
                customer=customer,
                invoice_number=row['invoice_number'],
                defaults={
                    'amount_due': row['amount_due'],
                    'tax_amount': row['tax_amount'],
                    'total_amount': row['total_amount'],
                    'issue_date': datetime.strptime(row['issue_date'], '%Y-%m-%d').date(),
                    'due_date': datetime.strptime(row['due_date'], '%Y-%m-%d').date(),
                    'status': row['status'],
                    'description': row['description'],
                    'reference_number': row['reference_number']
                }
            )
            if created:
                invoices_created += 1
        
        self.stdout.write(f'🧾 Loaded {invoices_created} invoices')

    def load_bank_transactions(self, company, user, data_dir):
        """Load bank transaction data."""
        transactions_file = os.path.join(data_dir, 'sample_bank_transactions.csv')
        if not os.path.exists(transactions_file):
            self.stdout.write('⚠️  Bank transactions data file not found, skipping...')
            return
            
        # Create a file upload status record
        file_upload = FileUploadStatus.objects.create(
            filename='sample_bank_transactions.csv',
            original_filename='sample_bank_transactions.csv',
            file_path=transactions_file,
            file_size=os.path.getsize(transactions_file),
            user=user,
            company=company,
            status='completed'
        )
        
        df = pd.read_csv(transactions_file)
        transactions_created = 0
        
        for _, row in df.iterrows():
            # Clean the raw data for JSON serialization
            raw_data = {}
            for key, value in row.items():
                if pd.notna(value):  # Only include non-null values
                    if isinstance(value, (int, float, str, bool)):
                        raw_data[key] = value
                    else:
                        raw_data[key] = str(value)
            
            transaction, created = BankTransaction.objects.get_or_create(
                company=company,
                bank_reference=row['bank_reference'],
                defaults={
                    'transaction_date': datetime.strptime(row['transaction_date'], '%Y-%m-%d').date(),
                    'description': row['description'],
                    'amount': row['amount'],
                    'reference_number': row.get('reference_number', '') if pd.notna(row.get('reference_number')) else '',
                    'transaction_type': row['transaction_type'],
                    'balance': row.get('balance', 0) if pd.notna(row.get('balance')) else 0,
                    'status': 'unmatched',  # Default status
                    'file_upload': file_upload,
                    'raw_data': raw_data
                }
            )
            if created:
                transactions_created += 1
        
        # Update file upload status
        file_upload.total_records = len(df)
        file_upload.processed_records = transactions_created
        file_upload.save()
        
        self.stdout.write(f'💰 Loaded {transactions_created} bank transactions')
        self.stdout.write(f'📁 Created file upload record: {file_upload.original_filename}')
