from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from core.models import Company, UserProfile, Customer, Invoice
from reconciliation.models import BankTransaction, ReconciliationLog
from decimal import Decimal
from datetime import date, timedelta
import random


class Command(BaseCommand):
    help = 'Create sample data for testing the reconciliation system'

    def add_arguments(self, parser):
        parser.add_argument(
            '--companies',
            type=int,
            default=2,
            help='Number of companies to create'
        )
        parser.add_argument(
            '--transactions',
            type=int,
            default=100,
            help='Number of transactions per company'
        )

    def handle(self, *args, **options):
        self.stdout.write('Creating sample data...')
        
        # Create companies
        companies = []
        for i in range(options['companies']):
            company = Company.objects.create(
                name=f"Test Company {i+1}",
                industry='tech',
                contact_email=f"contact{i+1}@testcompany.com"
            )
            companies.append(company)
            self.stdout.write(f"Created company: {company.name}")
            
            # Create admin user for company
            user = User.objects.create_user(
                username=f"admin{i+1}",
                email=f"admin{i+1}@testcompany.com",
                password="password123",
                first_name=f"Admin",
                last_name=f"User {i+1}"
            )
            
            UserProfile.objects.create(
                user=user,
                company=company,
                is_admin=True
            )
            
            # Create customers
            customers = []
            for j in range(10):
                customer = Customer.objects.create(
                    company=company,
                    name=f"Customer {j+1}",
                    email=f"customer{j+1}@company{i+1}.com",
                    customer_code=f"CUST{i+1}{j+1:03d}"
                )
                customers.append(customer)
            
            # Create invoices
            invoices = []
            for k in range(50):
                customer = random.choice(customers)
                invoice = Invoice.objects.create(
                    customer=customer,
                    invoice_number=f"INV{i+1}{k+1:04d}",
                    amount_due=Decimal(random.uniform(100, 5000)),
                    tax_amount=Decimal(0),
                    total_amount=Decimal(random.uniform(100, 5000)),
                    issue_date=date.today() - timedelta(days=random.randint(1, 90)),
                    due_date=date.today() + timedelta(days=random.randint(1, 30)),
                    status=random.choice(['sent', 'overdue']),
                    description=f"Invoice for services {k+1}",
                    reference_number=f"REF{i+1}{k+1:04d}"
                )
                invoices.append(invoice)
            
            # Create bank transactions
            from reconciliation.models import FileUploadStatus
            file_upload = FileUploadStatus.objects.create(
                filename="sample_data.csv",
                original_filename="sample_data.csv",
                file_path="/tmp/sample_data.csv",
                file_size=1024,
                user=user,
                company=company,
                status='completed',
                total_records=options['transactions'],
                processed_records=options['transactions']
            )
            
            transactions = []
            for m in range(options['transactions']):
                transaction = BankTransaction.objects.create(
                    company=company,
                    transaction_date=date.today() - timedelta(days=random.randint(1, 60)),
                    description=f"Payment from Customer {random.randint(1, 10)}",
                    amount=Decimal(random.uniform(100, 5000)),
                    reference_number=f"TXN{i+1}{m+1:04d}",
                    bank_reference=f"BANK{random.randint(10000, 99999)}",
                    transaction_type="credit",
                    status=random.choice(['matched', 'unmatched', 'unmatched', 'unmatched']),  # More unmatched
                    file_upload=file_upload
                )
                transactions.append(transaction)
            
            # Create some reconciliation logs for matched transactions
            matched_transactions = [t for t in transactions if t.status == 'matched']
            for transaction in matched_transactions:
                # Find a suitable invoice to match
                suitable_invoices = [
                    inv for inv in invoices 
                    if abs(float(inv.total_amount) - float(transaction.amount)) < 100
                ]
                
                if suitable_invoices:
                    invoice = random.choice(suitable_invoices)
                    ReconciliationLog.objects.create(
                        transaction=transaction,
                        invoice=invoice,
                        matched_by=random.choice(['manual', 'ml_auto']),
                        confidence_score=random.uniform(0.7, 1.0),
                        amount_matched=min(transaction.amount, invoice.total_amount),
                        user=user,
                        metadata={'auto_generated': True}
                    )
            
            self.stdout.write(
                f"Created {len(customers)} customers, {len(invoices)} invoices, "
                f"and {len(transactions)} transactions for {company.name}"
            )
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully created sample data for {len(companies)} companies'
            )
        )
