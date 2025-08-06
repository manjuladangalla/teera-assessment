from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from core.models import Company
from reconciliation.models import BankTransaction, FileUploadStatus
from decimal import Decimal
from datetime import date, timedelta
import random

class Command(BaseCommand):
    help = 'Add more unmatched transactions for testing'

    def add_arguments(self, parser):
        parser.add_argument(
            '--count',
            type=int,
            default=50,
            help='Number of unmatched transactions to create'
        )
        parser.add_argument(
            '--company',
            type=str,
            help='Company name (optional - if not provided, will use all companies)'
        )

    def handle(self, *args, **options):
        count = options['count']
        company_name = options.get('company')
        
        # Get companies
        if company_name:
            try:
                companies = [Company.objects.get(name__icontains=company_name)]
            except Company.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'Company containing "{company_name}" not found')
                )
                return
        else:
            companies = Company.objects.all()
            
        if not companies:
            self.stdout.write(self.style.ERROR('No companies found'))
            return

        self.stdout.write(f'Creating {count} unmatched transactions...')

        # Sample transaction descriptions and types
        transaction_descriptions = [
            "Online payment from customer",
            "Bank transfer received",
            "Check deposit",
            "Wire transfer payment",
            "Credit card payment",
            "ACH payment received",
            "Customer payment - invoice settlement",
            "Refund processed",
            "Direct debit payment",
            "Mobile payment received",
            "PayPal payment",
            "Stripe payment processing",
            "Square payment",
            "Venmo transfer",
            "Zelle payment received",
            "Cash deposit",
            "Money order payment",
            "Cashier's check deposit",
            "International wire transfer",
            "Recurring payment received",
            "Subscription payment",
            "Partial payment received",
            "Late payment with interest",
            "Early payment discount applied",
            "Settlement payment",
            "Insurance claim payment",
            "Dividend payment received",
            "Interest payment",
            "Loan payment received",
            "Rent payment",
            "Service fee payment",
            "Consulting fee payment",
            "Product sale payment",
            "License fee payment",
            "Maintenance fee received",
            "Support payment",
            "Training fee payment",
            "Commission payment received",
            "Bonus payment",
            "Salary payment received"
        ]

        transaction_types = ["credit", "debit"]
        
        total_created = 0

        for company in companies:
            # Get or create a file upload for this company
            user = User.objects.filter(profile__company=company).first()
            if not user:
                self.stdout.write(
                    self.style.WARNING(f'No user found for company {company.name}, skipping...')
                )
                continue

            file_upload, created = FileUploadStatus.objects.get_or_create(
                company=company,
                filename="manual_unmatched_transactions.csv",
                defaults={
                    'original_filename': "manual_unmatched_transactions.csv",
                    'file_path': "/tmp/manual_unmatched_transactions.csv",
                    'file_size': 2048,
                    'user': user,
                    'status': 'completed',
                    'total_records': count,
                    'processed_records': count
                }
            )

            # Create transactions for this company
            company_count = count if len(companies) == 1 else count // len(companies)
            
            created_transactions = []
            for i in range(company_count):
                # Generate realistic transaction data
                transaction_date = date.today() - timedelta(days=random.randint(1, 90))
                description = random.choice(transaction_descriptions)
                
                # Add some variation to descriptions
                customer_suffix = random.choice([
                    " - ACME Corp", " - Tech Solutions Inc", " - Global Services Ltd",
                    " - Innovation Partners", " - Digital Systems", " - Smart Solutions",
                    " - Future Tech", " - Prime Services", " - Elite Consulting",
                    " - Advanced Systems", " - NextGen Solutions", " - Pro Services"
                ])
                description += customer_suffix
                
                # Generate realistic amounts
                amount_ranges = [
                    (50, 500),      # Small payments
                    (500, 2000),    # Medium payments  
                    (2000, 10000),  # Large payments
                    (10000, 50000)  # Very large payments
                ]
                
                range_choice = random.choices(
                    amount_ranges,
                    weights=[40, 35, 20, 5],  # More small/medium payments
                    k=1
                )[0]
                
                amount = Decimal(random.uniform(range_choice[0], range_choice[1]))
                amount = amount.quantize(Decimal('0.01'))  # Round to 2 decimal places
                
                # Generate reference numbers
                ref_number = f"TXN{random.randint(100000, 999999)}"
                bank_ref = f"BANK{random.randint(1000000, 9999999)}"
                
                transaction = BankTransaction.objects.create(
                    company=company,
                    transaction_date=transaction_date,
                    description=description,
                    amount=amount,
                    reference_number=ref_number,
                    bank_reference=bank_ref,
                    transaction_type=random.choice(transaction_types),
                    status='unmatched',  # All transactions will be unmatched
                    file_upload=file_upload,
                    raw_data={
                        'generated_by': 'add_unmatched_transactions_command',
                        'batch_id': f'batch_{company.id}_{date.today().isoformat()}',
                        'created_at': date.today().isoformat()
                    }
                )
                created_transactions.append(transaction)
                total_created += 1

            self.stdout.write(
                f"Created {len(created_transactions)} unmatched transactions for {company.name}"
            )

        # Display summary
        self.stdout.write(
            self.style.SUCCESS(
                f'\nSuccessfully created {total_created} unmatched transactions!'
            )
        )
        
        # Show current statistics
        total_transactions = BankTransaction.objects.count()
        unmatched_transactions = BankTransaction.objects.filter(status='unmatched').count()
        
        self.stdout.write(f'\nCurrent database statistics:')
        self.stdout.write(f'  Total transactions: {total_transactions}')
        self.stdout.write(f'  Unmatched transactions: {unmatched_transactions}')
        self.stdout.write(f'  Match rate: {((total_transactions - unmatched_transactions) / total_transactions * 100):.1f}%')
