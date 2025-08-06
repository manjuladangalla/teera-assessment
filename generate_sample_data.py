
import pandas as pd
from datetime import datetime, timedelta
import random
import uuid
from decimal import Decimal
import os

def generate_sample_data():

    os.makedirs('/Users/mdangallage/teera-assessment/sample_data', exist_ok=True)

    customers = [
        {
            'name': 'Acme Corporation',
            'email': 'billing@acme.com',
            'phone': '+1-555-0101',
            'address': '123 Business Ave, New York, NY 10001',
            'customer_code': 'ACME001',
            'typical_amount_range': (3000, 15000),
            'payment_delay_days': 25
        },
        {
            'name': 'Tech Solutions Ltd',
            'email': 'accounts@techsol.com',
            'phone': '+1-555-0102',
            'address': '456 Technology Blvd, San Francisco, CA 94105',
            'customer_code': 'TECH001',
            'typical_amount_range': (2000, 8000),
            'payment_delay_days': 15
        },
        {
            'name': 'Global Enterprises',
            'email': 'finance@global.com',
            'phone': '+1-555-0103',
            'address': '789 Corporate St, Chicago, IL 60601',
            'customer_code': 'GLOB001',
            'typical_amount_range': (5000, 25000),
            'payment_delay_days': 30
        },
        {
            'name': 'Digital Marketing Co',
            'email': 'pay@digmark.com',
            'phone': '+1-555-0104',
            'address': '321 Digital Way, Austin, TX 78701',
            'customer_code': 'DIGI001',
            'typical_amount_range': (1500, 5000),
            'payment_delay_days': 20
        },
        {
            'name': 'Manufacturing Inc',
            'email': 'ap@mfginc.com',
            'phone': '+1-555-0105',
            'address': '654 Industrial Rd, Detroit, MI 48201',
            'customer_code': 'MFG001',
            'typical_amount_range': (8000, 30000),
            'payment_delay_days': 35
        },
        {
            'name': 'StartUp Ventures',
            'email': 'billing@startup.com',
            'phone': '+1-555-0106',
            'address': '987 Innovation Dr, Seattle, WA 98101',
            'customer_code': 'START001',
            'typical_amount_range': (1000, 4000),
            'payment_delay_days': 10
        },
        {
            'name': 'Enterprise Solutions',
            'email': 'accounts@entsol.com',
            'phone': '+1-555-0107',
            'address': '147 Enterprise Plaza, Boston, MA 02101',
            'customer_code': 'ENT001',
            'typical_amount_range': (4000, 12000),
            'payment_delay_days': 25
        },
        {
            'name': 'Retail Chain Ltd',
            'email': 'finance@retail.com',
            'phone': '+1-555-0108',
            'address': '258 Retail Blvd, Miami, FL 33101',
            'customer_code': 'RET001',
            'typical_amount_range': (3000, 8000),
            'payment_delay_days': 20
        }
    ]

    services = [
        'Software License Renewal',
        'Web Development Services',
        'Annual Support Contract',
        'SEO Optimization Package',
        'ERP System Implementation',
        'Mobile App Development',
        'Cloud Migration Services',
        'POS System Integration',
        'Patient Management System',
        'Risk Management Platform',
        'Learning Management System',
        'Project Management Software',
        'Fleet Management System',
        'Video Editing Software License',
        'Business Intelligence Dashboard',
        'Data Analytics Platform',
        'CRM Integration Services',
        'Security Audit Services',
        'Database Optimization',
        'API Development Services'
    ]

    invoices = []
    customers_data = []
    bank_transactions = []
    start_date = datetime(2025, 6, 1)
    current_balance = 100000.00

    invoice_counter = 1

    for customer in customers:
        customers_data.append({
            'customer_code': customer['customer_code'],
            'name': customer['name'],
            'email': customer['email'],
            'phone': customer['phone'],
            'address': customer['address'],
            'is_active': True
        })

    for week in range(12):
        week_start = start_date + timedelta(weeks=week)

        for _ in range(random.randint(3, 8)):
            customer = random.choice(customers)
            service = random.choice(services)

            invoice_date = week_start + timedelta(days=random.randint(0, 6))
            due_date = invoice_date + timedelta(days=30)

            base_amount = random.uniform(*customer['typical_amount_range'])
            base_amount = round(base_amount, 2)
            tax_rate = 0.15
            tax_amount = round(base_amount * tax_rate, 2)
            total_amount = base_amount + tax_amount

            invoice_number = f"INV-2025-{invoice_counter:03d}"
            reference_number = f"REF-{customer['customer_code']}-{invoice_counter:03d}"

            status = 'paid' if random.random() < 0.85 else 'sent'

            invoice = {
                'invoice_number': invoice_number,
                'customer_code': customer['customer_code'],
                'customer_name': customer['name'],
                'customer_email': customer['email'],
                'amount_due': base_amount,
                'tax_amount': tax_amount,
                'total_amount': total_amount,
                'issue_date': invoice_date.strftime('%Y-%m-%d'),
                'due_date': due_date.strftime('%Y-%m-%d'),
                'status': status,
                'description': f"{service} - {customer['name']}",
                'reference_number': reference_number
            }
            invoices.append(invoice)

            if status == 'paid':
                payment_date = invoice_date + timedelta(days=customer['payment_delay_days'])
                payment_date += timedelta(days=random.randint(-5, 10))

                current_balance += total_amount

                description_variants = [
                    f"Payment from {customer['name']} - {service[:20]}",
                    f"Transfer from {customer['name']}",
                    f"{customer['name']} - Invoice Payment",
                    f"Wire from {customer['name'][:15]} - {reference_number}",
                    f"ACH Credit - {customer['name']}"
                ]

                transaction = {
                    'transaction_date': payment_date.strftime('%Y-%m-%d'),
                    'description': random.choice(description_variants),
                    'amount': total_amount,
                    'reference_number': reference_number if random.random() < 0.8 else '',
                    'bank_reference': f"BK-{payment_date.strftime('%Y-%m%d')}-{invoice_counter:03d}",
                    'transaction_type': 'credit',
                    'balance': round(current_balance, 2)
                }
                bank_transactions.append(transaction)

            invoice_counter += 1

    additional_transactions = [
        {
            'transaction_date': '2025-07-01',
            'description': 'Monthly Service Fee',
            'amount': -25.00,
            'reference_number': 'BANK-FEE-001',
            'bank_reference': 'BK-2025-0701-FEE',
            'transaction_type': 'debit',
            'balance': current_balance - 25.00
        },
        {
            'transaction_date': '2025-08-01',
            'description': 'Monthly Service Fee',
            'amount': -25.00,
            'reference_number': 'BANK-FEE-002',
            'bank_reference': 'BK-2025-0801-FEE',
            'transaction_type': 'debit',
            'balance': current_balance - 50.00
        },
        {
            'transaction_date': '2025-07-15',
            'description': 'Interest Payment',
            'amount': 125.50,
            'reference_number': 'INT-PAYMENT-001',
            'bank_reference': 'BK-2025-0715-INT',
            'transaction_type': 'credit',
            'balance': current_balance + 75.50
        },
        {
            'transaction_date': '2025-08-05',
            'description': 'Unknown Deposit - Investigation Required',
            'amount': 750.00,
            'reference_number': 'UNKNOWN-001',
            'bank_reference': 'BK-2025-0805-UNK',
            'transaction_type': 'credit',
            'balance': current_balance + 825.50
        },
        {
            'transaction_date': '2025-08-10',
            'description': 'Wire Transfer Fee',
            'amount': -35.00,
            'reference_number': 'WIRE-FEE-001',
            'bank_reference': 'BK-2025-0810-WIRE',
            'transaction_type': 'debit',
            'balance': current_balance + 790.50
        }
    ]

    bank_transactions.extend(additional_transactions)

    bank_transactions.sort(key=lambda x: x['transaction_date'])

    return invoices, bank_transactions, customers_data

def create_excel_file():
    print("Generating sample invoice and bank transaction data...")

    invoices, bank_transactions, customers_data = generate_sample_data()

    with pd.ExcelWriter('/Users/mdangallage/teera-assessment/sample_data/sample_reconciliation_data.xlsx', 
                       engine='openpyxl') as writer:

        customers_df = pd.DataFrame(customers_data)
        customers_df.to_excel(writer, sheet_name='Customers', index=False)

        invoices_df = pd.DataFrame(invoices)
        invoices_df.to_excel(writer, sheet_name='Invoices', index=False)

        transactions_df = pd.DataFrame(bank_transactions)
        transactions_df.to_excel(writer, sheet_name='Bank_Transactions', index=False)

        summary_data = {
            'Metric': [
                'Total Customers',
                'Total Invoices',
                'Total Paid Invoices',
                'Total Sent Invoices',
                'Total Invoice Amount',
                'Total Bank Transactions',
                'Total Credits',
                'Total Debits',
                'Expected Matches'
            ],
            'Value': [
                len(customers_data),
                len(invoices),
                len([i for i in invoices if i['status'] == 'paid']),
                len([i for i in invoices if i['status'] == 'sent']),
                f"${sum(i['total_amount'] for i in invoices):,.2f}",
                len(bank_transactions),
                len([t for t in bank_transactions if t['transaction_type'] == 'credit']),
                len([t for t in bank_transactions if t['transaction_type'] == 'debit']),
                len([i for i in invoices if i['status'] == 'paid'])
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"✅ Created Excel file with {len(customers_data)} customers, {len(invoices)} invoices and {len(bank_transactions)} bank transactions")
    print(f"📊 File saved: /Users/mdangallage/teera-assessment/sample_data/sample_reconciliation_data.xlsx")

    customers_df.to_csv('/Users/mdangallage/teera-assessment/sample_data/sample_customers.csv', index=False)
    invoices_df.to_csv('/Users/mdangallage/teera-assessment/sample_data/sample_invoices.csv', index=False)
    transactions_df.to_csv('/Users/mdangallage/teera-assessment/sample_data/sample_bank_transactions.csv', index=False)

    print("✅ Also created Django-compatible CSV files:")
    print("   - sample_customers.csv (for Customer model)")
    print("   - sample_invoices.csv (for Invoice model)")
    print("   - sample_bank_transactions.csv (for BankTransaction model)")

    file_upload_data = {
        'filename': 'sample_bank_transactions.csv',
        'original_filename': 'sample_bank_transactions.csv',
        'file_size': os.path.getsize('/Users/mdangallage/teera-assessment/sample_data/sample_bank_transactions.csv'),
        'status': 'uploaded',
        'total_records': len(bank_transactions),
        'processed_records': 0,
        'failed_records': 0
    }

    file_upload_df = pd.DataFrame([file_upload_data])
    file_upload_df.to_csv('/Users/mdangallage/teera-assessment/sample_data/sample_file_upload_status.csv', index=False)

    print("   - sample_file_upload_status.csv (for FileUploadStatus model)")

    return len(customers_data), len(invoices), len(bank_transactions)

if __name__ == "__main__":
    create_excel_file()
