from django.contrib.auth.models import User
from core.models import Company, Customer, Invoice, UserProfile
from reconciliation.models import BankTransaction
from decimal import Decimal
from datetime import datetime, timedelta

# Get admin user
admin_user = User.objects.get(username='admin')

# Create a demo company if it doesn't exist
company, created = Company.objects.get_or_create(
    name='Demo Company Ltd',
    defaults={
        'industry': 'tech',
        'contact_email': 'demo@company.com',
        'registration_number': 'DEMO123',
        'is_active': True
    }
)

# Create user profile if it doesn't exist
profile, created = UserProfile.objects.get_or_create(
    user=admin_user,
    defaults={
        'company': company,
        'employee_id': 'EMP001',
        'department': 'Administration',
        'is_admin': True
    }
)

print(f'Company: {company.name}')
print(f'User Profile: {profile.user.username}')

# Create sample customers
customers = []
customer_data = [
    {'name': 'ABC Corp', 'email': 'billing@abccorp.com', 'code': 'ABC001'},
    {'name': 'XYZ Industries', 'email': 'accounts@xyz.com', 'code': 'XYZ001'},
    {'name': 'Tech Solutions Inc', 'email': 'payment@techsol.com', 'code': 'TECH001'},
]

for data in customer_data:
    customer, created = Customer.objects.get_or_create(
        company=company,
        customer_code=data['code'],
        defaults={
            'name': data['name'],
            'email': data['email'],
            'is_active': True
        }
    )
    customers.append(customer)
    print(f'Customer: {customer.name}')

# Create sample invoices
today = datetime.now().date()
for i, customer in enumerate(customers):
    for j in range(3):
        invoice_num = f'INV-{customer.customer_code}-{j+1:03d}'
        amount = Decimal((i+1) * 1000 + (j+1) * 100)
        
        invoice, created = Invoice.objects.get_or_create(
            customer=customer,
            invoice_number=invoice_num,
            defaults={
                'amount_due': amount,
                'tax_amount': amount * Decimal('0.1'),
                'total_amount': amount * Decimal('1.1'),
                'issue_date': today - timedelta(days=30-j*5),
                'due_date': today - timedelta(days=j*5),
                'status': 'sent',
                'description': f'Services for {customer.name} - Period {j+1}',
                'reference_number': f'REF-{invoice_num}'
            }
        )
        print(f'Invoice: {invoice.invoice_number} - ${invoice.total_amount}')

# Create sample bank transactions
transactions_data = [
    {'amount': 1100.00, 'desc': 'Wire Transfer from ABC Corp', 'ref': 'INV-ABC001-001'},
    {'amount': 1210.00, 'desc': 'ACH Payment XYZ Industries', 'ref': 'INV-XYZ001-001'},
    {'amount': 1320.00, 'desc': 'Online Payment Tech Solutions', 'ref': 'INV-TECH001-001'},
    {'amount': 1200.00, 'desc': 'Wire Payment ABC Corp', 'ref': 'INV-ABC001-002'},
    {'amount': 2200.00, 'desc': 'Bank Transfer XYZ Industries', 'ref': 'INV-XYZ001-002'},
    {'amount': 1430.00, 'desc': 'Direct Deposit Tech Solutions', 'ref': 'INV-TECH001-002'},
]

# Create a sample file upload record first
from reconciliation.models import FileUploadStatus
file_upload, created = FileUploadStatus.objects.get_or_create(
    filename='sample_bank_statement.csv',
    defaults={
        'original_filename': 'sample_bank_statement.csv',
        'file_path': '/uploads/sample_bank_statement.csv',
        'file_size': 1024,
        'user': admin_user,
        'company': company,
        'status': 'completed',
        'total_records': len(transactions_data),
        'processed_records': len(transactions_data),
        'failed_records': 0
    }
)
print(f'File Upload: {file_upload.filename}')

for i, trans_data in enumerate(transactions_data):
    bank_ref = f'BANK{datetime.now().strftime("%Y%m%d")}{i+1:04d}'
    transaction, created = BankTransaction.objects.get_or_create(
        company=company,
        bank_reference=bank_ref,
        defaults={
            'amount': Decimal(str(trans_data['amount'])),
            'description': trans_data['desc'],
            'reference_number': trans_data['ref'],
            'transaction_date': today - timedelta(days=i),
            'status': 'unmatched',
            'transaction_type': 'credit',
            'file_upload': file_upload
        }
    )
    print(f'Transaction: {transaction.bank_reference} - ${transaction.amount}')

print('\n✅ Sample data created successfully!')
print('You can now test the reconciliation system.')
