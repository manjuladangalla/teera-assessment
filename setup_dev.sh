#!/bin/bash

# Development Setup Script for Bank Reconciliation System
# This script sets up a development environment using SQLite instead of PostgreSQL

set -e  # Exit on any error

echo "🚀 Setting up Bank Reconciliation System (Development Mode)"
echo "============================================================"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📋 Installing Python dependencies..."
pip install -r requirements_dev.txt

# Create environment file for development
echo "🔑 Creating development environment file..."
cat > .env << EOF
# Development Environment Configuration
DEBUG=True
SECRET_KEY=dev-secret-key-change-in-production
DATABASE_URL=sqlite:///db.sqlite3

# Optional: Redis (if you have it installed)
REDIS_URL=redis://localhost:6379/0

# ML Settings
ML_MODEL_PATH=ml_models/
UPLOAD_PATH=uploads/

# Email (for development - uses console backend)
EMAIL_BACKEND=django.core.mail.backends.console.EmailBackend
EOF

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads
mkdir -p ml_models
mkdir -p static
mkdir -p media

# Database setup
echo "🗄️  Setting up database..."
python manage.py makemigrations
python manage.py makemigrations core
python manage.py makemigrations reconciliation
python manage.py makemigrations ml_engine
python manage.py migrate

# Create superuser (non-interactive)
echo "👤 Creating superuser..."
python manage.py shell << EOF
from django.contrib.auth import get_user_model
from core.models import Company, UserProfile

User = get_user_model()

# Create superuser if doesn't exist
if not User.objects.filter(username='admin').exists():
    user = User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
    
    # Create a sample company
    company = Company.objects.create(
        name='Demo Company',
        code='DEMO',
        email='demo@company.com'
    )
    
    # Create user profile
    UserProfile.objects.create(
        user=user,
        company=company,
        phone='123-456-7890'
    )
    
    print('Superuser created: admin / admin123')
else:
    print('Superuser already exists')
EOF

# Load sample data
echo "📊 Loading sample data..."
python manage.py shell << EOF
from core.models import Company, Customer, Invoice
from reconciliation.models import BankTransaction
from django.contrib.auth import get_user_model
from decimal import Decimal
import uuid

User = get_user_model()
admin_user = User.objects.get(username='admin')
company = admin_user.userprofile.company

# Create sample customers
customers = []
for i in range(3):
    customer, created = Customer.objects.get_or_create(
        company=company,
        name=f'Customer {i+1}',
        defaults={
            'email': f'customer{i+1}@example.com',
            'account_number': f'ACC{i+1:03d}'
        }
    )
    customers.append(customer)

# Create sample invoices
invoices = []
for i, customer in enumerate(customers):
    for j in range(2):
        invoice, created = Invoice.objects.get_or_create(
            company=company,
            customer=customer,
            invoice_number=f'INV-{i+1}-{j+1:03d}',
            defaults={
                'amount': Decimal(f'{(i+1)*100 + (j+1)*10}.00'),
                'description': f'Invoice for services {i+1}-{j+1}'
            }
        )
        invoices.append(invoice)

# Create sample bank transactions
transactions = [
    {'amount': 110.00, 'description': 'Payment from Customer 1', 'reference': 'INV-1-001'},
    {'amount': 120.00, 'description': 'Payment from Customer 1', 'reference': 'INV-1-002'},
    {'amount': 210.00, 'description': 'Payment from Customer 2', 'reference': 'INV-2-001'},
    {'amount': 220.00, 'description': 'Payment from Customer 2', 'reference': 'INV-2-002'},
    {'amount': 310.00, 'description': 'Payment from Customer 3', 'reference': 'INV-3-001'},
]

for i, trans_data in enumerate(transactions):
    BankTransaction.objects.get_or_create(
        company=company,
        transaction_id=f'TXN{i+1:06d}',
        defaults={
            'amount': Decimal(str(trans_data['amount'])),
            'description': trans_data['description'],
            'reference': trans_data['reference'],
            'status': 'pending'
        }
    )

print('Sample data created successfully!')
EOF

# Collect static files
echo "🎨 Collecting static files..."
python manage.py collectstatic --noinput

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🎯 Quick Start Guide:"
echo "===================="
echo "1. Start the development server:"
echo "   python manage.py runserver"
echo ""
echo "2. Access the application:"
echo "   Web Interface: http://127.0.0.1:8000/"
echo "   Admin Panel:   http://127.0.0.1:8000/admin/"
echo "   API Docs:      http://127.0.0.1:8000/api/docs/"
echo ""
echo "3. Login credentials:"
echo "   Username: admin"
echo "   Password: admin123"
echo ""
echo "4. Optional - Start Celery worker (in another terminal):"
echo "   source venv/bin/activate"
echo "   celery -A config worker -l info"
echo ""
echo "📚 Next Steps:"
echo "- Upload bank statement files via the web interface"
echo "- Test the ML matching engine"
echo "- Explore the API endpoints"
echo "- Try manual reconciliation features"
echo ""
echo "Happy coding! 🚀"
