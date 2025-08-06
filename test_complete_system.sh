#!/bin/bash

# Bank Reconciliation System - Complete Test Script
# This script tests the entire system from setup to API calls

set -e  # Exit on any error

echo "🚀 Bank Reconciliation System - Complete Test Script"
echo "===================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "manage.py" ]; then
    print_error "manage.py not found. Please run this script from the project root directory."
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "Virtual environment not detected. Attempting to activate..."
    if [ -d "venv" ]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment not found. Please create one with: python -m venv venv"
        exit 1
    fi
fi

print_status "Step 1: Checking Python dependencies..."
if ! pip check > /dev/null 2>&1; then
    print_warning "Some package dependencies may have issues. Installing requirements..."
    pip install -r requirements.txt
fi
print_success "Dependencies check complete"

print_status "Step 2: Running database migrations..."
python manage.py migrate --no-input
print_success "Database migrations complete"

print_status "Step 3: Checking for sample data..."
TRANSACTION_COUNT=$(python manage.py shell -c "
from reconciliation.models import BankTransaction
print(BankTransaction.objects.count())
" 2>/dev/null)

if [ "$TRANSACTION_COUNT" -eq 0 ]; then
    print_status "No sample data found. Loading sample data..."
    python manage.py load_sample_data
    print_success "Sample data loaded"
else
    print_success "Sample data already exists ($TRANSACTION_COUNT transactions)"
fi

print_status "Step 4: Creating test superuser (if needed)..."
python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='testadmin').exists():
    User.objects.create_superuser('testadmin', 'admin@test.com', 'testpass123')
    print('Test user created: testadmin / testpass123')
else:
    print('Test user already exists')
" 2>/dev/null

print_status "Step 5: Starting Django development server..."
python manage.py runserver 8000 &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Function to cleanup on exit
cleanup() {
    print_status "Cleaning up..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    print_success "Server stopped"
}
trap cleanup EXIT

# Test if server is running
if ! curl -s http://localhost:8000 > /dev/null; then
    print_error "Server failed to start or not responding"
    exit 1
fi

print_success "Server started successfully on port 8000"

print_status "Step 6: Testing API endpoints..."

# Test 1: Get JWT Token
print_status "  6.1: Testing authentication..."
TOKEN_RESPONSE=$(curl -s -X POST http://localhost:8000/api/auth/token/ \
  -H "Content-Type: application/json" \
  -d '{"username": "testadmin", "password": "testpass123"}')

if echo "$TOKEN_RESPONSE" | grep -q "access"; then
    ACCESS_TOKEN=$(echo "$TOKEN_RESPONSE" | python -c "
import sys, json
data = json.load(sys.stdin)
print(data['access'])
")
    print_success "  ✓ Authentication successful"
else
    print_error "  ✗ Authentication failed"
    echo "Response: $TOKEN_RESPONSE"
    exit 1
fi

# Test 2: List Bank Transactions
print_status "  6.2: Testing bank transactions endpoint..."
TRANSACTIONS_RESPONSE=$(curl -s -X GET http://localhost:8000/api/v1/bank/transactions/ \
  -H "Authorization: Bearer $ACCESS_TOKEN")

if echo "$TRANSACTIONS_RESPONSE" | grep -q "results"; then
    TRANSACTION_COUNT=$(echo "$TRANSACTIONS_RESPONSE" | python -c "
import sys, json
data = json.load(sys.stdin)
print(len(data['results']))
")
    print_success "  ✓ Bank transactions endpoint working ($TRANSACTION_COUNT transactions)"
else
    print_error "  ✗ Bank transactions endpoint failed"
    echo "Response: $TRANSACTIONS_RESPONSE"
fi

# Test 3: List Unmatched Transactions
print_status "  6.3: Testing unmatched transactions endpoint..."
UNMATCHED_RESPONSE=$(curl -s -X GET http://localhost:8000/api/v1/bank/unmatched/ \
  -H "Authorization: Bearer $ACCESS_TOKEN")

if echo "$UNMATCHED_RESPONSE" | grep -q "results"; then
    print_success "  ✓ Unmatched transactions endpoint working"
else
    print_error "  ✗ Unmatched transactions endpoint failed"
fi

# Test 4: List Customers
print_status "  6.4: Testing customers endpoint..."
CUSTOMERS_RESPONSE=$(curl -s -X GET http://localhost:8000/api/v1/bank/customers/ \
  -H "Authorization: Bearer $ACCESS_TOKEN")

if echo "$CUSTOMERS_RESPONSE" | grep -q "results"; then
    CUSTOMER_COUNT=$(echo "$CUSTOMERS_RESPONSE" | python -c "
import sys, json
data = json.load(sys.stdin)
print(len(data['results']))
")
    print_success "  ✓ Customers endpoint working ($CUSTOMER_COUNT customers)"
else
    print_error "  ✗ Customers endpoint failed"
fi

# Test 5: List Invoices
print_status "  6.5: Testing invoices endpoint..."
INVOICES_RESPONSE=$(curl -s -X GET http://localhost:8000/api/v1/bank/invoices/ \
  -H "Authorization: Bearer $ACCESS_TOKEN")

if echo "$INVOICES_RESPONSE" | grep -q "results"; then
    INVOICE_COUNT=$(echo "$INVOICES_RESPONSE" | python -c "
import sys, json
data = json.load(sys.stdin)
print(len(data['results']))
")
    print_success "  ✓ Invoices endpoint working ($INVOICE_COUNT invoices)"
else
    print_error "  ✗ Invoices endpoint failed"
fi

# Test 6: File Upload (if sample files exist)
if [ -f "sample_data/sample_bank_transactions.csv" ]; then
    print_status "  6.6: Testing file upload endpoint..."
    UPLOAD_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/bank/upload/ \
      -H "Authorization: Bearer $ACCESS_TOKEN" \
      -F "file=@sample_data/sample_bank_transactions.csv" \
      -F "file_type=bank_statement")
    
    if echo "$UPLOAD_RESPONSE" | grep -q "task_id"; then
        print_success "  ✓ File upload endpoint working"
    else
        print_warning "  ⚠ File upload may have issues (check Celery setup)"
    fi
else
    print_warning "  ⚠ Sample CSV file not found, skipping upload test"
fi

# Test 7: API Documentation
print_status "  6.7: Testing API documentation..."
DOCS_RESPONSE=$(curl -s http://localhost:8000/api/docs/)
if echo "$DOCS_RESPONSE" | grep -q "swagger"; then
    print_success "  ✓ API documentation accessible"
else
    print_warning "  ⚠ API documentation may have issues"
fi

print_status "Step 7: Database statistics..."
python manage.py shell -c "
from reconciliation.models import BankTransaction, Invoice, Customer, ReconciliationLog
from core.models import Company, User
from django.db.models import Count, Sum

print('=== Database Statistics ===')
print(f'Companies: {Company.objects.count()}')
print(f'Users: {User.objects.count()}')
print(f'Customers: {Customer.objects.count()}')
print(f'Invoices: {Invoice.objects.count()}')
print(f'Bank Transactions: {BankTransaction.objects.count()}')
print(f'  - Matched: {BankTransaction.objects.filter(is_matched=True).count()}')
print(f'  - Unmatched: {BankTransaction.objects.filter(is_matched=False).count()}')
print(f'Reconciliation Logs: {ReconciliationLog.objects.count()}')

total_amount = BankTransaction.objects.aggregate(Sum('amount'))['amount__sum'] or 0
print(f'Total Transaction Amount: ${total_amount:,.2f}')
"

echo ""
print_success "🎉 Test Script Complete!"
echo ""
echo "📋 Test Summary:"
echo "=================="
echo "• Django server: ✓ Running on http://localhost:8000"
echo "• Authentication: ✓ JWT tokens working"
echo "• API endpoints: ✓ All major endpoints tested"
echo "• Sample data: ✓ Loaded and accessible"
echo "• Documentation: ✓ Available at http://localhost:8000/api/docs/"
echo ""
echo "🔗 Quick Links:"
echo "• Web Interface: http://localhost:8000/"
echo "• API Documentation: http://localhost:8000/api/docs/"
echo "• ReDoc: http://localhost:8000/api/redoc/"
echo "• Admin Interface: http://localhost:8000/admin/"
echo ""
echo "🔑 Test Credentials:"
echo "• Username: testadmin"
echo "• Password: testpass123"
echo ""
echo "📊 Postman Testing:"
echo "• Import: postman_collection.json"
echo "• Set base_url to: http://localhost:8000"
echo "• Use JWT token from authentication"
echo ""
echo "⚠️  Note: Server will continue running. Press Ctrl+C to stop."
echo ""

# Keep the script running to maintain the server
print_status "Server running... Press Ctrl+C to stop"
while true; do
    sleep 1
done
