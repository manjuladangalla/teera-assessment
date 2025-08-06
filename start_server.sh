#!/bin/bash

# Bank Reconciliation System - Development Server Startup Script
# Enhanced with Deep Learning capabilities

set -e  # Exit on any error

echo "🏦 Starting Bank Reconciliation Sprint_status "Django development server with ML capabilities starting..."
echo ""
echo "📍 Server Endpoints:"
echo "  🌐 Main Application: http://localhost:$SERVER_PORT"
echo "  📚 API Documentation: http://localhost:$SERVER_PORT/api/docs/"
echo "  🔧 Admin Interface: http://localhost:$SERVER_PORT/admin/"
echo ""
echo "🤖 ML-Powered API Endpoints:"
echo "  GET  /api/v1/bank/transactions/{id}/ml_suggestions/"
echo "  POST /api/v1/bank/transactions/trigger_ml_matching/"
echo "  GET  /api/v1/ml/performance/"
echo "  POST /api/v1/ml/retrain/"L..."
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if virtual environment exists
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "No virtual environment detected. Attempting to activate..."
    if [[ -d "venv" ]]; then
        source venv/bin/activate
        print_status "Virtual environment activated"
    else
        print_warning "No venv directory found. Consider creating one: python -m venv venv"
    fi
fi

echo ""
echo "� Step 1: Installing Dependencies"
echo "================================="

print_info "Installing all dependencies (including ML)..."
pip install -r requirements.txt

print_status "All dependencies installed"

echo ""
echo "🔧 Step 2: Checking System Capabilities"
echo "======================================"

# Check Python version
python_version=$(python --version 2>&1)
print_info "Python version: $python_version"

# Check if ML dependencies are available
python -c "
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

try:
    import torch
    import transformers
    import sklearn
    import pandas as pd
    import numpy as np
    print('✓ All ML dependencies available')
    print(f'✓ PyTorch version: {torch.__version__}')
    print(f'✓ Transformers version: {transformers.__version__}')
    print(f'✓ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✓ CUDA device: {torch.cuda.get_device_name()}')
    else:
        print('ℹ Using CPU for ML operations')
except ImportError as e:
    print(f'⚠ Some ML dependencies missing: {e}')
    print('  Basic functionality will work, ML features may be limited')
"

echo ""
echo "🗄️ Step 3: Database Setup"
echo "========================"

# Check if migrations are needed
print_info "Checking for database migrations..."
python manage.py makemigrations --check --dry-run > /dev/null 2>&1
if [ $? -ne 0 ]; then
    print_warning "New migrations detected. Running migrations..."
    python manage.py makemigrations
    python manage.py migrate
else
    print_status "Database is up to date"
fi

echo ""
echo "📊 Step 4: ML System Check"
echo "========================="

# Check if we have training data for ML
python -c "
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

try:
    from reconciliation.models import ReconciliationLog, BankTransaction, Invoice
    
    log_count = ReconciliationLog.objects.count()
    transaction_count = BankTransaction.objects.count()
    invoice_count = Invoice.objects.count()
    
    print(f'📈 Reconciliation logs: {log_count}')
    print(f'💳 Bank transactions: {transaction_count}')
    print(f'📄 Invoices: {invoice_count}')
    
    if log_count < 10:
        print('⚠ Limited training data - ML features may have reduced accuracy')
        print('  Import more reconciliation data for better ML performance')
    else:
        print(f'✓ Sufficient training data available ({log_count} samples)')
        
    # Check if ML model exists
    import os
    if os.path.exists('ml_models/reconciliation_model.pth'):
        print('✓ ML model found - AI suggestions will be available')
    else:
        print('ℹ No trained ML model found')
        print('  Run: python manage.py train_reconciliation_model')
        
except Exception as e:
    print(f'⚠ Could not check ML system: {e}')
"

echo ""
echo "📂 Step 5: Static Files"
echo "======================"
# Collect static files if needed
print_info "Collecting static files..."
python manage.py collectstatic --noinput > /dev/null 2>&1
print_status "Static files ready"

echo ""
echo "🚀 Step 6: Starting Development Server"
echo "===================================="

# Check if port 8000 is available first
DEFAULT_PORT=8000
if lsof -Pi :$DEFAULT_PORT -sTCP:LISTEN -t >/dev/null ; then
    print_warning "Port $DEFAULT_PORT is already in use!"
    print_info "Would you like to:"
    echo "  1) Kill existing processes on port $DEFAULT_PORT and continue"
    echo "  2) Use alternative port 8001"
    echo "  3) Exit and handle manually"
    read -r -p "Choose option (1/2/3): " choice
    
    case $choice in
        1)
            print_info "Killing processes on port $DEFAULT_PORT..."
            lsof -ti:$DEFAULT_PORT | xargs kill -9 2>/dev/null || true
            sleep 2
            print_status "Port $DEFAULT_PORT is now available"
            SERVER_PORT=$DEFAULT_PORT
            ;;
        2)
            SERVER_PORT=8001
            print_info "Using alternative port $SERVER_PORT"
            ;;
        3)
            print_info "Exiting. You can manually kill processes with: lsof -ti:8000 | xargs kill -9"
            exit 0
            ;;
        *)
            print_info "Invalid choice. Using port 8001"
            SERVER_PORT=8001
            ;;
    esac
else
    SERVER_PORT=$DEFAULT_PORT
fi

print_status "Django development server with ML capabilities starting..."
echo ""
echo "📍 Server Endpoints:"
echo "  🌐 Main Application: http://localhost:8000"
echo "  � API Documentation: http://localhost:8000/api/docs/"
echo "  🔧 Admin Interface: http://localhost:8000/admin/"
echo ""
echo "🤖 ML-Powered API Endpoints:"
echo "  GET  /api/v1/bank/transactions/{id}/ml_suggestions/"
echo "  POST /api/v1/bank/transactions/trigger_ml_matching/"
echo "  GET  /api/v1/ml/performance/"
echo "  POST /api/v1/ml/retrain/"
echo ""
echo "🧪 Quick ML Test:"
echo "  curl -X GET 'http://localhost:$SERVER_PORT/api/v1/bank/transactions/1/ml_suggestions/?top_k=5' \\"
echo "       -H 'Authorization: Bearer YOUR_TOKEN'"
echo ""
echo "💡 ML Management Commands:"
echo "  python manage.py train_reconciliation_model    # Train ML model"
echo "  python manage.py predict_matches               # Test predictions"
echo ""
print_info "Press Ctrl+C to stop the server"
echo "=============================================="

print_status "Starting server on port $SERVER_PORT..."

# Run the server
python manage.py runserver 0.0.0.0:$SERVER_PORT
