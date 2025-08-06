#!/bin/bash

# Bank Reconciliation System Setup Script
# This script helps set up the development environment

set -e

echo "🏦 Bank Reconciliation System Setup"
echo "==================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "✅ Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created - please edit with your settings"
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p ml_models
mkdir -p media/uploads
mkdir -p staticfiles

# Database setup
echo "🗄️  Setting up database..."

# Check if we should use PostgreSQL or SQLite
if command -v psql &> /dev/null; then
    echo "PostgreSQL detected. Setting up PostgreSQL database..."
    
    # Check if database exists
    if psql -lqt | cut -d \| -f 1 | grep -qw teera_reconciliation; then
        echo "✅ Database 'teera_reconciliation' already exists"
    else
        echo "Creating PostgreSQL database..."
        createdb teera_reconciliation
        echo "✅ PostgreSQL database created"
    fi
    
    # Run migrations
    echo "🔄 Running migrations..."
    python manage.py migrate
else
    echo "PostgreSQL not found. Using SQLite for development..."
    echo "🔄 Running migrations..."
    python manage.py migrate
fi

# Create superuser if it doesn't exist
echo "👤 Creating superuser..."
python manage.py shell -c "
from django.contrib.auth.models import User
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
    print('Superuser created: admin/admin123')
else:
    print('Superuser already exists')
"

# Collect static files
echo "📦 Collecting static files..."
python manage.py collectstatic --noinput

# Create sample data
echo "📊 Creating sample data..."
python manage.py create_sample_data --companies 2 --transactions 50

# Check if Redis is running
if pgrep redis-server > /dev/null; then
    echo "✅ Redis is running"
else
    echo "⚠️  Redis is not running. Please start Redis for Celery tasks."
    echo "   On macOS: brew services start redis"
    echo "   On Ubuntu: sudo systemctl start redis-server"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start development server: python manage.py runserver"
echo "3. Visit: http://localhost:8000"
echo "4. Admin panel: http://localhost:8000/admin (admin/admin123)"
echo "5. API docs: http://localhost:8000/api/docs/"
echo ""
echo "Optional (for background tasks):"
echo "6. Start Celery worker: celery -A config worker -l info"
echo "7. Start Celery beat: celery -A config beat -l info"
echo ""
echo "Test the API:"
echo "8. Run API demo: python api_demo.py"
