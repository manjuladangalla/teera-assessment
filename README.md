# Advanced Bank Reconciliation System

A sophisticated Django-based reconciliation system that uses machine learning to automatically match bank transactions with invoices, providing intelligent automation while maintaining full audit trails and manual override capabilities.

## 🚀 Features

### Core Functionality

- **Multi-tenant Architecture**: Complete data isolation between companies
- **File Upload & Processing**: Support for CSV and Excel bank statements
- **Deep Learning Matching**: TensorFlow/scikit-learn powered transaction matching
- **Manual Reconciliation**: Web interface for manual transaction matching
- **Audit & Logging**: Complete audit trail for all reconciliation activities
- **Asynchronous Processing**: Celery-based background task processing
- **RESTful API**: Comprehensive API with JWT authentication
- **Reporting**: PDF and Excel export capabilities

### Machine Learning Features

- **Feature Extraction**: Advanced text and numeric feature extraction
- **Model Training**: Automatic model retraining based on new data
- **Confidence Scoring**: ML confidence scores for match predictions
- **Performance Tracking**: Model accuracy and performance metrics
- **ONNX Support**: Model serving with ONNX runtime

### Security & Compliance

- **JWT Authentication**: Secure token-based authentication
- **Company Isolation**: Strict data separation between tenants
- **Audit Logging**: Complete activity tracking
- **Role-based Access**: Admin and user role management
- **IP Tracking**: Request origin tracking for security

## 🏗️ Architecture

### Backend Stack

- **Framework**: Django 4.x
- **Database**: PostgreSQL (SQLite for development)
- **API**: Django REST Framework
- **Authentication**: JWT with Simple JWT
- **Task Queue**: Celery with Redis
- **ML Framework**: scikit-learn, TensorFlow
- **File Processing**: pandas, openpyxl

### Frontend Options

- **Web Interface**: Django Templates + Bootstrap 5
- **API Documentation**: Swagger/OpenAPI with drf-spectacular
- **Admin Interface**: Django Admin (enhanced)

## ⚡ Quick Start Testing

### 5-Minute Setup

```bash
# 1. Clone and setup
git clone <repository-url>
cd teera-assessment
python -m venv venv
source venv/bin/activate

# 2. Install and migrate
pip install -r requirements.txt
python manage.py migrate

# 3. Create admin user
python manage.py createsuperuser

# 4. Load sample data
python manage.py load_sample_data

# 5. Start server
python manage.py runserver
```

### Quick API Test

```bash
# Get JWT token (replace with your credentials)
curl -X POST http://localhost:8000/api/auth/token/ \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your_password"}'

# Test with token (replace YOUR_TOKEN)
export TOKEN="your_access_token_here"
curl -X GET http://localhost:8000/api/v1/bank/transactions/ \
  -H "Authorization: Bearer $TOKEN"
```

### Quick Postman Test

1. **Import Collection**: Use the Postman collection below
2. **Set Environment**: `base_url = http://localhost:8000`
3. **Authenticate**: Run "Obtain JWT Token" request first
4. **Test Endpoints**: All other requests will use the token automatically

**Postman Environment Setup:**

```json
{
  "name": "Bank Reconciliation Local",
  "values": [
    { "key": "base_url", "value": "http://localhost:8000" },
    { "key": "access_token", "value": "" },
    { "key": "refresh_token", "value": "" }
  ]
}
```

### One-Command Complete Test

For the quickest way to test everything:

```bash
# Run the automated test script
./test_complete_system.sh
```

This script will:

- ✅ Setup virtual environment
- ✅ Install dependencies
- ✅ Run migrations
- ✅ Load sample data
- ✅ Start server
- ✅ Test all API endpoints
- ✅ Show database statistics
- ✅ Provide access URLs

### Import Postman Collection

Import the ready-to-use Postman collection:

- **File**: `postman_collection.json` (in project root)
- **Contains**: 20+ pre-configured API requests
- **Features**: Automatic token management, tests, environment variables

> 📖 **For detailed testing instructions, see [QUICK_TESTING_GUIDE.md](QUICK_TESTING_GUIDE.md)**

## 📦 Installation

### Prerequisites

- Python 3.9+
- PostgreSQL 12+
- Redis 6+
- Node.js (optional, for frontend development)

### Quick Start

1. **Clone the repository**

```bash
git clone <repository-url>
cd teera-assessment
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Environment setup**

```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Database setup**

```bash
# For PostgreSQL
createdb teera_reconciliation

# For development (SQLite)
python manage.py migrate
```

6. **Create superuser**

```bash
python manage.py createsuperuser
```

7. **Create sample data (optional)**

```bash
python manage.py create_sample_data --companies 2 --transactions 100
```

8. **Start development server**

```bash
# Terminal 1: Django server
python manage.py runserver

# Terminal 2: Celery worker (optional for development)
celery -A config worker -l info

# Terminal 3: Celery beat (optional for scheduled tasks)
celery -A config beat -l info
```

## 🔧 Configuration

### Environment Variables

```bash
# Basic Configuration
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database Configuration
DB_NAME=teera_reconciliation
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# ML Model Configuration
ML_MODEL_RETRAIN_THRESHOLD=1000
```

### Production Settings

For production deployment, create a `config/settings_production.py`:

```python
from .settings import *

DEBUG = False
ALLOWED_HOSTS = ['your-domain.com']

# Use environment variables for sensitive settings
SECRET_KEY = os.environ['SECRET_KEY']

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ['DB_NAME'],
        'USER': os.environ['DB_USER'],
        'PASSWORD': os.environ['DB_PASSWORD'],
        'HOST': os.environ['DB_HOST'],
        'PORT': os.environ['DB_PORT'],
    }
}

# Security settings
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
```

## 📚 API Documentation

The API is fully documented with OpenAPI/Swagger. Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/api/docs/
- **ReDoc**: http://localhost:8000/api/redoc/
- **OpenAPI Schema**: http://localhost:8000/api/schema/

### Key Endpoints

```
# Authentication
POST /api/auth/token/                 # Obtain JWT token
POST /api/auth/token/refresh/         # Refresh JWT token

# Bank Transactions
GET  /api/v1/bank/transactions/       # List transactions
GET  /api/v1/bank/unmatched/          # List unmatched transactions
POST /api/v1/bank/reconcile/{id}/     # Manual reconciliation

# File Upload
POST /api/v1/bank/upload/             # Upload bank statement

# Reconciliation Logs
GET  /api/v1/bank/logs/               # List reconciliation logs

# Reports
GET  /api/v1/bank/summaries/          # List reconciliation summaries
GET  /api/v1/bank/summaries/{id}/export_pdf/   # Export PDF report
```

### Authentication

Use JWT tokens for API authentication:

```bash
# Obtain token
curl -X POST http://localhost:8000/api/auth/token/ \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'

# Use token in requests
curl -X GET http://localhost:8000/api/v1/bank/transactions/ \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 🤖 Machine Learning

### Model Training

The system automatically trains ML models for each company based on historical reconciliation data.

```bash
# Train models for all companies
python manage.py train_ml_models

# Train model for specific company
python manage.py train_ml_models --company-id <company-uuid>
```

### Feature Engineering

The ML engine extracts various features:

- **Text Features**: TF-IDF vectors from transaction descriptions
- **Numeric Features**: Amount similarities, date proximities
- **Categorical Features**: Transaction types, customer patterns
- **Temporal Features**: Day of week, month patterns

### Model Performance

Models are evaluated on:

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: False negative rate
- **F1-Score**: Harmonic mean of precision and recall

## 🔄 Background Tasks

The system uses Celery for asynchronous processing:

### Scheduled Tasks

- **Nightly Reconciliation**: Automatic matching at 2 AM daily
- **Model Retraining**: Weekly model updates on Sundays
- **Report Generation**: Async PDF/Excel generation

### Task Monitoring

```bash
# Monitor Celery tasks
celery -A config flower  # Web UI at http://localhost:5555

# Check task status
celery -A config inspect active
```

## 📊 Web Interface

Access the web interface at http://localhost:8000/

### Key Features

- **Dashboard**: Overview of reconciliation statistics
- **Transaction Management**: Browse and filter transactions
- **File Upload**: Web-based file upload interface
- **Manual Reconciliation**: Interactive matching interface
- **Reports**: Generate and download reports

### Default Users

After running `create_sample_data`:

- Username: `admin1`, Password: `password123`
- Username: `admin2`, Password: `password123`

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
python manage.py test

# Run specific app tests
python manage.py test reconciliation

# Run with coverage
pip install coverage
coverage run --source='.' manage.py test
coverage report
coverage html  # Generates htmlcov/index.html
```

### Complete System Testing Guide

#### 1. Terminal-Based Testing

**Step 1: Initial Setup**

```bash
# Clone and setup the project
git clone <repository-url>
cd teera-assessment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup database
python manage.py migrate

# Create superuser
python manage.py createsuperuser
# Follow prompts to create admin user
```

**Step 2: Load Sample Data**

```bash
# Generate fresh sample data (optional)
python generate_sample_data.py

# Load sample data into database
python manage.py load_sample_data

# Verify data loading
python manage.py shell -c "
from core.models import Company, User
from reconciliation.models import BankTransaction, Invoice, Customer
print(f'Companies: {Company.objects.count()}')
print(f'Users: {User.objects.count()}')
print(f'Customers: {Customer.objects.count()}')
print(f'Invoices: {Invoice.objects.count()}')
print(f'Bank Transactions: {BankTransaction.objects.count()}')
"
```

**Step 3: Start the Development Server**

```bash
# Terminal 1: Start Django server
python manage.py runserver

# Terminal 2: Start Celery worker (optional for development)
celery -A config worker -l info

# Terminal 3: Start Celery beat (optional for scheduled tasks)
celery -A config beat -l info
```

**Step 4: Test API Endpoints with curl**

```bash
# 1. Obtain JWT Token
curl -X POST http://localhost:8000/api/auth/token/ \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'

# Save the access_token from response
export TOKEN="your_access_token_here"

# 2. Test Bank Transactions List
curl -X GET http://localhost:8000/api/v1/bank/transactions/ \
  -H "Authorization: Bearer $TOKEN"

# 3. Test Unmatched Transactions
curl -X GET http://localhost:8000/api/v1/bank/unmatched/ \
  -H "Authorization: Bearer $TOKEN"

# 4. Test File Upload (using sample data)
curl -X POST http://localhost:8000/api/v1/bank/upload/ \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@sample_data/sample_bank_transactions.csv" \
  -F "file_type=bank_statement"

# 5. Test Customer List
curl -X GET http://localhost:8000/api/v1/bank/customers/ \
  -H "Authorization: Bearer $TOKEN"

# 6. Test Invoice List
curl -X GET http://localhost:8000/api/v1/bank/invoices/ \
  -H "Authorization: Bearer $TOKEN"

# 7. Test Reconciliation Logs
curl -X GET http://localhost:8000/api/v1/bank/logs/ \
  -H "Authorization: Bearer $TOKEN"
```

**Step 5: Test ML Model Training**

```bash
# Train ML models for all companies
python manage.py train_ml_models

# Train model for specific company (if you know the company ID)
python manage.py train_ml_models --company-id <company-uuid>

# Check model training status
python manage.py shell -c "
from reconciliation.models import MLModelVersion
models = MLModelVersion.objects.all()
for model in models:
    print(f'Company: {model.company.name}, Version: {model.version}, Accuracy: {model.accuracy}')
"
```

#### 2. Postman Testing Guide

**Step 1: Setup Postman Environment**

Create a new Postman environment with these variables:

- `base_url`: `http://localhost:8000`
- `access_token`: (will be set after authentication)
- `refresh_token`: (will be set after authentication)

**Step 2: Authentication Collection**

Create a collection called "Bank Reconciliation API" with these requests:

**2.1 Obtain JWT Token**

```
Method: POST
URL: {{base_url}}/api/auth/token/
Headers:
  Content-Type: application/json
Body (raw JSON):
{
  "username": "your_username",
  "password": "your_password"
}

Tests (JavaScript):
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

if (pm.response.code === 200) {
    const jsonData = pm.response.json();
    pm.environment.set("access_token", jsonData.access);
    pm.environment.set("refresh_token", jsonData.refresh);
}
```

**2.2 Refresh Token**

```
Method: POST
URL: {{base_url}}/api/auth/token/refresh/
Headers:
  Content-Type: application/json
Body (raw JSON):
{
  "refresh": "{{refresh_token}}"
}

Tests (JavaScript):
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

if (pm.response.code === 200) {
    const jsonData = pm.response.json();
    pm.environment.set("access_token", jsonData.access);
}
```

**Step 3: API Testing Collection**

**3.1 List Bank Transactions**

```
Method: GET
URL: {{base_url}}/api/v1/bank/transactions/
Headers:
  Authorization: Bearer {{access_token}}

Tests:
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response has transactions", function () {
    const jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('results');
    pm.expect(jsonData.results).to.be.an('array');
});
```

**3.2 List Unmatched Transactions**

```
Method: GET
URL: {{base_url}}/api/v1/bank/unmatched/
Headers:
  Authorization: Bearer {{access_token}}

Tests:
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
```

**3.3 Upload Bank Statement File**

```
Method: POST
URL: {{base_url}}/api/v1/bank/upload/
Headers:
  Authorization: Bearer {{access_token}}
Body (form-data):
  file: [Select sample_data/sample_bank_transactions.csv]
  file_type: bank_statement

Tests:
pm.test("Status code is 201", function () {
    pm.response.to.have.status(201);
});

pm.test("Upload response contains task_id", function () {
    const jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('task_id');
});
```

**3.4 List Customers**

```
Method: GET
URL: {{base_url}}/api/v1/bank/customers/
Headers:
  Authorization: Bearer {{access_token}}

Tests:
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response has customers", function () {
    const jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('results');
    pm.expect(jsonData.results).to.be.an('array');
});
```

**3.5 List Invoices**

```
Method: GET
URL: {{base_url}}/api/v1/bank/invoices/
Headers:
  Authorization: Bearer {{access_token}}

Tests:
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response has invoices", function () {
    const jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('results');
    pm.expect(jsonData.results).to.be.an('array');
});
```

**3.6 Manual Reconciliation**

```
Method: POST
URL: {{base_url}}/api/v1/bank/reconcile/{{transaction_id}}/
Headers:
  Authorization: Bearer {{access_token}}
  Content-Type: application/json
Body (raw JSON):
{
  "invoice_id": "{{invoice_id}}",
  "confidence_score": 0.95,
  "notes": "Manual reconciliation via Postman"
}

Pre-request Script:
// Get a transaction ID and invoice ID from previous requests
// You can set these manually or fetch them dynamically

Tests:
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
```

**3.7 Reconciliation Logs**

```
Method: GET
URL: {{base_url}}/api/v1/bank/logs/
Headers:
  Authorization: Bearer {{access_token}}

Tests:
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
```

**Step 4: Advanced Testing Scenarios**

**4.1 Pagination Testing**

```
Method: GET
URL: {{base_url}}/api/v1/bank/transactions/?page=1&page_size=10
Headers:
  Authorization: Bearer {{access_token}}
```

**4.2 Filtering Testing**

```
Method: GET
URL: {{base_url}}/api/v1/bank/transactions/?is_matched=false&amount_min=1000
Headers:
  Authorization: Bearer {{access_token}}
```

**4.3 Bulk Upload Testing**

```
Method: POST
URL: {{base_url}}/api/v1/bank/upload/
Headers:
  Authorization: Bearer {{access_token}}
Body (form-data):
  file: [Select sample_data/sample_invoices.csv]
  file_type: invoice
```

#### 3. Web Interface Testing

**Step 1: Access Web Interface**

1. Open browser and go to `http://localhost:8000/`
2. Login with your superuser credentials
3. Navigate through the dashboard

**Step 2: Test File Upload Interface**

1. Go to `http://localhost:8000/upload/`
2. Upload `sample_data/sample_bank_transactions.csv`
3. Monitor the upload progress and processing status

**Step 3: Test Manual Reconciliation**

1. Go to `http://localhost:8000/reconciliation/`
2. View unmatched transactions
3. Test manual matching interface

#### 4. API Documentation Testing

**Access Interactive API Documentation:**

- **Swagger UI**: `http://localhost:8000/api/docs/`
- **ReDoc**: `http://localhost:8000/api/redoc/`
- **OpenAPI Schema**: `http://localhost:8000/api/schema/`

**Test directly in Swagger UI:**

1. Click "Authorize" button
2. Enter `Bearer your_access_token`
3. Test endpoints directly in the interface

#### 5. Performance Testing

**Load Testing with curl (simple)**

```bash
# Test concurrent requests
for i in {1..10}; do
  curl -X GET http://localhost:8000/api/v1/bank/transactions/ \
    -H "Authorization: Bearer $TOKEN" &
done
wait

# Test file upload performance
time curl -X POST http://localhost:8000/api/v1/bank/upload/ \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@sample_data/sample_bank_transactions.csv" \
  -F "file_type=bank_statement"
```

#### 6. Database State Verification

**Check Data Integrity:**

```bash
python manage.py shell -c "
from reconciliation.models import BankTransaction, Invoice, ReconciliationLog
from django.db.models import Count, Sum

print('=== Database Statistics ===')
print(f'Total Bank Transactions: {BankTransaction.objects.count()}')
print(f'Matched Transactions: {BankTransaction.objects.filter(is_matched=True).count()}')
print(f'Unmatched Transactions: {BankTransaction.objects.filter(is_matched=False).count()}')
print(f'Total Invoices: {Invoice.objects.count()}')
print(f'Paid Invoices: {Invoice.objects.filter(status=\"paid\").count()}')
print(f'Total Reconciliation Logs: {ReconciliationLog.objects.count()}')
print(f'Total Transaction Amount: {BankTransaction.objects.aggregate(Sum(\"amount\"))}')
"
```

#### 7. Troubleshooting Common Issues

**Issue 1: Authentication Failed**

```bash
# Check if user exists
python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
print('Available users:')
for user in User.objects.all():
    print(f'  - {user.username} (active: {user.is_active})')
"
```

**Issue 2: No Data Available**

```bash
# Reload sample data
python manage.py load_sample_data
```

**Issue 3: Server Not Starting**

```bash
# Check for port conflicts
lsof -i :8000

# Run with different port
python manage.py runserver 8001
```

**Issue 4: File Upload Issues**

```bash
# Check media directory permissions
ls -la media/
chmod 755 media/
chmod 755 uploads/
```

This comprehensive testing guide covers all aspects of the bank reconciliation system, from basic setup to advanced API testing with both terminal commands and Postman collections.

## 📁 File Format Requirements

### CSV Format

```csv
date,description,amount,reference,type
2024-01-15,"Payment from Customer ABC",1500.00,"INV001","credit"
2024-01-16,"Bank charges",-25.00,"FEE001","debit"
```

### Excel Format

Excel files should have the same column structure in the first sheet.

### Required Fields

- `date`: Transaction date (YYYY-MM-DD, DD/MM/YYYY, or MM/DD/YYYY)
- `description`: Transaction description
- `amount`: Transaction amount (positive or negative)

### Optional Fields

- `reference`: Reference number
- `bank_reference`: Bank's reference
- `type`: Transaction type (credit/debit)
- `balance`: Account balance after transaction

## 🔒 Security Considerations

### Data Protection

- All data is isolated by company
- JWT tokens for API access
- IP address logging for audit trails
- Secure file upload validation

### Best Practices

- Regular password updates
- Token expiration (60 minutes)
- HTTPS in production
- Regular security audits

## 🚀 Deployment

### Docker Deployment

```dockerfile
# Dockerfile example
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python manage.py collectstatic --noinput

EXPOSE 8000
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000"]
```

### Docker Compose

```yaml
version: "3.8"
services:
  web:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis
    environment:
      - DEBUG=False

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=teera_reconciliation
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres

  redis:
    image: redis:6-alpine

  worker:
    build: .
    command: celery -A config worker -l info
    depends_on:
      - db
      - redis
```

## 📈 Monitoring & Maintenance

### Health Checks

- Database connectivity
- Redis connectivity
- ML model status
- Background task status

### Maintenance Tasks

- Regular database backups
- Log rotation
- Model retraining
- Performance optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Contact the development team
- Check the API documentation at `/api/docs/`

## 🔮 Future Enhancements

- **OCR Integration**: Process scanned bank statements
- **Blockchain Audit Trail**: Immutable reconciliation logs
- **Advanced Analytics**: Predictive analytics and forecasting
- **Mobile App**: React Native mobile application
- **Integration APIs**: Connect with popular accounting software
- **Real-time Processing**: Stream processing for real-time reconciliation
