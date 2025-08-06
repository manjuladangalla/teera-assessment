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
