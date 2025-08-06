# 🚀 Bank Reconciliation System - Quick Testing Guide

## ⚡ Super Quick Start (1 minute)

```bash
git clone <repo-url> && cd teera-assessment
./test_complete_system.sh
```

## 🔑 Default Test Credentials

After running the test script:

- **Username**: `testadmin`
- **Password**: `testpass123`

## 🌐 Important URLs

| Service                | URL                              | Description          |
| ---------------------- | -------------------------------- | -------------------- |
| **Web Interface**      | http://localhost:8000/           | Main dashboard       |
| **API Docs (Swagger)** | http://localhost:8000/api/docs/  | Interactive API docs |
| **API Docs (ReDoc)**   | http://localhost:8000/api/redoc/ | Alternative API docs |
| **Admin Interface**    | http://localhost:8000/admin/     | Django admin         |
| **API Base**           | http://localhost:8000/api/v1/    | REST API endpoints   |

## 📋 Key API Endpoints

| Method | Endpoint                       | Description            |
| ------ | ------------------------------ | ---------------------- |
| `POST` | `/api/auth/token/`             | Get JWT token          |
| `GET`  | `/api/v1/bank/transactions/`   | List transactions      |
| `GET`  | `/api/v1/bank/unmatched/`      | Unmatched transactions |
| `POST` | `/api/v1/bank/upload/`         | Upload files           |
| `GET`  | `/api/v1/bank/customers/`      | List customers         |
| `GET`  | `/api/v1/bank/invoices/`       | List invoices          |
| `POST` | `/api/v1/bank/reconcile/{id}/` | Manual reconcile       |
| `GET`  | `/api/v1/bank/logs/`           | Reconciliation logs    |

## 🔧 Quick Terminal Tests

### 1. Get JWT Token

```bash
curl -X POST http://localhost:8000/api/auth/token/ \
  -H "Content-Type: application/json" \
  -d '{"username": "testadmin", "password": "testpass123"}'
```

### 2. Test API (replace TOKEN)

```bash
export TOKEN="your_access_token_here"
curl -X GET http://localhost:8000/api/v1/bank/transactions/ \
  -H "Authorization: Bearer $TOKEN"
```

### 3. Upload Sample File

```bash
curl -X POST http://localhost:8000/api/v1/bank/upload/ \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@sample_data/sample_bank_transactions.csv" \
  -F "file_type=bank_statement"
```

## 📦 Postman Quick Setup

### 1. Import Collection

- Import file: `postman_collection.json`

### 2. Create Environment

```json
{
  "name": "Bank Reconciliation Local",
  "values": [{ "key": "base_url", "value": "http://localhost:8000" }]
}
```

### 3. First Request

- Run "Obtain JWT Token" request
- All other requests will work automatically

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000
# Kill the process
kill -9 <PID>
# Or use different port
python manage.py runserver 8001
```

### Virtual Environment Issues

```bash
# Create new virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Database Issues

```bash
# Reset database
rm db.sqlite3
python manage.py migrate
python manage.py load_sample_data
```

### No Sample Data

```bash
# Load fresh sample data
python manage.py load_sample_data
```

## 📊 Sample Data Overview

After loading sample data, you'll have:

- **8 customers** (Enterprise Solutions, Digital Marketing Co, etc.)
- **52 invoices** with various statuses
- **48 bank transactions** with different amounts
- **Realistic transaction descriptions** for ML training

## 🤖 ML Model Testing

```bash
# Train ML models
python manage.py train_ml_models

# Check model status
python manage.py shell -c "
from reconciliation.models import MLModelVersion
for model in MLModelVersion.objects.all():
    print(f'{model.company.name}: v{model.version} ({model.accuracy}% accuracy)')
"
```

## 🔍 Quick Health Check

```bash
# Check system status
python manage.py shell -c "
from reconciliation.models import *
from core.models import *
print(f'Companies: {Company.objects.count()}')
print(f'Transactions: {BankTransaction.objects.count()}')
print(f'Invoices: {Invoice.objects.count()}')
print(f'Matched: {BankTransaction.objects.filter(is_matched=True).count()}')
"
```

## 🎯 Testing Checklist

- [ ] Server starts on port 8000
- [ ] Can get JWT token with test credentials
- [ ] API endpoints return data
- [ ] File upload works (returns task_id)
- [ ] Swagger docs accessible
- [ ] Sample data loaded correctly
- [ ] Manual reconciliation works
- [ ] Web interface accessible

## 📞 Need Help?

1. Check server logs for errors
2. Verify virtual environment is activated
3. Ensure all dependencies are installed
4. Try the automated test script: `./test_complete_system.sh`
5. Check API documentation at `/api/docs/`

---

**Happy Testing! 🚀**
