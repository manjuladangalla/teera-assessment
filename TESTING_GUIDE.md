# Bank Reconciliation System - Local Testing Guide

## 🚀 System is Running!

Your advanced bank reconciliation system is now running locally at:

- **Web Interface**: http://127.0.0.1:8000/
- **Admin Panel**: http://127.0.0.1:8000/admin/
- **API Documentation**: http://127.0.0.1:8000/api/docs/

## 🔐 Login Credentials

**Admin User:**

- Username: `admin`
- Password: `admin123`

## 📊 Sample Data Available

The system has been populated with:

- **1 Company**: Demo Company Ltd
- **3 Customers**: ABC Corp, XYZ Industries, Tech Solutions Inc
- **9 Invoices**: 3 per customer with amounts ranging from $1,210 to $3,630
- **6 Bank Transactions**: Various payment types with matching references
- **1 File Upload Record**: Sample bank statement file

## 🧪 Testing Features

### 1. Admin Panel Testing

- Go to http://127.0.0.1:8000/admin/
- Login with admin credentials
- Explore the following sections:
  - **Core**: Companies, Customers, Invoices, User Profiles
  - **Reconciliation**: Bank Transactions, Reconciliation Logs, File Upload Status
  - **ML Engine**: Training Data, Model Predictions, Feature Extractions

### 2. API Testing

- Visit http://127.0.0.1:8000/api/docs/ for interactive API documentation
- Key endpoints to test:
  - `GET /api/v1/bank-transactions/` - View bank transactions
  - `GET /api/v1/invoices/` - View invoices
  - `GET /api/v1/customers/` - View customers
  - `POST /api/v1/auth/token/` - Get JWT token for authentication

### 3. Web Interface Testing

- Visit http://127.0.0.1:8000/
- Navigate through the dashboard
- Test the reconciliation interface
- Upload sample bank statement files

### 4. Manual Reconciliation Testing

1. Go to Bank Transactions section
2. Find unmatched transactions
3. Use the manual reconciliation feature to match with invoices
4. Check confidence scores and matching accuracy

### 5. API Authentication Testing

```bash
# Get JWT token
curl -X POST http://127.0.0.1:8000/api/v1/auth/token/ \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Use token in subsequent requests
curl -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  http://127.0.0.1:8000/api/v1/bank-transactions/
```

## 🔍 Test Scenarios

### Scenario 1: Perfect Matches

- Transaction `BANK202508060002` ($1,210) should match Invoice `INV-ABC001-001` ($1,210)
- Transaction `BANK202508060003` ($1,320) should match Invoice `INV-ABC001-002` ($1,320)

### Scenario 2: Reference-Based Matching

- Look for transactions with reference numbers matching invoice numbers
- Test the ML engine's ability to find matches based on text similarity

### Scenario 3: Partial Matches

- Some transactions may partially match invoices
- Test the system's handling of partial reconciliation

## 🛠️ Technical Features to Test

### Multi-Tenant Architecture

- All data is scoped to "Demo Company Ltd"
- Test that user permissions are properly enforced

### File Upload Processing

- The system includes file processors for CSV and Excel files
- Currently in minimal mode (ML features disabled)

### Audit Logging

- All reconciliation actions are logged
- Check the AuditLog model in admin panel

### API Features

- JWT authentication
- Pagination
- Filtering and search
- Comprehensive error handling

## 📈 Performance Testing

### Database Queries

- Monitor Django Debug Toolbar for query optimization
- Test with larger datasets

### API Response Times

- Use the interactive API docs to test response times
- Check pagination with large result sets

## 🐛 Troubleshooting

### Common Issues:

1. **Server not starting**: Check if port 8000 is available
2. **Database errors**: Ensure migrations are applied
3. **Permission errors**: Verify user has proper company association

### Debug Mode:

- DEBUG is enabled in development
- Check console for detailed error messages
- Use Django Admin for data inspection

## 🚀 Next Steps

### Enable Full Features:

1. Install pandas for ML capabilities
2. Set up Redis for Celery task processing
3. Configure email backend for notifications
4. Add PostgreSQL for production-like testing

### Advanced Testing:

1. Upload real bank statement files
2. Test with larger datasets
3. Performance benchmarking
4. Security testing

## 📝 Notes

- This is a minimal setup optimized for quick testing
- ML features are commented out due to Python 3.13 compatibility
- SQLite is used for simplicity (PostgreSQL recommended for production)
- Background tasks are disabled (Celery integration available)

---

**Happy Testing! 🎉**

The system demonstrates enterprise-level architecture with:

- ✅ Multi-tenant design
- ✅ REST API with authentication
- ✅ Comprehensive data models
- ✅ File processing capabilities
- ✅ Audit logging
- ✅ Admin interface
- ✅ Responsive web UI
- ✅ ML-ready architecture
