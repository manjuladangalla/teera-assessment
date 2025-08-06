# System Design Document: Advanced Bank Reconciliation System

## Overview

This document outlines the architecture and design decisions for the Advanced Bank Reconciliation System built with Django, PostgreSQL, and machine learning capabilities.

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   ML Engine     │
│                 │    │                 │    │                 │
│ - Web Interface │◄──►│ - Django REST   │◄──►│ - scikit-learn  │
│ - API Clients   │    │ - JWT Auth      │    │ - TensorFlow    │
│ - Mobile Apps   │    │ - Multi-tenant  │    │ - ONNX Runtime  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Storage  │    │   Database      │    │   Task Queue    │
│                 │    │                 │    │                 │
│ - Bank Files    │    │ - PostgreSQL    │    │ - Celery        │
│ - ML Models     │    │ - Multi-schema  │    │ - Redis Broker  │
│ - Reports       │    │ - Audit Logs    │    │ - Background    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Data Model Design

### Core Entities

#### Company (Multi-tenant Root)

- **Purpose**: Root entity for multi-tenancy
- **Key Fields**: id (UUID), name, industry, contact_email
- **Relationships**: One-to-many with all other entities
- **Isolation**: All data scoped by company_id

#### User & UserProfile

- **Purpose**: Authentication and authorization
- **Key Fields**: Django User + company, is_admin, employee_id
- **Security**: Company-scoped permissions

#### Customer

- **Purpose**: Invoice recipients
- **Key Fields**: company, name, email, customer_code
- **Relationships**: One-to-many with Invoice

#### Invoice

- **Purpose**: Bills awaiting payment
- **Key Fields**: customer, invoice_number, amounts, dates, status
- **Business Logic**: Amount validation, status transitions

#### BankTransaction

- **Purpose**: Bank statement entries
- **Key Fields**: company, date, description, amount, status
- **Processing**: Async file upload and parsing

#### ReconciliationLog

- **Purpose**: Match history and audit trail
- **Key Fields**: transaction, invoice, match_type, confidence
- **Audit**: Complete reconciliation history

### Database Schema Design

```sql
-- Multi-tenant architecture with company isolation
CREATE TABLE core_company (
    id UUID PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    industry VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- All tables include company_id for isolation
CREATE TABLE reconciliation_banktransaction (
    id UUID PRIMARY KEY,
    company_id UUID REFERENCES core_company(id),
    transaction_date DATE NOT NULL,
    description TEXT NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'unmatched',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_transaction_company_status ON reconciliation_banktransaction(company_id, status);
CREATE INDEX idx_transaction_date ON reconciliation_banktransaction(transaction_date);
CREATE INDEX idx_transaction_amount ON reconciliation_banktransaction(amount);
```

## Machine Learning Architecture

### Feature Engineering Pipeline

```python
# Text Features
- TF-IDF vectors from transaction descriptions
- Character n-grams for fuzzy matching
- Named entity recognition for company names

# Numeric Features
- Amount similarity ratios
- Date proximity calculations
- Statistical features (percentiles, z-scores)

# Categorical Features
- Transaction types (credit/debit)
- Customer industry classifications
- Seasonal patterns

# Time-series Features
- Day of week patterns
- Monthly seasonality
- Payment term analysis
```

### Model Training Pipeline

1. **Data Preparation**

   - Extract positive examples from ReconciliationLog
   - Generate negative examples through sampling
   - Balance dataset to prevent bias

2. **Feature Extraction**

   - Transaction features: text, amount, date
   - Invoice features: customer, amount, terms
   - Pair features: similarity scores, differences

3. **Model Training**

   - Random Forest for interpretability
   - Cross-validation for generalization
   - Hyperparameter tuning with GridSearch

4. **Model Evaluation**

   - Precision/Recall for imbalanced data
   - Confidence calibration for reliability
   - Feature importance analysis

5. **Model Deployment**
   - ONNX export for production serving
   - Version tracking and rollback capability
   - A/B testing framework

### Prediction Workflow

```python
def predict_matches(transaction):
    # 1. Extract features
    features = feature_extractor.extract(transaction)

    # 2. Get candidate invoices
    candidates = get_candidate_invoices(transaction)

    # 3. Score each candidate
    scores = []
    for invoice in candidates:
        pair_features = extract_pair_features(transaction, invoice)
        score = model.predict_proba(pair_features)[1]
        scores.append((invoice, score))

    # 4. Filter by confidence threshold
    matches = [(inv, score) for inv, score in scores if score > threshold]

    # 5. Return ranked matches
    return sorted(matches, key=lambda x: x[1], reverse=True)
```

## API Design

### RESTful API Structure

```
/api/v1/
├── auth/
│   ├── token/          # JWT token obtain
│   ├── token/refresh/  # JWT token refresh
│   └── token/verify/   # JWT token verify
├── core/
│   ├── companies/      # Company management
│   ├── customers/      # Customer CRUD
│   └── invoices/       # Invoice CRUD
├── bank/
│   ├── transactions/   # Transaction list/detail
│   ├── unmatched/      # Unmatched transactions
│   ├── upload/         # File upload endpoint
│   ├── reconcile/      # Manual reconciliation
│   ├── logs/           # Reconciliation history
│   └── summaries/      # Reporting endpoints
└── ml/
    ├── models/         # ML model management
    ├── predictions/    # Prediction endpoints
    └── training/       # Model training triggers
```

### Authentication & Authorization

```python
# JWT Token Structure
{
    "user_id": 123,
    "company_id": "uuid-here",
    "username": "user@company.com",
    "permissions": ["reconcile", "admin"],
    "exp": 1234567890
}

# Permission Classes
class IsCompanyMember(BasePermission):
    def has_permission(self, request, view):
        return request.user.profile.company == resource.company

class IsCompanyAdmin(BasePermission):
    def has_permission(self, request, view):
        return request.user.profile.is_admin
```

## File Processing Architecture

### Asynchronous File Processing

```python
# Upload Flow
1. User uploads file → FileUploadStatus created
2. File saved to temporary storage
3. Celery task queued for processing
4. Background worker processes file
5. BankTransaction records created
6. ML matching triggered automatically
7. User notified of completion
```

### File Format Support

#### CSV Processing

- Automatic delimiter detection
- Header validation and mapping
- Error handling and reporting
- Progress tracking

#### Excel Processing

- Multi-sheet support
- Formula evaluation
- Date format auto-detection
- Memory-efficient streaming

### Data Validation Pipeline

```python
class BankStatementValidator:
    def validate_file(self, file_path):
        # 1. File format validation
        # 2. Required header checking
        # 3. Data type validation
        # 4. Business rule validation
        # 5. Duplicate detection
        # 6. Data quality scoring
        return validation_report
```

## Background Task Architecture

### Celery Task Design

```python
# File Processing Task
@shared_task(bind=True)
def process_bank_statement_file(self, file_upload_id):
    # Progress tracking with task state updates
    self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100})

    # Error handling with retry logic
    try:
        result = process_file(file_upload_id)
        return result
    except Exception as exc:
        self.retry(countdown=60, max_retries=3, exc=exc)

# ML Training Task
@shared_task
def train_ml_model(company_id):
    # Resource-intensive task with timeout
    # Model versioning and deployment
    # Performance metric tracking
    pass

# Scheduled Reconciliation
@periodic_task(run_every=crontab(hour=2, minute=0))
def nightly_reconciliation():
    # Batch processing for all companies
    # ML-based auto-matching
    # Report generation
    pass
```

### Task Monitoring

- Celery Flower for task monitoring
- Custom task status tracking
- Error notification system
- Performance metrics collection

## Security Architecture

### Multi-tenant Data Isolation

```python
# Model-level isolation
class CompanyQuerySet(models.QuerySet):
    def for_company(self, company):
        return self.filter(company=company)

class BankTransaction(models.Model):
    company = models.ForeignKey(Company)
    objects = CompanyQuerySet.as_manager()

    class Meta:
        # Prevent cross-company data access
        constraints = [
            models.UniqueConstraint(
                fields=['company', 'reference_number'],
                name='unique_reference_per_company'
            )
        ]
```

### Audit Logging

```python
# Comprehensive audit trail
class AuditLog(models.Model):
    user = models.ForeignKey(User)
    company = models.ForeignKey(Company)
    action = models.CharField(max_length=20)
    model_name = models.CharField(max_length=100)
    object_id = models.CharField(max_length=100)
    changes = models.JSONField()  # Before/after values
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

# Automatic audit logging
@receiver(post_save)
def log_model_changes(sender, instance, **kwargs):
    if hasattr(instance, 'company'):
        create_audit_log(instance, 'update')
```

### File Upload Security

```python
# File validation and sanitization
class SecureFileUpload:
    ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls']
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    def validate_file(self, uploaded_file):
        # 1. Extension validation
        # 2. MIME type verification
        # 3. Content scanning
        # 4. Size limits
        # 5. Virus scanning (if available)
        pass
```

## Performance Optimization

### Database Optimization

```sql
-- Strategic indexing
CREATE INDEX CONCURRENTLY idx_transaction_company_date
ON reconciliation_banktransaction(company_id, transaction_date DESC);

CREATE INDEX CONCURRENTLY idx_reconciliation_log_active
ON reconciliation_reconciliationlog(transaction_id, is_active)
WHERE is_active = true;

-- Partitioning for large datasets
CREATE TABLE reconciliation_banktransaction_y2024
PARTITION OF reconciliation_banktransaction
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

### Caching Strategy

```python
# Redis caching for expensive operations
@cache_result(timeout=3600)
def get_company_statistics(company_id):
    # Expensive aggregation queries
    return statistics

# Model-level caching
class BankTransaction(models.Model):
    @cached_property
    def potential_matches(self):
        return self.calculate_ml_matches()
```

### API Performance

```python
# Pagination and filtering
class BankTransactionViewSet(viewsets.ModelViewSet):
    pagination_class = LargeResultsSetPagination
    filter_backends = [DjangoFilterBackend, SearchFilter]

    def get_queryset(self):
        return BankTransaction.objects.select_related(
            'company', 'file_upload'
        ).prefetch_related(
            'reconciliation_logs__invoice'
        )
```

## Deployment Architecture

### Production Environment

```yaml
# Docker Compose Production
version: "3.8"
services:
  web:
    build: .
    environment:
      - DJANGO_SETTINGS_MODULE=config.settings_production
    depends_on:
      - db
      - redis

  worker:
    build: .
    command: celery -A config worker -l info
    depends_on:
      - db
      - redis

  beat:
    build: .
    command: celery -A config beat -l info
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=teera_reconciliation

  redis:
    image: redis:6-alpine

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
```

### Monitoring & Logging

```python
# Structured logging
LOGGING = {
    'formatters': {
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json'
        }
    }
}
```

## Scalability Considerations

### Horizontal Scaling

1. **Database Sharding**: Partition by company_id
2. **Read Replicas**: Separate read/write workloads
3. **Microservices**: Split ML engine into separate service
4. **CDN**: Static file delivery optimization

### Performance Monitoring

1. **Application Metrics**: Response times, error rates
2. **Database Metrics**: Query performance, connection pools
3. **ML Metrics**: Prediction accuracy, training times
4. **Business Metrics**: Reconciliation rates, user adoption

## Disaster Recovery

### Backup Strategy

1. **Database Backups**: Daily full, hourly incremental
2. **File Backups**: ML models, uploaded files
3. **Configuration Backups**: Environment settings
4. **Cross-region Replication**: Geographic redundancy

### Recovery Procedures

1. **RTO Target**: 4 hours maximum downtime
2. **RPO Target**: 1 hour maximum data loss
3. **Testing**: Monthly disaster recovery drills
4. **Documentation**: Step-by-step recovery procedures

This comprehensive system design ensures scalability, security, and maintainability while delivering advanced reconciliation capabilities through machine learning integration.
