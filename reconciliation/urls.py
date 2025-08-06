"""
Advanced API URLs for bank reconciliation system.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    BankTransactionViewSet, ReconciliationLogViewSet,
    FileUploadViewSet as BasicFileUploadViewSet, MatchingRuleViewSet,
    ReconciliationSummaryViewSet
)
from .api_views import (
    BankTransactionViewSet as AdvancedBankTransactionViewSet,
    FileUploadViewSet, ReconciliationLogViewSet as AdvancedReconciliationLogViewSet,
    MLModelViewSet, ReportViewSet
)
from .api_root import APIRootView

# API Router for DRF ViewSets
router = DefaultRouter()
router.register(r'bank/transactions', AdvancedBankTransactionViewSet, basename='banktransaction')
router.register(r'bank/uploads', FileUploadViewSet, basename='fileupload')
router.register(r'bank/logs', AdvancedReconciliationLogViewSet, basename='reconciliationlog')
router.register(r'bank/rules', MatchingRuleViewSet, basename='matchingrule')
router.register(r'bank/summaries', ReconciliationSummaryViewSet, basename='reconciliationsummary')
router.register(r'ml/models', MLModelViewSet, basename='mlmodel')
router.register(r'reports', ReportViewSet, basename='reports')

urlpatterns = [
    # API Root - unauthenticated endpoint discovery
    path('', APIRootView.as_view(), name='api-root'),
    
    # Custom simplified endpoints
    path('bank/upload/', FileUploadViewSet.as_view({'post': 'create'}), name='bank-upload'),
    path('bank/unmatched/', AdvancedBankTransactionViewSet.as_view({'get': 'unmatched'}), name='bank-unmatched'),
    path('bank/reconcile/', AdvancedBankTransactionViewSet.as_view({'post': 'bulk_reconcile'}), name='bank-reconcile'),
    path('bank/logs/', AdvancedReconciliationLogViewSet.as_view({'get': 'list'}), name='bank-logs'),
    path('bank/summary/', ReportViewSet.as_view({'get': 'generate_summary'}), name='bank-summary'),
    
    # Include DRF router URLs (full endpoints)
    path('', include(router.urls)),
    
    # Legacy endpoints (for backward compatibility)
    path('legacy/', include([
        path('', include(router.urls)),
    ])),
]

# Add API endpoint documentation
"""
API Endpoints:

## Simplified Bank Reconciliation Endpoints (NEW)
- POST   /api/v1/bank/upload/                    - Upload bank statement file
- GET    /api/v1/bank/unmatched/                 - List unmatched bank transactions
- POST   /api/v1/bank/reconcile/                 - Manually reconcile transaction(s)
- GET    /api/v1/bank/logs/                      - View reconciliation logs
- GET    /api/v1/bank/summary/                   - Download reconciliation summary PDF/XLSX

## Full Bank Transaction Management (Detailed)
- GET    /api/v1/bank/transactions/              - List all transactions
- POST   /api/v1/bank/transactions/              - Create transaction
- GET    /api/v1/bank/transactions/{id}/         - Get transaction details
- PUT    /api/v1/bank/transactions/{id}/         - Update transaction
- DELETE /api/v1/bank/transactions/{id}/         - Delete transaction
- GET    /api/v1/bank/transactions/unmatched/    - Get unmatched transactions
- GET    /api/v1/bank/transactions/{id}/ml_suggestions/ - Get ML suggestions
- POST   /api/v1/bank/transactions/trigger_ml_matching/ - Trigger ML matching
- POST   /api/v1/bank/transactions/{id}/reconcile/ - Manual reconciliation (single)
- POST   /api/v1/bank/transactions/bulk_operations/ - Bulk operations
- POST   /api/v1/bank/transactions/bulk_reconcile/ - Bulk reconcile
- GET    /api/v1/bank/transactions/statistics/   - Get statistics

## File Upload Management (Detailed)
- GET    /api/v1/bank/uploads/                   - List uploads
- POST   /api/v1/bank/uploads/                   - Upload file
- GET    /api/v1/bank/uploads/{id}/              - Get upload details
- GET    /api/v1/bank/uploads/{id}/status/       - Get upload status

## Reconciliation Logs (Detailed)
- GET    /api/v1/bank/logs/                      - List reconciliation logs
- GET    /api/v1/bank/logs/{id}/                 - Get log details
- POST   /api/v1/bank/logs/{id}/rollback/        - Rollback reconciliation

## ML Model Management
- GET    /api/v1/ml/models/                      - List ML models
- GET    /api/v1/ml/models/{id}/                 - Get model details
- POST   /api/v1/ml/models/retrain/              - Trigger retraining
- GET    /api/v1/ml/models/performance/          - Get model performance

## Report Generation
- POST   /api/v1/reports/generate_summary/       - Generate report
- GET    /api/v1/reports/download/               - Download report

All endpoints require JWT authentication and enforce company-based data isolation.
"""
