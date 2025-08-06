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

# API Router for DRF ViewSets (Full CRUD endpoints)
router = DefaultRouter()
router.register(r'bank/rules', MatchingRuleViewSet, basename='matchingrule')
router.register(r'bank/summaries', ReconciliationSummaryViewSet, basename='reconciliationsummary')
router.register(r'ml/models', MLModelViewSet, basename='mlmodel')

urlpatterns = [
    # API Root - unauthenticated endpoint discovery
    path('', APIRootView.as_view(), name='api-root'),
    
    # Essential business endpoints (for Postman collection)
    path('bank/upload/', FileUploadViewSet.as_view({'post': 'create'}), name='bank-upload'),
    path('bank/unmatched/', AdvancedBankTransactionViewSet.as_view({'get': 'unmatched'}), name='bank-unmatched'),
    path('bank/reconcile/', AdvancedBankTransactionViewSet.as_view({'post': 'bulk_reconcile'}), name='bank-reconcile'),
    path('bank/logs/', AdvancedReconciliationLogViewSet.as_view({'get': 'list'}), name='bank-logs'),
    path('bank/summary/', ReportViewSet.as_view({'post': 'generate_summary'}), name='bank-summary'),
    
    # Full CRUD endpoints for complex operations
    path('bank/transactions/', AdvancedBankTransactionViewSet.as_view({'get': 'list', 'post': 'create'}), name='transactions-list'),
    path('bank/transactions/<uuid:pk>/', AdvancedBankTransactionViewSet.as_view({'get': 'retrieve', 'put': 'update', 'delete': 'destroy'}), name='transactions-detail'),
    path('bank/uploads/', FileUploadViewSet.as_view({'get': 'list'}), name='uploads-list'),
    path('bank/uploads/<uuid:pk>/', FileUploadViewSet.as_view({'get': 'retrieve'}), name='uploads-detail'),
    
    # Include remaining router URLs
    path('', include(router.urls)),
]
