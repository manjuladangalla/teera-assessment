from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create router and register viewsets
router = DefaultRouter()
router.register(r'bank/transactions', views.BankTransactionViewSet, basename='banktransaction')
router.register(r'bank/logs', views.ReconciliationLogViewSet, basename='reconciliationlog')
router.register(r'bank/uploads', views.FileUploadViewSet, basename='fileupload')
router.register(r'bank/rules', views.MatchingRuleViewSet, basename='matchingrule')
router.register(r'bank/summaries', views.ReconciliationSummaryViewSet, basename='reconciliationsummary')

app_name = 'reconciliation'

urlpatterns = [
    # Include router URLs
    path('', include(router.urls)),
    
    # Custom endpoints
    path('bank/upload/', views.FileUploadViewSet.as_view({'post': 'upload'}), name='bank-upload'),
    path('bank/unmatched/', views.BankTransactionViewSet.as_view({'get': 'unmatched'}), name='bank-unmatched'),
    path('bank/reconcile/<uuid:pk>/', views.BankTransactionViewSet.as_view({'post': 'reconcile'}), name='bank-reconcile'),
]
