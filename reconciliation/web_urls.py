from django.urls import path
from . import web_views
from .api_views_simple import DebugReconciliationView, simple_reconcile

app_name = 'reconciliation_web'

urlpatterns = [
    path('', web_views.DashboardView.as_view(), name='dashboard'),
    path('transactions/', web_views.TransactionListView.as_view(), name='transaction_list'),
    path('transactions/<uuid:pk>/', web_views.TransactionDetailView.as_view(), name='transaction_detail'),
    path('upload/', web_views.FileUploadView.as_view(), name='file_upload'),
    path('reconcile/', web_views.ManualReconciliationView.as_view(), name='manual_reconciliation'),
    path('reports/', web_views.ReportsView.as_view(), name='reports'),

    path('api/reconcile/', web_views.ManualReconciliationView.as_view(), name='ajax_reconcile'),
    path('api/suggest-matches/<uuid:transaction_id>/', web_views.SuggestMatchesView.as_view(), name='suggest_matches'),
    path('api/transaction-search/', web_views.TransactionSearchView.as_view(), name='transaction_search'),

    path('debug/reconcile/<uuid:transaction_id>/', DebugReconciliationView.debug_reconcile, name='debug_reconcile'),
    path('simple/reconcile/<uuid:transaction_id>/', simple_reconcile, name='simple_reconcile'),
]
