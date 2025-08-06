from django.urls import path
from . import web_views

app_name = 'reconciliation_web'

urlpatterns = [
    path('', web_views.DashboardView.as_view(), name='dashboard'),
    path('transactions/', web_views.TransactionListView.as_view(), name='transaction_list'),
    path('transactions/<uuid:pk>/', web_views.TransactionDetailView.as_view(), name='transaction_detail'),
    path('upload/', web_views.FileUploadView.as_view(), name='file_upload'),
    path('reconcile/', web_views.ManualReconciliationView.as_view(), name='manual_reconciliation'),
    path('reports/', web_views.ReportsView.as_view(), name='reports'),
]
