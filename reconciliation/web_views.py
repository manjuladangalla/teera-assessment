from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, DetailView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.db.models import Count, Sum, Q
from django.utils import timezone
from datetime import timedelta

from .models import BankTransaction, ReconciliationLog, FileUploadStatus
from .utils import get_user_company
from core.models import Invoice


class DashboardView(LoginRequiredMixin, TemplateView):
    """Dashboard view with summary statistics."""
    template_name = 'dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        try:
            company = get_user_company(self.request.user)
            
            # Get current month statistics
            now = timezone.now()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            transactions = BankTransaction.objects.filter(
                company=company,
                transaction_date__gte=month_start.date()
            )
            
            stats = {
                'total_transactions': transactions.count(),
                'matched_transactions': transactions.filter(status='matched').count(),
                'unmatched_transactions': transactions.filter(status='unmatched').count(),
                'ml_matches': ReconciliationLog.objects.filter(
                    transaction__company=company,
                    matched_by='ml_auto',
                    created_at__gte=month_start
                ).count(),
            }
            
            if stats['total_transactions'] > 0:
                stats['match_percentage'] = round(
                    (stats['matched_transactions'] / stats['total_transactions']) * 100, 1
                )
            else:
                stats['match_percentage'] = 0
            
            context['stats'] = stats
            
            # Recent transactions
            context['recent_transactions'] = transactions.order_by('-transaction_date')[:10]
            
            # Last sync time (placeholder)
            context['last_sync'] = timezone.now() - timedelta(hours=1)
            
        except Exception as e:
            messages.error(self.request, f"Error loading dashboard: {e}")
            context['stats'] = {}
            context['recent_transactions'] = []
        
        return context


class TransactionListView(LoginRequiredMixin, ListView):
    """List view for bank transactions."""
    model = BankTransaction
    template_name = 'transaction_list.html'
    context_object_name = 'transactions'
    paginate_by = 50
    
    def get_queryset(self):
        company = get_user_company(self.request.user)
        queryset = BankTransaction.objects.filter(company=company).order_by('-transaction_date')
        
        # Apply filters
        status = self.request.GET.get('status')
        if status:
            queryset = queryset.filter(status=status)
        
        date_from = self.request.GET.get('date_from')
        if date_from:
            queryset = queryset.filter(transaction_date__gte=date_from)
        
        date_to = self.request.GET.get('date_to')
        if date_to:
            queryset = queryset.filter(transaction_date__lte=date_to)
        
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(
                Q(description__icontains=search) |
                Q(reference_number__icontains=search) |
                Q(bank_reference__icontains=search)
            )
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['filter_status'] = self.request.GET.get('status', '')
        context['filter_date_from'] = self.request.GET.get('date_from', '')
        context['filter_date_to'] = self.request.GET.get('date_to', '')
        context['filter_search'] = self.request.GET.get('search', '')
        return context


class TransactionDetailView(LoginRequiredMixin, DetailView):
    """Detail view for a single transaction."""
    model = BankTransaction
    template_name = 'transaction_detail.html'
    context_object_name = 'transaction'
    
    def get_queryset(self):
        company = get_user_company(self.request.user)
        return BankTransaction.objects.filter(company=company)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get reconciliation logs
        context['reconciliation_logs'] = self.object.reconciliation_logs.filter(
            is_active=True
        ).select_related('invoice', 'user')
        
        # Get potential matches (for manual reconciliation)
        if self.object.status == 'unmatched':
            company = get_user_company(self.request.user)
            potential_invoices = Invoice.objects.filter(
                customer__company=company,
                status__in=['sent', 'overdue'],
                total_amount__gte=float(self.object.amount) * 0.8,
                total_amount__lte=float(self.object.amount) * 1.2
            )[:10]
            context['potential_invoices'] = potential_invoices
        
        return context


class FileUploadView(LoginRequiredMixin, TemplateView):
    """File upload view."""
    template_name = 'file_upload.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get recent uploads
        company = get_user_company(self.request.user)
        context['recent_uploads'] = FileUploadStatus.objects.filter(
            company=company
        ).order_by('-created_at')[:10]
        
        return context


class ManualReconciliationView(LoginRequiredMixin, TemplateView):
    """Manual reconciliation interface."""
    template_name = 'manual_reconciliation.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        company = get_user_company(self.request.user)
        
        # Get unmatched transactions
        context['unmatched_transactions'] = BankTransaction.objects.filter(
            company=company,
            status='unmatched'
        ).order_by('-transaction_date')[:50]
        
        # Get unpaid invoices
        context['unpaid_invoices'] = Invoice.objects.filter(
            customer__company=company,
            status__in=['sent', 'overdue']
        ).order_by('-due_date')[:100]
        
        return context


class ReportsView(LoginRequiredMixin, TemplateView):
    """Reports and analytics view."""
    template_name = 'reports.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        company = get_user_company(self.request.user)
        
        # Monthly statistics
        now = timezone.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        transactions = BankTransaction.objects.filter(
            company=company,
            transaction_date__gte=month_start.date()
        )
        
        context['monthly_stats'] = {
            'total_transactions': transactions.count(),
            'total_amount': transactions.aggregate(Sum('amount'))['amount__sum'] or 0,
            'matched_transactions': transactions.filter(status='matched').count(),
            'unmatched_transactions': transactions.filter(status='unmatched').count(),
        }
        
        # Reconciliation method breakdown
        logs = ReconciliationLog.objects.filter(
            transaction__company=company,
            created_at__gte=month_start,
            is_active=True
        )
        
        context['reconciliation_methods'] = logs.values('matched_by').annotate(
            count=Count('id')
        ).order_by('-count')
        
        return context
