from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import TemplateView, ListView, DetailView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.db.models import Count, Sum, Q, F
from django.utils import timezone
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from datetime import timedelta
import json

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
    template_name = 'reconciliation/transaction_list.html'
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
    template_name = 'reconciliation/transaction_detail.html'
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
    template_name = 'reconciliation/file_upload.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get recent uploads
        company = get_user_company(self.request.user)
        context['recent_uploads'] = FileUploadStatus.objects.filter(
            company=company
        ).order_by('-created_at')[:10]
        
        return context


class ManualReconciliationView(LoginRequiredMixin, TemplateView):
    """Enhanced manual reconciliation interface with smart matching."""
    template_name = 'reconciliation/manual_reconciliation.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        company = get_user_company(self.request.user)
        
        # Get unmatched transactions with better filtering
        unmatched_transactions = BankTransaction.objects.filter(
            company=company,
            status='unmatched'
        ).select_related('file_upload').order_by('-transaction_date', '-created_at')
        
        # Apply search filter if provided
        search_query = self.request.GET.get('search', '').strip()
        if search_query:
            unmatched_transactions = unmatched_transactions.filter(
                Q(description__icontains=search_query) |
                Q(reference_number__icontains=search_query) |
                Q(bank_reference__icontains=search_query) |
                Q(amount__icontains=search_query.replace('$', '').replace(',', ''))
            )
        
        # Apply date filter if provided
        date_from = self.request.GET.get('date_from')
        date_to = self.request.GET.get('date_to')
        if date_from:
            unmatched_transactions = unmatched_transactions.filter(transaction_date__gte=date_from)
        if date_to:
            unmatched_transactions = unmatched_transactions.filter(transaction_date__lte=date_to)
        
        # Limit results for performance
        context['unmatched_transactions'] = unmatched_transactions[:100]
        
        # Get unpaid invoices with better filtering
        unpaid_invoices = Invoice.objects.filter(
            customer__company=company,
            status__in=['sent', 'overdue', 'pending']
        ).select_related('customer').order_by('-due_date', '-created_at')
        
        # Apply invoice search filter if provided
        invoice_search = self.request.GET.get('invoice_search', '').strip()
        if invoice_search:
            unpaid_invoices = unpaid_invoices.filter(
                Q(invoice_number__icontains=invoice_search) |
                Q(customer__name__icontains=invoice_search) |
                Q(description__icontains=invoice_search) |
                Q(total_amount__icontains=invoice_search.replace('$', '').replace(',', ''))
            )
        
        # Limit results for performance
        context['unpaid_invoices'] = unpaid_invoices[:200]
        
        # Add statistics for the interface
        context['stats'] = {
            'total_unmatched': unmatched_transactions.count(),
            'total_unpaid_invoices': unpaid_invoices.count(),
            'unmatched_amount': unmatched_transactions.aggregate(
                total=Sum('amount')
            )['total'] or 0,
            'unpaid_amount': unpaid_invoices.aggregate(
                total=Sum('total_amount')
            )['total'] or 0,
        }
        
        # Add recent reconciliations for reference
        context['recent_reconciliations'] = ReconciliationLog.objects.filter(
            transaction__company=company,
            is_active=True
        ).select_related('transaction', 'invoice', 'user').order_by('-created_at')[:10]
        
        # Add filter values to context for form state
        context['filters'] = {
            'search': search_query,
            'date_from': date_from or '',
            'date_to': date_to or '',
            'invoice_search': invoice_search,
        }
        
        return context
    
    def post(self, request, *args, **kwargs):
        """Handle AJAX reconciliation requests."""
        if not request.content_type == 'application/json':
            return JsonResponse({'error': 'Content-Type must be application/json'}, status=400)
        
        try:
            data = json.loads(request.body)
            transaction_id = data.get('transaction_id')
            invoice_ids = data.get('invoice_ids', [])
            notes = data.get('notes', '')
            
            if not transaction_id or not invoice_ids:
                return JsonResponse({'error': 'Transaction ID and invoice IDs are required'}, status=400)
            
            company = get_user_company(request.user)
            
            # Get the transaction
            transaction = get_object_or_404(
                BankTransaction,
                id=transaction_id,
                company=company,
                status='unmatched'
            )
            
            # Validate and get invoices
            invoices = Invoice.objects.filter(
                id__in=invoice_ids,
                customer__company=company,
                status__in=['sent', 'overdue', 'pending']
            )
            
            if len(invoices) != len(invoice_ids):
                return JsonResponse({'error': 'Some invoices not found or already paid'}, status=400)
            
            # Create reconciliation logs
            reconciliation_logs = []
            total_invoice_amount = 0
            
            for invoice in invoices:
                total_invoice_amount += invoice.total_amount
                
                log = ReconciliationLog.objects.create(
                    transaction=transaction,
                    invoice=invoice,
                    matched_by='manual',
                    confidence_score=1.0,  # Manual matches have 100% confidence
                    amount_matched=min(invoice.total_amount, transaction.amount),
                    user=request.user,
                    ip_address=request.META.get('REMOTE_ADDR'),
                    metadata={
                        'manual_reconciliation': True,
                        'reconciled_at': timezone.now().isoformat(),
                        'user_agent': request.META.get('HTTP_USER_AGENT', ''),
                        'notes': notes
                    }
                )
                reconciliation_logs.append(log)
                
                # Update invoice status
                invoice.status = 'paid'
                invoice.paid_date = timezone.now().date()
                invoice.save()
            
            # Update transaction status
            if len(invoices) == 1 and abs(total_invoice_amount - transaction.amount) < 0.01:
                transaction.status = 'matched'
            else:
                transaction.status = 'partially_matched'
            transaction.save()
            
            return JsonResponse({
                'success': True,
                'message': f'Successfully reconciled {len(invoices)} invoice(s)',
                'reconciliation_logs': len(reconciliation_logs),
                'transaction_status': transaction.status
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


class SuggestMatchesView(LoginRequiredMixin, TemplateView):
    """AJAX view to suggest invoice matches for a transaction."""
    
    def get(self, request, transaction_id, *args, **kwargs):
        """Get ML-powered match suggestions for a transaction."""
        try:
            company = get_user_company(request.user)
            
            transaction = get_object_or_404(
                BankTransaction,
                id=transaction_id,
                company=company,
                status='unmatched'
            )
            
            # Get potential matches based on amount similarity
            amount_tolerance = float(request.GET.get('tolerance', 0.05))  # 5% default tolerance
            min_amount = transaction.amount * (1 - amount_tolerance)
            max_amount = transaction.amount * (1 + amount_tolerance)
            
            potential_invoices = Invoice.objects.filter(
                customer__company=company,
                status__in=['sent', 'overdue', 'pending'],
                total_amount__gte=min_amount,
                total_amount__lte=max_amount
            ).select_related('customer').order_by(
                # Order by closest amount match
                F('total_amount__sub', transaction.amount).__abs__()
            )[:10]
            
            suggestions = []
            for invoice in potential_invoices:
                amount_diff = abs(invoice.total_amount - transaction.amount)
                confidence = max(0, 1 - (amount_diff / transaction.amount))
                
                suggestions.append({
                    'invoice_id': str(invoice.id),
                    'invoice_number': invoice.invoice_number,
                    'customer_name': invoice.customer.name,
                    'amount': float(invoice.total_amount),
                    'due_date': invoice.due_date.isoformat(),
                    'status': invoice.status,
                    'confidence': round(confidence, 3),
                    'amount_difference': float(amount_diff)
                })
            
            return JsonResponse({
                'suggestions': suggestions,
                'transaction_amount': float(transaction.amount),
                'tolerance': amount_tolerance
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


class TransactionSearchView(LoginRequiredMixin, TemplateView):
    """AJAX view for real-time transaction and invoice searching."""
    
    def get(self, request, *args, **kwargs):
        """Search transactions and invoices with real-time filtering."""
        try:
            company = get_user_company(request.user)
            search_query = request.GET.get('q', '').strip()
            search_type = request.GET.get('type', 'both')  # 'transactions', 'invoices', or 'both'
            limit = int(request.GET.get('limit', 50))
            
            results = {}
            
            if search_type in ['transactions', 'both']:
                transactions = BankTransaction.objects.filter(
                    company=company,
                    status='unmatched'
                )
                
                if search_query:
                    transactions = transactions.filter(
                        Q(description__icontains=search_query) |
                        Q(reference_number__icontains=search_query) |
                        Q(bank_reference__icontains=search_query) |
                        Q(amount__icontains=search_query.replace('$', '').replace(',', ''))
                    )
                
                transactions = transactions.order_by('-transaction_date')[:limit]
                
                results['transactions'] = [
                    {
                        'id': str(t.id),
                        'date': t.transaction_date.isoformat(),
                        'description': t.description,
                        'amount': float(t.amount),
                        'reference': t.reference_number or '',
                        'type': t.transaction_type or 'N/A'
                    }
                    for t in transactions
                ]
            
            if search_type in ['invoices', 'both']:
                invoices = Invoice.objects.filter(
                    customer__company=company,
                    status__in=['sent', 'overdue', 'pending']
                ).select_related('customer')
                
                if search_query:
                    invoices = invoices.filter(
                        Q(invoice_number__icontains=search_query) |
                        Q(customer__name__icontains=search_query) |
                        Q(description__icontains=search_query) |
                        Q(total_amount__icontains=search_query.replace('$', '').replace(',', ''))
                    )
                
                invoices = invoices.order_by('-due_date')[:limit]
                
                results['invoices'] = [
                    {
                        'id': str(i.id),
                        'number': i.invoice_number,
                        'customer': i.customer.name,
                        'amount': float(i.total_amount),
                        'due_date': i.due_date.isoformat(),
                        'status': i.status
                    }
                    for i in invoices
                ]
            
            return JsonResponse(results)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


class ReportsView(LoginRequiredMixin, TemplateView):
    """Reports and analytics view."""
    template_name = 'reconciliation/reports.html'
    
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
