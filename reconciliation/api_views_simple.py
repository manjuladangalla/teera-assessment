"""
Simplified API views for debugging reconciliation issues.
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json
import logging

from .models import BankTransaction, ReconciliationLog
from .permissions import IsCompanyMember
from core.models import Invoice

logger = logging.getLogger(__name__)


class DebugReconciliationView:
    """Debug view for reconciliation API issues."""
    
    @staticmethod
    @csrf_exempt
    def debug_reconcile(request, transaction_id):
        """Debug endpoint to test reconciliation data."""
        if request.method != 'POST':
            return JsonResponse({'error': 'Method not allowed'}, status=405)
        
        try:
            # Log the request details
            logger.info(f"Debug reconcile request for transaction: {transaction_id}")
            logger.info(f"Content-Type: {request.content_type}")
            logger.info(f"Request body: {request.body}")
            
            # Parse the request data
            if request.content_type == 'application/json':
                data = json.loads(request.body)
            else:
                data = request.POST.dict()
            
            logger.info(f"Parsed data: {data}")
            
            # Check if user is authenticated
            if not request.user.is_authenticated:
                return JsonResponse({'error': 'Authentication required'}, status=401)
            
            # Check if transaction exists
            try:
                transaction = BankTransaction.objects.get(
                    id=transaction_id,
                    company=request.user.profile.company
                )
            except BankTransaction.DoesNotExist:
                return JsonResponse({'error': 'Transaction not found'}, status=404)
            
            # Validate required fields
            required_fields = ['invoice_ids']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return JsonResponse({
                    'error': 'Missing required fields',
                    'missing_fields': missing_fields,
                    'received_data': data
                }, status=400)
            
            # Check if invoices exist
            invoice_ids = data.get('invoice_ids', [])
            if not invoice_ids:
                return JsonResponse({'error': 'At least one invoice ID required'}, status=400)
            
            invoices = Invoice.objects.filter(
                id__in=invoice_ids,
                customer__company=request.user.profile.company
            )
            
            found_invoice_ids = [str(inv.id) for inv in invoices]
            missing_invoices = [inv_id for inv_id in invoice_ids if inv_id not in found_invoice_ids]
            
            if missing_invoices:
                return JsonResponse({
                    'error': 'Some invoices not found',
                    'missing_invoices': missing_invoices,
                    'found_invoices': found_invoice_ids
                }, status=400)
            
            return JsonResponse({
                'success': True,
                'message': 'Debug successful',
                'transaction_id': str(transaction.id),
                'transaction_amount': float(transaction.amount),
                'invoice_count': len(invoices),
                'invoice_amounts': [float(inv.total_amount) for inv in invoices],
                'total_invoice_amount': sum(float(inv.total_amount) for inv in invoices),
                'user': request.user.username,
                'company': request.user.profile.company.name if hasattr(request.user, 'profile') else 'Unknown'
            })
            
        except json.JSONDecodeError as e:
            return JsonResponse({
                'error': 'Invalid JSON data',
                'detail': str(e),
                'raw_body': request.body.decode('utf-8')
            }, status=400)
        except Exception as e:
            logger.error(f"Debug reconcile error: {e}")
            return JsonResponse({
                'error': 'Internal server error',
                'detail': str(e)
            }, status=500)


@csrf_exempt
def simple_reconcile(request, transaction_id):
    """Simple reconciliation endpoint."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        # Parse data
        data = json.loads(request.body) if request.body else {}
        
        # Get transaction
        transaction = BankTransaction.objects.get(
            id=transaction_id,
            company=request.user.profile.company
        )
        
        # Get invoices
        invoice_ids = data.get('invoice_ids', [])
        invoices = Invoice.objects.filter(
            id__in=invoice_ids,
            customer__company=request.user.profile.company
        )
        
        # Create reconciliation logs
        logs_created = 0
        for invoice in invoices:
            log = ReconciliationLog.objects.create(
                transaction=transaction,
                invoice=invoice,
                matched_by='manual',
                confidence_score=1.0,
                amount_matched=min(invoice.total_amount, transaction.amount),
                user=request.user,
                metadata={
                    'notes': data.get('notes', ''),
                    'simplified_api': True
                }
            )
            logs_created += 1
            
            # Update invoice status
            invoice.status = 'paid'
            invoice.save()
        
        # Update transaction status
        transaction.status = 'matched' if logs_created == 1 else 'partially_matched'
        transaction.save()
        
        return JsonResponse({
            'success': True,
            'message': f'Reconciled {logs_created} invoice(s)',
            'transaction_status': transaction.status,
            'logs_created': logs_created
        })
        
    except BankTransaction.DoesNotExist:
        return JsonResponse({'error': 'Transaction not found'}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
