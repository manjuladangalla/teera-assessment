from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.db import transaction
from django.http import HttpResponse
from django.utils import timezone
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter
import tempfile
import os

from core.models import Company, Invoice
from .models import (
    BankTransaction, ReconciliationLog, FileUploadStatus, MatchingRule,
    ReconciliationSummary
)
from .serializers import (
    BankTransactionSerializer, BankTransactionDetailSerializer,
    ReconciliationLogSerializer, FileUploadStatusSerializer, 
    MatchingRuleSerializer, FileUploadSerializer, ReconciliationSummarySerializer
)
# Temporarily comment out tasks for minimal setup
# from .tasks import process_bank_statement_file, generate_reconciliation_report
from .utils import get_user_company, create_audit_log
from .permissions import IsCompanyMember


class BankTransactionViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for bank transactions."""
    permission_classes = [permissions.IsAuthenticated, IsCompanyMember]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['status', 'transaction_type', 'transaction_date']
    search_fields = ['description', 'reference_number', 'bank_reference']
    ordering_fields = ['transaction_date', 'amount', 'created_at']
    ordering = ['-transaction_date']
    
    def get_queryset(self):
        company = get_user_company(self.request.user)
        return BankTransaction.objects.filter(company=company).select_related(
            'company', 'file_upload'
        )
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return BankTransactionDetailSerializer
        return BankTransactionSerializer
    
    @action(detail=False, methods=['get'])
    def unmatched(self, request):
        """Get unmatched transactions."""
        queryset = self.get_queryset().filter(status='unmatched')
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def reconcile(self, request, pk=None):
        """Manually reconcile a transaction with invoices."""
        transaction = self.get_object()
        serializer = ManualReconciliationSerializer(data=request.data)
        
        if serializer.is_valid():
            company = get_user_company(request.user)
            
            with transaction.atomic():
                # Create reconciliation logs
                for invoice_id, amount in zip(
                    serializer.validated_data['invoice_ids'],
                    serializer.validated_data['amounts']
                ):
                    invoice = get_object_or_404(
                        Invoice, id=invoice_id, customer__company=company
                    )
                    
                    reconciliation_log = ReconciliationLog.objects.create(
                        transaction=transaction,
                        invoice=invoice,
                        matched_by='manual',
                        amount_matched=amount,
                        user=request.user,
                        ip_address=self.get_client_ip(request),
                        metadata={
                            'notes': serializer.validated_data.get('notes', ''),
                            'manual_reconciliation': True
                        }
                    )
                
                # Update transaction status
                total_matched = sum(serializer.validated_data['amounts'])
                if total_matched >= transaction.amount:
                    transaction.status = 'matched'
                else:
                    transaction.status = 'partially_matched'
                transaction.save()
                
                # Create audit log
                create_audit_log(
                    user=request.user,
                    company=company,
                    action='reconcile',
                    model_name='BankTransaction',
                    object_id=str(transaction.id),
                    changes={
                        'status': transaction.status,
                        'reconciled_invoices': serializer.validated_data['invoice_ids'],
                        'amounts': serializer.validated_data['amounts']
                    },
                    ip_address=self.get_client_ip(request)
                )
            
            return Response({'message': 'Transaction reconciled successfully'})
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def get_client_ip(self, request):
        """Get client IP address."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


class ReconciliationLogViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for reconciliation logs."""
    serializer_class = ReconciliationLogSerializer
    permission_classes = [permissions.IsAuthenticated, IsCompanyMember]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['matched_by', 'is_active']
    search_fields = ['transaction__description', 'invoice__invoice_number']
    ordering_fields = ['created_at', 'confidence_score']
    ordering = ['-created_at']
    
    def get_queryset(self):
        company = get_user_company(self.request.user)
        return ReconciliationLog.objects.filter(
            transaction__company=company
        ).select_related('transaction', 'invoice', 'user')


class FileUploadViewSet(viewsets.ModelViewSet):
    """ViewSet for file uploads."""
    serializer_class = FileUploadStatusSerializer
    permission_classes = [permissions.IsAuthenticated, IsCompanyMember]
    filter_backends = [DjangoFilterBackend, OrderingFilter]
    filterset_fields = ['status']
    ordering_fields = ['created_at', 'total_records']
    ordering = ['-created_at']
    
    def get_queryset(self):
        company = get_user_company(self.request.user)
        return FileUploadStatus.objects.filter(company=company)
    
    @action(detail=False, methods=['post'])
    def upload(self, request):
        """Upload and process bank statement file."""
        serializer = FileUploadSerializer(data=request.data)
        
        if serializer.is_valid():
            uploaded_file = serializer.validated_data['file']
            company = get_user_company(request.user)
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=os.path.splitext(uploaded_file.name)[1]
            ) as temp_file:
                for chunk in uploaded_file.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            # Create file upload status record
            file_upload = FileUploadStatus.objects.create(
                filename=os.path.basename(temp_file_path),
                original_filename=uploaded_file.name,
                file_path=temp_file_path,
                file_size=uploaded_file.size,
                user=request.user,
                company=company,
                status='uploaded'
            )
            
            # Queue file processing task
            # Temporarily disabled for minimal setup
            # process_bank_statement_file.delay(file_upload.id)
            
            # Process synchronously for now
            return Response({
                'message': 'File uploaded successfully. Processing disabled in minimal mode.',
                'file_upload_id': file_upload.id
            }, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class MatchingRuleViewSet(viewsets.ModelViewSet):
    """ViewSet for matching rules."""
    serializer_class = MatchingRuleSerializer
    permission_classes = [permissions.IsAuthenticated, IsCompanyMember]
    filter_backends = [DjangoFilterBackend, OrderingFilter]
    filterset_fields = ['rule_type', 'is_active']
    ordering_fields = ['priority', 'name', 'success_count']
    ordering = ['priority']
    
    def get_queryset(self):
        # Handle schema generation gracefully
        if getattr(self, 'swagger_fake_view', False):
            return MatchingRule.objects.none()
            
        if not self.request.user.is_authenticated:
            return MatchingRule.objects.none()
            
        company = get_user_company(self.request.user)
        return MatchingRule.objects.filter(company=company)
    
    def perform_create(self, serializer):
        company = get_user_company(self.request.user)
        serializer.save(company=company, created_by=self.request.user)


class ReconciliationSummaryViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for reconciliation summaries."""
    serializer_class = ReconciliationSummarySerializer
    permission_classes = [permissions.IsAuthenticated, IsCompanyMember]
    filter_backends = [DjangoFilterBackend, OrderingFilter]
    filterset_fields = ['period_start', 'period_end']
    ordering_fields = ['period_start', 'created_at']
    ordering = ['-period_start']
    
    def get_queryset(self):
        # Handle schema generation gracefully
        if getattr(self, 'swagger_fake_view', False):
            return ReconciliationSummary.objects.none()
            
        if not self.request.user.is_authenticated:
            return ReconciliationSummary.objects.none()
            
        company = get_user_company(self.request.user)
        return ReconciliationSummary.objects.filter(company=company)
    
    @action(detail=True, methods=['get'])
    def export_pdf(self, request, pk=None):
        """Export reconciliation summary as PDF."""
        summary = self.get_object()
        
        # Queue report generation task
        task = generate_reconciliation_report.delay(
            summary.id, 'pdf', request.user.id
        )
        
        return Response({
            'message': 'Report generation started',
            'task_id': task.id
        })
    
    @action(detail=True, methods=['get'])
    def export_xlsx(self, request, pk=None):
        """Export reconciliation summary as XLSX."""
        summary = self.get_object()
        
        # Queue report generation task
        task = generate_reconciliation_report.delay(
            summary.id, 'xlsx', request.user.id
        )
        
        return Response({
            'message': 'Report generation started',
            'task_id': task.id
        })
