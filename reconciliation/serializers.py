from rest_framework import serializers
from django.contrib.auth.models import User
from core.models import Company, Customer, Invoice
from reconciliation.models import (
    BankTransaction, ReconciliationLog, FileUploadStatus,
    MatchingRule, ReconciliationSummary
)


class CompanySerializer(serializers.ModelSerializer):
    class Meta:
        model = Company
        fields = ['id', 'name', 'industry', 'contact_email', 'is_active', 'created_at']
        read_only_fields = ['id', 'created_at']


class CustomerSerializer(serializers.ModelSerializer):
    company_name = serializers.CharField(source='company.name', read_only=True)
    
    class Meta:
        model = Customer
        fields = [
            'id', 'name', 'email', 'phone', 'address', 'customer_code',
            'is_active', 'company_name', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class InvoiceSerializer(serializers.ModelSerializer):
    customer_name = serializers.CharField(source='customer.name', read_only=True)
    company_name = serializers.CharField(source='customer.company.name', read_only=True)
    
    class Meta:
        model = Invoice
        fields = [
            'id', 'customer', 'customer_name', 'company_name', 'invoice_number',
            'amount_due', 'tax_amount', 'total_amount', 'issue_date', 'due_date',
            'status', 'description', 'reference_number', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class BankTransactionSerializer(serializers.ModelSerializer):
    company_name = serializers.CharField(source='company.name', read_only=True)
    file_upload_name = serializers.CharField(source='file_upload.original_filename', read_only=True)
    
    class Meta:
        model = BankTransaction
        fields = [
            'id', 'company_name', 'transaction_date', 'description', 'amount',
            'reference_number', 'bank_reference', 'transaction_type', 'balance',
            'status', 'file_upload_name', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class BankTransactionDetailSerializer(BankTransactionSerializer):
    reconciliation_logs = serializers.SerializerMethodField()
    
    class Meta(BankTransactionSerializer.Meta):
        fields = BankTransactionSerializer.Meta.fields + ['reconciliation_logs', 'raw_data']
    
    def get_reconciliation_logs(self, obj):
        logs = obj.reconciliation_logs.filter(is_active=True)
        return ReconciliationLogSerializer(logs, many=True).data


class ReconciliationLogSerializer(serializers.ModelSerializer):
    transaction_description = serializers.CharField(source='transaction.description', read_only=True)
    invoice_number = serializers.CharField(source='invoice.invoice_number', read_only=True)
    user_name = serializers.CharField(source='user.get_full_name', read_only=True)
    
    class Meta:
        model = ReconciliationLog
        fields = [
            'id', 'transaction', 'invoice', 'transaction_description', 'invoice_number',
            'matched_by', 'confidence_score', 'amount_matched', 'user', 'user_name',
            'metadata', 'is_active', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class ManualReconciliationSerializer(serializers.Serializer):
    transaction_id = serializers.UUIDField()
    invoice_ids = serializers.ListField(
        child=serializers.UUIDField(),
        min_length=1,
        max_length=10
    )
    amounts = serializers.ListField(
        child=serializers.DecimalField(max_digits=15, decimal_places=2),
        min_length=1,
        max_length=10
    )
    notes = serializers.CharField(max_length=1000, required=False, allow_blank=True)
    
    def validate(self, data):
        if len(data['invoice_ids']) != len(data['amounts']):
            raise serializers.ValidationError(
                "Number of invoice IDs must match number of amounts"
            )
        return data


class FileUploadStatusSerializer(serializers.ModelSerializer):
    user_name = serializers.CharField(source='user.get_full_name', read_only=True)
    company_name = serializers.CharField(source='company.name', read_only=True)
    progress_percentage = serializers.SerializerMethodField()
    
    class Meta:
        model = FileUploadStatus
        fields = [
            'id', 'filename', 'original_filename', 'file_size', 'user_name',
            'company_name', 'status', 'total_records', 'processed_records',
            'failed_records', 'progress_percentage', 'error_log', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']
    
    def get_progress_percentage(self, obj):
        if obj.total_records == 0:
            return 0
        return round((obj.processed_records / obj.total_records) * 100, 2)


class MatchingRuleSerializer(serializers.ModelSerializer):
    created_by_name = serializers.CharField(source='created_by.get_full_name', read_only=True)
    
    class Meta:
        model = MatchingRule
        fields = [
            'id', 'name', 'rule_type', 'conditions', 'priority', 'is_active',
            'created_by_name', 'success_count', 'created_at'
        ]
        read_only_fields = ['id', 'success_count', 'created_at']


class ReconciliationSummarySerializer(serializers.ModelSerializer):
    company_name = serializers.CharField(source='company.name', read_only=True)
    match_percentage = serializers.SerializerMethodField()
    
    class Meta:
        model = ReconciliationSummary
        fields = [
            'id', 'company_name', 'period_start', 'period_end',
            'total_transactions', 'matched_transactions', 'unmatched_transactions',
            'total_amount', 'matched_amount', 'ml_matches', 'manual_matches',
            'average_confidence', 'match_percentage', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']
    
    def get_match_percentage(self, obj):
        if obj.total_transactions == 0:
            return 0
        return round((obj.matched_transactions / obj.total_transactions) * 100, 2)


class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    
    def validate_file(self, value):
        # Check file extension
        allowed_extensions = ['.csv', '.xlsx', '.xls']
        if not any(value.name.lower().endswith(ext) for ext in allowed_extensions):
            raise serializers.ValidationError(
                f"File must be one of: {', '.join(allowed_extensions)}"
            )
        
        # Check file size (50MB limit)
        if value.size > 50 * 1024 * 1024:
            raise serializers.ValidationError("File size cannot exceed 50MB")
        
        return value
