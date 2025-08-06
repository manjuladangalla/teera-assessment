from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from core.models import TimestampedModel, Company, Invoice, Customer
import uuid


class BankTransaction(TimestampedModel):
    """Bank transaction model for uploaded bank statements."""
    STATUS_CHOICES = [
        ('unmatched', 'Unmatched'),
        ('matched', 'Matched'),
        ('partially_matched', 'Partially Matched'),
        ('disputed', 'Disputed'),
        ('ignored', 'Ignored'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='bank_transactions')
    transaction_date = models.DateField()
    description = models.TextField()
    amount = models.DecimalField(max_digits=15, decimal_places=2)
    reference_number = models.CharField(max_length=255, blank=True, null=True)
    bank_reference = models.CharField(max_length=255, blank=True, null=True)
    transaction_type = models.CharField(max_length=50, blank=True, null=True)  # debit/credit
    balance = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='unmatched')
    file_upload = models.ForeignKey('FileUploadStatus', on_delete=models.CASCADE, related_name='transactions')
    raw_data = models.JSONField(default=dict)  # Store original CSV/Excel row data
    
    class Meta:
        db_table = 'reconciliation_banktransaction'
        indexes = [
            models.Index(fields=['company', 'status']),
            models.Index(fields=['transaction_date']),
            models.Index(fields=['reference_number']),
            models.Index(fields=['amount']),
        ]
    
    def __str__(self):
        return f"{self.transaction_date} - {self.description[:50]} - {self.amount}"


class ReconciliationLog(TimestampedModel):
    """Log of reconciliation matches between transactions and invoices."""
    MATCH_TYPE_CHOICES = [
        ('manual', 'Manual'),
        ('ml_auto', 'ML Automatic'),
        ('rule_based', 'Rule Based'),
        ('partial', 'Partial Match'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    transaction = models.ForeignKey(BankTransaction, on_delete=models.CASCADE, related_name='reconciliation_logs')
    invoice = models.ForeignKey(Invoice, on_delete=models.CASCADE, related_name='reconciliation_logs')
    matched_by = models.CharField(max_length=20, choices=MATCH_TYPE_CHOICES)
    confidence_score = models.FloatField(null=True, blank=True)  # ML confidence score (0-1)
    amount_matched = models.DecimalField(max_digits=15, decimal_places=2)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    metadata = models.JSONField(default=dict)  # Store ML features, notes, etc.
    is_active = models.BooleanField(default=True)  # For soft deletes/reversals
    
    class Meta:
        db_table = 'reconciliation_reconciliationlog'
        indexes = [
            models.Index(fields=['transaction', 'is_active']),
            models.Index(fields=['invoice', 'is_active']),
            models.Index(fields=['matched_by', 'confidence_score']),
            models.Index(fields=['created_at']),
        ]
    
    def clean(self):
        if self.amount_matched > self.transaction.amount:
            raise ValidationError('Matched amount cannot exceed transaction amount')
        if self.amount_matched > self.invoice.total_amount:
            raise ValidationError('Matched amount cannot exceed invoice amount')
    
    def __str__(self):
        return f"Match: {self.transaction.id} -> {self.invoice.invoice_number}"


class FileUploadStatus(TimestampedModel):
    """Track file upload and processing status."""
    STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename = models.CharField(max_length=255)
    original_filename = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    file_size = models.PositiveIntegerField()  # in bytes
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploaded')
    total_records = models.PositiveIntegerField(default=0)
    processed_records = models.PositiveIntegerField(default=0)
    failed_records = models.PositiveIntegerField(default=0)
    error_log = models.TextField(blank=True, null=True)
    processing_metadata = models.JSONField(default=dict)
    
    class Meta:
        db_table = 'reconciliation_fileuploadstatus'
        indexes = [
            models.Index(fields=['company', 'status']),
            models.Index(fields=['user', 'created_at']),
        ]
    
    def __str__(self):
        return f"{self.original_filename} - {self.status}"


class MatchingRule(TimestampedModel):
    """Custom matching rules for automatic reconciliation."""
    RULE_TYPE_CHOICES = [
        ('exact_reference', 'Exact Reference Match'),
        ('amount_date', 'Amount and Date Match'),
        ('description_pattern', 'Description Pattern Match'),
        ('customer_reference', 'Customer Reference Match'),
        ('custom', 'Custom Rule'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='matching_rules')
    name = models.CharField(max_length=255)
    rule_type = models.CharField(max_length=30, choices=RULE_TYPE_CHOICES)
    conditions = models.JSONField()  # Store rule conditions as JSON
    priority = models.PositiveIntegerField(default=100)  # Lower number = higher priority
    is_active = models.BooleanField(default=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    success_count = models.PositiveIntegerField(default=0)
    
    class Meta:
        db_table = 'reconciliation_matchingrule'
        ordering = ['priority', 'name']
        indexes = [
            models.Index(fields=['company', 'is_active', 'priority']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.company.name})"


class MLModelVersion(TimestampedModel):
    """Track ML model versions and performance."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='ml_models')
    version = models.CharField(max_length=50)
    model_path = models.CharField(max_length=500)
    training_data_count = models.PositiveIntegerField()
    accuracy_score = models.FloatField(null=True, blank=True)
    precision_score = models.FloatField(null=True, blank=True)
    recall_score = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    is_active = models.BooleanField(default=False)
    training_metadata = models.JSONField(default=dict)
    
    class Meta:
        db_table = 'reconciliation_mlmodelversion'
        unique_together = ['company', 'version']
        indexes = [
            models.Index(fields=['company', 'is_active']),
            models.Index(fields=['accuracy_score']),
        ]
    
    def __str__(self):
        return f"Model {self.version} for {self.company.name}"


class ReconciliationSummary(TimestampedModel):
    """Summary statistics for reconciliation reporting."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    period_start = models.DateField()
    period_end = models.DateField()
    total_transactions = models.PositiveIntegerField(default=0)
    matched_transactions = models.PositiveIntegerField(default=0)
    unmatched_transactions = models.PositiveIntegerField(default=0)
    total_amount = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    matched_amount = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    ml_matches = models.PositiveIntegerField(default=0)
    manual_matches = models.PositiveIntegerField(default=0)
    average_confidence = models.FloatField(null=True, blank=True)
    generated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    
    class Meta:
        db_table = 'reconciliation_reconciliationsummary'
        unique_together = ['company', 'period_start', 'period_end']
        indexes = [
            models.Index(fields=['company', 'period_start']),
        ]
    
    def __str__(self):
        return f"Summary {self.period_start} - {self.period_end} ({self.company.name})"
