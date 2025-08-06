from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
import uuid

class TimestampedModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

class Company(TimestampedModel):
    INDUSTRY_CHOICES = [
        ('tech', 'Technology'),
        ('finance', 'Finance'),
        ('healthcare', 'Healthcare'),
        ('retail', 'Retail'),
        ('manufacturing', 'Manufacturing'),
        ('other', 'Other'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, unique=True)
    industry = models.CharField(max_length=50, choices=INDUSTRY_CHOICES, default='other')
    registration_number = models.CharField(max_length=100, blank=True, null=True)
    contact_email = models.EmailField()
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name_plural = "Companies"
        db_table = 'core_company'

    def __str__(self):
        return self.name

class UserProfile(TimestampedModel):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='users')
    employee_id = models.CharField(max_length=50, blank=True, null=True)
    department = models.CharField(max_length=100, blank=True, null=True)
    is_admin = models.BooleanField(default=False)

    class Meta:
        unique_together = ['user', 'company']
        db_table = 'core_userprofile'

    def __str__(self):
        return f"{self.user.get_full_name()} - {self.company.name}"

class Customer(TimestampedModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='customers')
    name = models.CharField(max_length=255)
    email = models.EmailField(blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    customer_code = models.CharField(max_length=50, blank=True, null=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        unique_together = ['company', 'customer_code']
        db_table = 'core_customer'

    def __str__(self):
        return f"{self.name} ({self.company.name})"

class Invoice(TimestampedModel):
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('sent', 'Sent'),
        ('paid', 'Paid'),
        ('overdue', 'Overdue'),
        ('cancelled', 'Cancelled'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='invoices')
    invoice_number = models.CharField(max_length=100)
    amount_due = models.DecimalField(max_digits=15, decimal_places=2)
    tax_amount = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    total_amount = models.DecimalField(max_digits=15, decimal_places=2)
    issue_date = models.DateField()
    due_date = models.DateField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    description = models.TextField(blank=True, null=True)
    reference_number = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        unique_together = ['customer', 'invoice_number']
        db_table = 'core_invoice'
        indexes = [
            models.Index(fields=['customer', 'status']),
            models.Index(fields=['invoice_number']),
            models.Index(fields=['due_date']),
        ]

    def clean(self):
        if self.total_amount != self.amount_due + self.tax_amount:
            raise ValidationError('Total amount must equal amount due plus tax amount')

    def __str__(self):
        return f"Invoice {self.invoice_number} - {self.customer.name}"

class AuditLog(TimestampedModel):
    ACTION_CHOICES = [
        ('create', 'Create'),
        ('update', 'Update'),
        ('delete', 'Delete'),
        ('reconcile', 'Reconcile'),
        ('unreoncile', 'Unreconcile'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    model_name = models.CharField(max_length=100)
    object_id = models.CharField(max_length=100)
    changes = models.JSONField(default=dict)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True, null=True)

    class Meta:
        db_table = 'core_auditlog'
        indexes = [
            models.Index(fields=['company', 'created_at']),
            models.Index(fields=['user', 'action']),
            models.Index(fields=['model_name', 'object_id']),
        ]

    def __str__(self):
        return f"{self.action} {self.model_name} by {self.user}"
