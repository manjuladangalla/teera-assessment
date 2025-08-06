from rest_framework import serializers
from core.models import Company, Customer, Invoice


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
