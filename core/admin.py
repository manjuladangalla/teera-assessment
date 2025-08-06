from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from .models import Company, UserProfile, Customer, Invoice, AuditLog


class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Profile'


class UserAdmin(BaseUserAdmin):
    inlines = (UserProfileInline,)


@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ['name', 'industry', 'contact_email', 'is_active', 'created_at']
    list_filter = ['industry', 'is_active', 'created_at']
    search_fields = ['name', 'contact_email', 'registration_number']
    readonly_fields = ['id', 'created_at', 'updated_at']


@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display = ['name', 'company', 'email', 'customer_code', 'is_active']
    list_filter = ['company', 'is_active', 'created_at']
    search_fields = ['name', 'email', 'customer_code']
    readonly_fields = ['id', 'created_at', 'updated_at']


@admin.register(Invoice)
class InvoiceAdmin(admin.ModelAdmin):
    list_display = ['invoice_number', 'customer', 'total_amount', 'status', 'due_date']
    list_filter = ['status', 'issue_date', 'due_date', 'customer__company']
    search_fields = ['invoice_number', 'customer__name', 'reference_number']
    readonly_fields = ['id', 'created_at', 'updated_at']
    date_hierarchy = 'issue_date'


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ['action', 'model_name', 'user', 'company', 'created_at']
    list_filter = ['action', 'model_name', 'company', 'created_at']
    search_fields = ['object_id', 'user__username']
    readonly_fields = ['id', 'created_at', 'updated_at']
    date_hierarchy = 'created_at'


# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)
