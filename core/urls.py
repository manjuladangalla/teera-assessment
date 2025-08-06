from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework import viewsets, permissions
from .models import Company, Customer, Invoice
from .serializers import CompanySerializer, CustomerSerializer, InvoiceSerializer

class CompanyViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Company.objects.all()
    serializer_class = CompanySerializer
    permission_classes = [permissions.IsAuthenticated]

class CustomerViewSet(viewsets.ModelViewSet):
    serializer_class = CustomerSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user_company = self.request.user.profile.company
        return Customer.objects.filter(company=user_company)

class InvoiceViewSet(viewsets.ModelViewSet):
    serializer_class = InvoiceSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user_company = self.request.user.profile.company
        return Invoice.objects.filter(customer__company=user_company)

router = DefaultRouter()
router.register(r'companies', CompanyViewSet)
router.register(r'customers', CustomerViewSet, basename='customer')
router.register(r'invoices', InvoiceViewSet, basename='invoice')

app_name = 'core'

urlpatterns = [
    path('', include(router.urls)),
]
