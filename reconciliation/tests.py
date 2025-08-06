from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APITestCase
from rest_framework import status
from decimal import Decimal
from datetime import date

from core.models import Company, UserProfile, Customer, Invoice
from .models import BankTransaction, ReconciliationLog, FileUploadStatus

class BankTransactionModelTest(TestCase):

    def setUp(self):
        self.company = Company.objects.create(
            name="Test Company",
            industry="tech",
            contact_email="test@company.com"
        )

        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123"
        )

        self.profile = UserProfile.objects.create(
            user=self.user,
            company=self.company
        )

        self.file_upload = FileUploadStatus.objects.create(
            filename="test.csv",
            original_filename="test.csv",
            file_path="/tmp/test.csv",
            file_size=1024,
            user=self.user,
            company=self.company
        )

    def test_transaction_creation(self):
        transaction = BankTransaction.objects.create(
            company=self.company,
            transaction_date=date.today(),
            description="Test transaction",
            amount=Decimal("1000.00"),
            reference_number="REF001",
            file_upload=self.file_upload
        )

        self.assertEqual(transaction.company, self.company)
        self.assertEqual(transaction.amount, Decimal("1000.00"))
        self.assertEqual(transaction.status, "unmatched")

    def test_transaction_str_representation(self):
        transaction = BankTransaction.objects.create(
            company=self.company,
            transaction_date=date.today(),
            description="Test transaction description",
            amount=Decimal("1000.00"),
            file_upload=self.file_upload
        )

        expected = f"{transaction.transaction_date} - Test transaction description - 1000.00"
        self.assertEqual(str(transaction), expected)

class ReconciliationAPITest(APITestCase):

    def setUp(self):
        self.company = Company.objects.create(
            name="Test Company",
            industry="tech",
            contact_email="test@company.com"
        )

        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123"
        )

        self.profile = UserProfile.objects.create(
            user=self.user,
            company=self.company
        )

        self.customer = Customer.objects.create(
            company=self.company,
            name="Test Customer",
            email="customer@test.com"
        )

        self.invoice = Invoice.objects.create(
            customer=self.customer,
            invoice_number="INV001",
            amount_due=Decimal("1000.00"),
            tax_amount=Decimal("0.00"),
            total_amount=Decimal("1000.00"),
            issue_date=date.today(),
            due_date=date.today()
        )

        self.file_upload = FileUploadStatus.objects.create(
            filename="test.csv",
            original_filename="test.csv",
            file_path="/tmp/test.csv",
            file_size=1024,
            user=self.user,
            company=self.company
        )

        self.transaction = BankTransaction.objects.create(
            company=self.company,
            transaction_date=date.today(),
            description="Payment from Test Customer",
            amount=Decimal("1000.00"),
            reference_number="REF001",
            file_upload=self.file_upload
        )

    def test_authentication_required(self):
        response = self.client.get('/api/v1/bank/transactions/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_list_transactions(self):
        self.client.force_authenticate(user=self.user)
        response = self.client.get('/api/v1/bank/transactions/')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)

    def test_manual_reconciliation(self):
        self.client.force_authenticate(user=self.user)

        data = {
            'transaction_id': str(self.transaction.id),
            'invoice_ids': [str(self.invoice.id)],
            'amounts': [str(self.transaction.amount)],
            'notes': 'Manual reconciliation test'
        }

        response = self.client.post(
            f'/api/v1/bank/reconcile/{self.transaction.id}/',
            data,
            format='json'
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertTrue(
            ReconciliationLog.objects.filter(
                transaction=self.transaction,
                invoice=self.invoice
            ).exists()
        )

        self.transaction.refresh_from_db()
        self.assertEqual(self.transaction.status, 'matched')

class ReconciliationLogModelTest(TestCase):

    def setUp(self):
        self.company = Company.objects.create(
            name="Test Company",
            industry="tech",
            contact_email="test@company.com"
        )

        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123"
        )

        self.customer = Customer.objects.create(
            company=self.company,
            name="Test Customer",
            email="customer@test.com"
        )

        self.invoice = Invoice.objects.create(
            customer=self.customer,
            invoice_number="INV001",
            amount_due=Decimal("1000.00"),
            tax_amount=Decimal("0.00"),
            total_amount=Decimal("1000.00"),
            issue_date=date.today(),
            due_date=date.today()
        )

        self.file_upload = FileUploadStatus.objects.create(
            filename="test.csv",
            original_filename="test.csv",
            file_path="/tmp/test.csv",
            file_size=1024,
            user=self.user,
            company=self.company
        )

        self.transaction = BankTransaction.objects.create(
            company=self.company,
            transaction_date=date.today(),
            description="Payment from Test Customer",
            amount=Decimal("1000.00"),
            reference_number="REF001",
            file_upload=self.file_upload
        )

    def test_reconciliation_log_creation(self):
        log = ReconciliationLog.objects.create(
            transaction=self.transaction,
            invoice=self.invoice,
            matched_by='manual',
            amount_matched=Decimal("1000.00"),
            user=self.user
        )

        self.assertEqual(log.transaction, self.transaction)
        self.assertEqual(log.invoice, self.invoice)
        self.assertEqual(log.matched_by, 'manual')
        self.assertTrue(log.is_active)

    def test_reconciliation_log_validation(self):
        from django.core.exceptions import ValidationError

        log = ReconciliationLog(
            transaction=self.transaction,
            invoice=self.invoice,
            matched_by='manual',
            amount_matched=Decimal("2000.00"),
            user=self.user
        )

        with self.assertRaises(ValidationError):
            log.clean()
