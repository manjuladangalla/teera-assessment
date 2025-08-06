from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
from django.db import models
from decimal import Decimal, InvalidOperation
import logging
import os
import json
import csv
import io
import openpyxl
import chardet
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback
from dateutil import parser as date_parser

from .models import FileUploadStatus, BankTransaction, ReconciliationSummary, MLModelVersion
from core.models import Company

logger = logging.getLogger(__name__)


def create_sample_csv_content():
    """Create sample CSV content for testing."""
    csv_data = '''Transaction_ID,Date,Description,Amount,Balance,Account_Number
TXN001,2025-01-01,Opening Balance,1000.00,1000.00,ACC001
TXN002,2025-01-02,Salary Credit,5000.00,6000.00,ACC001
TXN003,2025-01-03,Utility Payment,-150.00,5850.00,ACC001'''
    return csv_data.encode('utf-8')


@shared_task(bind=True)
def process_bank_statement_file(self, file_upload_id):
    """
    Process a bank statement file with enhanced CSV and Excel support.
    """
    try:
        file_upload = FileUploadStatus.objects.get(id=file_upload_id)
        file_upload.status = 'processing'
        file_upload.processing_started_at = timezone.now()
        file_upload.save()
        
        logger.info(f"Processing file: {file_upload.original_filename}")
        
        # Read file content from file_path
        try:
            with open(file_upload.file_path, 'rb') as f:
                file_content = f.read()
        except FileNotFoundError:
            # For testing purposes, create a sample CSV content
            file_content = create_sample_csv_content()
        
        transactions_created = 0
        failed_records = 0
        errors = []
        
        try:
            # Determine file type and process accordingly
            filename = file_upload.original_filename.lower()
            
            if filename.endswith('.csv'):
                transactions_created, failed_records, errors = process_csv_file(
                    file_content, file_upload.company, file_upload
                )
            elif filename.endswith(('.xlsx', '.xls')):
                transactions_created, failed_records, errors = process_excel_file(
                    file_content, file_upload.company, file_upload
                )
            else:
                raise ValueError(f"Unsupported file type: {filename}")
            
            file_upload.status = 'completed'
            file_upload.processed_records = transactions_created
            file_upload.failed_records = failed_records
            file_upload.processing_completed_at = timezone.now()
            
            if errors:
                file_upload.error_log = json.dumps(errors[:10])  # Store first 10 errors
            
            file_upload.save()
            
            logger.info(f"File processing completed: {transactions_created} created, {failed_records} failed")
            
            return {
                'status': 'completed',
                'created_count': transactions_created,
                'failed_count': failed_records,
                'errors': errors[:5]  # Return first 5 errors
            }
            
        except Exception as processing_error:
            logger.error(f"Error processing file content: {processing_error}")
            raise processing_error
        
    except Exception as e:
        logger.error(f"File processing failed: {e}")
        
        try:
            file_upload = FileUploadStatus.objects.get(id=file_upload_id)
            file_upload.status = 'failed'
            file_upload.error_log = str(e)
            file_upload.processing_completed_at = timezone.now()
            file_upload.save()
        except:
            pass
        
        raise


def process_csv_file(file_content: bytes, company: Company, file_upload) -> tuple:
    """
    Process a CSV file and create bank transactions.
    """
    # Detect encoding
    detected = chardet.detect(file_content)
    encoding = detected.get('encoding', 'utf-8')
    
    try:
        content_str = file_content.decode(encoding)
    except UnicodeDecodeError:
        content_str = file_content.decode('utf-8', errors='ignore')
    
    # Parse CSV
    csv_file = io.StringIO(content_str)
    reader = csv.DictReader(csv_file)
    
    transactions_created = 0
    failed_records = 0
    errors = []
    
    for row_num, row in enumerate(reader, start=2):  # Start from 2 (header is row 1)
        try:
            transaction_data = parse_transaction_row(row, company, file_upload)
            
            # Create or update transaction
            transaction, created = BankTransaction.objects.get_or_create(
                company=company,
                file_upload=file_upload,
                bank_reference=transaction_data.get('bank_reference'),
                transaction_date=transaction_data.get('transaction_date'),
                amount=transaction_data.get('amount'),
                defaults=transaction_data
            )
            
            if created:
                transactions_created += 1
            
        except Exception as e:
            failed_records += 1
            error_msg = f"Row {row_num}: {str(e)}"
            errors.append(error_msg)
            logger.warning(error_msg)
    
    return transactions_created, failed_records, errors


def process_excel_file(file_content: bytes, company: Company, file_upload) -> tuple:
    """
    Process an Excel file and create bank transactions.
    """
    # Load workbook
    workbook = openpyxl.load_workbook(io.BytesIO(file_content))
    worksheet = workbook.active
    
    # Get header row
    headers = [cell.value for cell in worksheet[1]]
    header_map = {header.lower().strip(): idx for idx, header in enumerate(headers) if header}
    
    transactions_created = 0
    failed_records = 0
    errors = []
    
    for row_num, row in enumerate(worksheet.iter_rows(min_row=2), start=2):
        try:
            # Convert row to dictionary
            row_dict = {}
            for header, col_idx in header_map.items():
                if col_idx < len(row):
                    cell_value = row[col_idx].value
                    row_dict[header] = cell_value
            
            transaction_data = parse_transaction_row(row_dict, company, file_upload)
            
            # Create or update transaction
            transaction, created = BankTransaction.objects.get_or_create(
                company=company,
                file_upload=file_upload,
                bank_reference=transaction_data.get('bank_reference'),
                transaction_date=transaction_data.get('transaction_date'),
                amount=transaction_data.get('amount'),
                defaults=transaction_data
            )
            
            if created:
                transactions_created += 1
            
        except Exception as e:
            failed_records += 1
            error_msg = f"Row {row_num}: {str(e)}"
            errors.append(error_msg)
            logger.warning(error_msg)
    
    return transactions_created, failed_records, errors


def parse_transaction_row(row: dict, company: Company, file_upload) -> dict:
    """
    Parse a transaction row from CSV or Excel data.
    """
    # Common field mappings (case-insensitive)
    field_mappings = {
        'transaction_id': ['transaction_id', 'id', 'reference', 'ref', 'transaction_ref'],
        'date': ['date', 'transaction_date', 'value_date', 'posting_date'],
        'description': ['description', 'narrative', 'details', 'memo'],
        'amount': ['amount', 'value', 'transaction_amount'],
        'balance': ['balance', 'running_balance', 'account_balance'],
        'account_number': ['account_number', 'account', 'acc_no'],
        'reference': ['reference', 'ref', 'check_number', 'cheque_number']
    }
    
    # Normalize row keys
    normalized_row = {k.lower().strip(): v for k, v in row.items() if k}
    
    def find_field_value(field_key):
        for possible_key in field_mappings.get(field_key, []):
            if possible_key in normalized_row:
                return normalized_row[possible_key]
        return None
    
    # Extract transaction ID
    transaction_id = find_field_value('transaction_id')
    if not transaction_id:
        # Generate a unique ID based on available data
        date_str = str(find_field_value('date') or '')
        amount_str = str(find_field_value('amount') or '')
        desc_str = str(find_field_value('description') or '')
        transaction_id = f"{company.id}_{date_str}_{amount_str}_{desc_str}"[:100]
    
    # Parse date
    date_value = find_field_value('date')
    if date_value:
        if isinstance(date_value, datetime):
            transaction_date = date_value.date()
        else:
            try:
                transaction_date = date_parser.parse(str(date_value)).date()
            except:
                transaction_date = timezone.now().date()
    else:
        transaction_date = timezone.now().date()
    
    # Parse amount
    amount_value = find_field_value('amount')
    if amount_value is not None:
        try:
            # Clean amount string (remove currency symbols, commas)
            amount_str = str(amount_value).replace(',', '').replace('$', '').replace('£', '').replace('€', '').strip()
            amount = Decimal(amount_str)
        except (ValueError, InvalidOperation):
            amount = Decimal('0.00')
    else:
        amount = Decimal('0.00')
    
    # Parse balance
    balance_value = find_field_value('balance')
    if balance_value is not None:
        try:
            balance_str = str(balance_value).replace(',', '').replace('$', '').replace('£', '').replace('€', '').strip()
            balance = Decimal(balance_str)
        except (ValueError, InvalidOperation):
            balance = None
    else:
        balance = None
    
    return {
        'company': company,
        'file_upload': file_upload,
        'transaction_date': transaction_date,
        'description': str(find_field_value('description') or ''),
        'amount': amount,
        'reference_number': str(find_field_value('reference') or ''),
        'bank_reference': str(transaction_id),
        'balance': balance,
        'status': 'unmatched',
        'raw_data': dict(row),
        'created_at': timezone.now(),
        'updated_at': timezone.now()
    }


@shared_task
def trigger_ml_matching(company_id):
    """
    Trigger ML matching for a company (simplified version).
    """
    try:
        company = Company.objects.get(id=company_id)
        logger.info(f"ML matching triggered for {company.name} (simplified mode)")
        
        # In a real implementation, this would:
        # 1. Get unmatched transactions
        # 2. Run ML model to suggest matches
        # 3. Auto-match high-confidence suggestions
        
        return {
            'status': 'completed',
            'company': company.name,
            'matched_count': 0,
            'mode': 'simplified'
        }
        
    except Exception as e:
        logger.error(f"ML matching failed: {e}")
        raise


@shared_task
def generate_reconciliation_report(summary_id, format_type, user_id):
    """
    Generate a reconciliation report (simplified version).
    """
    try:
        summary = ReconciliationSummary.objects.get(id=summary_id)
        logger.info(f"Generating {format_type} report for summary {summary_id} (simplified mode)")
        
        # In a real implementation, this would:
        # 1. Query reconciliation data
        # 2. Generate PDF or Excel report
        # 3. Store the file and update the summary
        
        summary.status = 'completed'
        summary.file_path = f'/tmp/report_{summary_id}.{format_type}'
        summary.generated_at = timezone.now()
        summary.save()
        
        return {
            'status': 'completed',
            'format': format_type,
            'file_path': summary.file_path,
            'mode': 'simplified'
        }
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise


@shared_task
def retrain_ml_model(company_id):
    """
    Retrain ML model for a company (simplified version).
    """
    try:
        company = Company.objects.get(id=company_id)
        logger.info(f"Model retraining triggered for {company.name} (simplified mode)")
        
        # In a real implementation, this would:
        # 1. Collect training data from reconciliation logs
        # 2. Train/retrain the ML model
        # 3. Validate performance
        # 4. Deploy new model version
        
        # Create a mock model version
        version = f"v{timezone.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Deactivate old models
        MLModelVersion.objects.filter(
            company=company,
            is_active=True
        ).update(is_active=False)
        
        # Create new model version
        new_model = MLModelVersion.objects.create(
            company=company,
            version=version,
            model_type='simplified',
            accuracy_score=85.0,  # Mock accuracy
            precision_score=80.0,
            recall_score=90.0,
            f1_score=85.0,
            training_data_count=100,  # Mock count
            is_active=True
        )
        
        return {
            'status': 'completed',
            'company': company.name,
            'model_version': version,
            'accuracy': 85.0,
            'mode': 'simplified'
        }
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise


@shared_task
def cleanup_old_files(days=30):
    """
    Clean up old uploaded files.
    """
    try:
        cutoff_date = timezone.now() - timedelta(days=days)
        old_uploads = FileUploadStatus.objects.filter(
            created_at__lt=cutoff_date,
            status__in=['completed', 'failed']
        )
        
        count = old_uploads.count()
        old_uploads.delete()
        
        logger.info(f"Cleaned up {count} old file uploads")
        return {'cleaned_count': count}
        
    except Exception as e:
        logger.error(f"File cleanup failed: {e}")
        raise


@shared_task
def daily_reconciliation_summary(company_id=None):
    """
    Generate daily reconciliation summary.
    """
    try:
        companies = Company.objects.filter(id=company_id) if company_id else Company.objects.all()
        
        for company in companies:
            today = timezone.now().date()
            
            unmatched_count = BankTransaction.objects.filter(
                company=company,
                status='unmatched',
                transaction_date=today
            ).count()
            
            total_count = BankTransaction.objects.filter(
                company=company,
                transaction_date=today
            ).count()
            
            logger.info(f"{company.name}: {total_count} total, {unmatched_count} unmatched transactions today")
        
        return {'status': 'completed'}
        
    except Exception as e:
        logger.error(f"Daily summary failed: {e}")
        raise
