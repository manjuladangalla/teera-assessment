from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
from decimal import Decimal
import pandas as pd
import logging
import os

from .models import FileUploadStatus, BankTransaction, ReconciliationSummary
from .file_processors import CSVProcessor, ExcelProcessor
from .ml_matching import MLMatchingEngine
from .report_generators import PDFReportGenerator, ExcelReportGenerator
from core.models import Company

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def process_bank_statement_file(self, file_upload_id):
    """Process uploaded bank statement file."""
    try:
        file_upload = FileUploadStatus.objects.get(id=file_upload_id)
        file_upload.status = 'processing'
        file_upload.save()
        
        logger.info(f"Processing file: {file_upload.original_filename}")
        
        # Determine file type and processor
        file_extension = os.path.splitext(file_upload.original_filename)[1].lower()
        
        if file_extension == '.csv':
            processor = CSVProcessor()
        elif file_extension in ['.xlsx', '.xls']:
            processor = ExcelProcessor()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Process the file
        transactions_data = processor.process_file(file_upload.file_path)
        
        # Create bank transaction records
        created_count = 0
        failed_count = 0
        
        for row_data in transactions_data:
            try:
                transaction = BankTransaction.objects.create(
                    company=file_upload.company,
                    transaction_date=row_data['date'],
                    description=row_data['description'],
                    amount=Decimal(str(row_data['amount'])),
                    reference_number=row_data.get('reference', ''),
                    bank_reference=row_data.get('bank_reference', ''),
                    transaction_type=row_data.get('type', ''),
                    balance=Decimal(str(row_data.get('balance', 0))) if row_data.get('balance') else None,
                    file_upload=file_upload,
                    raw_data=row_data
                )
                created_count += 1
                
                # Update progress
                if created_count % 100 == 0:
                    file_upload.processed_records = created_count
                    file_upload.save()
                    self.update_state(
                        state='PROGRESS',
                        meta={'current': created_count, 'total': len(transactions_data)}
                    )
                
            except Exception as e:
                logger.error(f"Failed to create transaction: {e}")
                failed_count += 1
        
        # Update file upload status
        file_upload.status = 'completed'
        file_upload.total_records = len(transactions_data)
        file_upload.processed_records = created_count
        file_upload.failed_records = failed_count
        file_upload.save()
        
        # Trigger ML matching for new transactions
        if created_count > 0:
            trigger_ml_matching.delay(file_upload.company.id)
        
        # Clean up temporary file
        try:
            os.unlink(file_upload.file_path)
        except OSError:
            pass
        
        logger.info(f"File processing completed. Created: {created_count}, Failed: {failed_count}")
        
        return {
            'status': 'completed',
            'created_count': created_count,
            'failed_count': failed_count
        }
        
    except Exception as e:
        logger.error(f"File processing failed: {e}")
        file_upload.status = 'failed'
        file_upload.error_log = str(e)
        file_upload.save()
        raise


@shared_task(bind=True)
def trigger_ml_matching(self, company_id):
    """Trigger ML-based matching for unmatched transactions."""
    try:
        company = Company.objects.get(id=company_id)
        ml_engine = MLMatchingEngine(company)
        
        # Get unmatched transactions
        unmatched_transactions = BankTransaction.objects.filter(
            company=company,
            status='unmatched'
        )
        
        matched_count = 0
        for transaction in unmatched_transactions:
            matches = ml_engine.find_matches(transaction)
            if matches:
                ml_engine.create_reconciliation_logs(transaction, matches)
                matched_count += 1
        
        logger.info(f"ML matching completed for {company.name}. Matched: {matched_count}")
        
        return {
            'company': company.name,
            'matched_count': matched_count
        }
        
    except Exception as e:
        logger.error(f"ML matching failed: {e}")
        raise


@shared_task
def generate_reconciliation_report(summary_id, format_type, user_id):
    """Generate reconciliation report in PDF or Excel format."""
    try:
        from django.contrib.auth.models import User
        
        summary = ReconciliationSummary.objects.get(id=summary_id)
        user = User.objects.get(id=user_id)
        
        if format_type == 'pdf':
            generator = PDFReportGenerator()
        elif format_type == 'xlsx':
            generator = ExcelReportGenerator()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        report_path = generator.generate_report(summary)
        
        # Send email with report attachment
        send_mail(
            subject=f'Reconciliation Report - {summary.period_start} to {summary.period_end}',
            message=f'Please find attached your reconciliation report for {summary.company.name}.',
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            fail_silently=False,
        )
        
        return {
            'status': 'completed',
            'report_path': report_path
        }
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise


@shared_task
def nightly_reconciliation_batch():
    """Nightly batch job for automatic reconciliation."""
    try:
        companies = Company.objects.filter(is_active=True)
        
        for company in companies:
            logger.info(f"Running nightly reconciliation for {company.name}")
            
            # Trigger ML matching
            trigger_ml_matching.delay(company.id)
            
            # Generate daily summary
            generate_daily_summary.delay(company.id)
        
        return {
            'status': 'completed',
            'companies_processed': companies.count()
        }
        
    except Exception as e:
        logger.error(f"Nightly reconciliation failed: {e}")
        raise


@shared_task
def generate_daily_summary(company_id):
    """Generate daily reconciliation summary."""
    try:
        company = Company.objects.get(id=company_id)
        today = timezone.now().date()
        
        # Check if summary already exists
        summary, created = ReconciliationSummary.objects.get_or_create(
            company=company,
            period_start=today,
            period_end=today,
            defaults={
                'total_transactions': 0,
                'matched_transactions': 0,
                'unmatched_transactions': 0,
                'total_amount': Decimal('0'),
                'matched_amount': Decimal('0'),
                'ml_matches': 0,
                'manual_matches': 0,
            }
        )
        
        if created:
            # Calculate statistics
            transactions = BankTransaction.objects.filter(
                company=company,
                transaction_date=today
            )
            
            summary.total_transactions = transactions.count()
            summary.matched_transactions = transactions.filter(status='matched').count()
            summary.unmatched_transactions = transactions.filter(status='unmatched').count()
            summary.total_amount = sum(t.amount for t in transactions)
            
            # Calculate matched amount and match types
            from .models import ReconciliationLog
            logs = ReconciliationLog.objects.filter(
                transaction__company=company,
                transaction__transaction_date=today,
                is_active=True
            )
            
            summary.matched_amount = sum(log.amount_matched for log in logs)
            summary.ml_matches = logs.filter(matched_by='ml_auto').count()
            summary.manual_matches = logs.filter(matched_by='manual').count()
            
            if logs.exists():
                summary.average_confidence = sum(
                    log.confidence_score for log in logs if log.confidence_score
                ) / logs.filter(confidence_score__isnull=False).count()
            
            summary.save()
            
            logger.info(f"Daily summary generated for {company.name}")
        
        return {
            'company': company.name,
            'summary_id': str(summary.id),
            'created': created
        }
        
    except Exception as e:
        logger.error(f"Daily summary generation failed: {e}")
        raise


@shared_task
def retrain_ml_model(company_id):
    """Retrain ML model with new reconciliation data."""
    try:
        company = Company.objects.get(id=company_id)
        ml_engine = MLMatchingEngine(company)
        
        # Check if retraining is needed
        from .models import ReconciliationLog
        new_reconciliations = ReconciliationLog.objects.filter(
            transaction__company=company,
            created_at__gte=company.ml_models.filter(is_active=True).first().created_at
        ).count()
        
        if new_reconciliations >= settings.ML_MODEL_RETRAIN_THRESHOLD:
            logger.info(f"Retraining ML model for {company.name}")
            ml_engine.retrain_model()
            
            return {
                'company': company.name,
                'status': 'retrained',
                'new_reconciliations': new_reconciliations
            }
        
        return {
            'company': company.name,
            'status': 'not_needed',
            'new_reconciliations': new_reconciliations
        }
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise
