from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
from django.db import models
from decimal import Decimal
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback

from .models import FileUploadStatus, BankTransaction, ReconciliationSummary, MLModelVersion
from core.models import Company

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def process_bank_statement_file(self, file_upload_id):
    """
    Process a bank statement file (simplified version).
    """
    try:
        file_upload = FileUploadStatus.objects.get(id=file_upload_id)
        file_upload.status = 'processing'
        file_upload.processing_started_at = timezone.now()
        file_upload.save()
        
        logger.info(f"Processing file: {file_upload.original_filename}")
        
        # Simplified processing - for now just mark as completed
        # In a real implementation, this would parse CSV/Excel files
        # and create BankTransaction records
        
        file_upload.status = 'completed'
        file_upload.processed_records = 0  # Would be actual count
        file_upload.failed_records = 0
        file_upload.processing_completed_at = timezone.now()
        file_upload.save()
        
        logger.info("File processing completed (simplified mode)")
        
        return {
            'status': 'completed',
            'created_count': 0,
            'failed_count': 0,
            'mode': 'simplified'
        }
        
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
