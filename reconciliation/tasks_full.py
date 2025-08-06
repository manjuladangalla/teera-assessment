from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
from django.db import models
from decimal import Decimal
# Temporarily commented out to avoid dependency issues
# import pandas as pd
# import numpy as np
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback

from .models import FileUploadStatus, BankTransaction, ReconciliationSummary, MLModelVersion
# Temporarily commented out missing imports
# from .advanced_file_processor import AdvancedFileProcessor
# from .ml_matching import MLMatchingEngine
# from .report_generators import PDFReportGenerator, ExcelReportGenerator
# from ml_engine.deep_learning_engine import DeepLearningMatchingEngine, MLModelManager
from core.models import Company

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def process_bank_statement_file(self, file_upload_id):
    """
    Advanced file processing with comprehensive validation and ML integration.
    """
    try:
        file_upload = FileUploadStatus.objects.get(id=file_upload_id)
        file_upload.status = 'processing'
        file_upload.processing_started_at = timezone.now()
        file_upload.save()
        
        logger.info(f"Processing file: {file_upload.original_filename}")
        
        # Initialize advanced file processor
        processor = AdvancedFileProcessor(file_upload)
        
        # Step 1: Validate file
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'validation', 'message': 'Validating file format and content'}
        )
        
        validation_result = processor.validate_file()
        
        if not validation_result['is_valid']:
            file_upload.status = 'failed'
            file_upload.error_log = json.dumps({
                'stage': 'validation',
                'errors': validation_result['errors'],
                'warnings': validation_result['warnings']
            })
            file_upload.save()
            raise ValueError(f"File validation failed: {validation_result['errors']}")
        
        # Log validation warnings
        if validation_result['warnings']:
            logger.warning(f"File validation warnings: {validation_result['warnings']}")
        
        # Step 2: Process file and create transactions
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'processing', 'message': 'Creating transaction records'}
        )
        
        processing_result = processor.process_file()
        
        # Update file upload status
        file_upload.total_records = processing_result['total_records']
        file_upload.processed_records = processing_result['processed_records']
        file_upload.failed_records = processing_result['failed_records']
        
        if processing_result['errors']:
            file_upload.error_log = json.dumps({
                'stage': 'processing',
                'errors': processing_result['errors']
            })
        
        # Step 3: ML-based duplicate detection and enhancement
        if processing_result['created_transactions']:
            self.update_state(
                state='PROGRESS',
                meta={'stage': 'ml_analysis', 'message': 'Running ML analysis on transactions'}
            )
            
            ml_enhancement_result = enhance_transactions_with_ml.delay(
                processing_result['created_transactions'],
                file_upload.company.id
            )
        
        # Step 4: Auto-matching with ML
        if processing_result['processed_records'] > 0:
            self.update_state(
                state='PROGRESS',
                meta={'stage': 'auto_matching', 'message': 'Running automatic ML matching'}
            )
            
            trigger_advanced_ml_matching.delay(
                file_upload.company.id,
                processing_result['created_transactions']
            )
        
        # Final status update
        file_upload.status = 'completed' if processing_result['failed_records'] == 0 else 'completed_with_errors'
        file_upload.processing_completed_at = timezone.now()
        file_upload.save()
        
        # Clean up temporary file
        try:
            if hasattr(file_upload, 'file_path') and os.path.exists(file_upload.file_path):
                os.unlink(file_upload.file_path)
        except OSError:
            pass
        
        logger.info(f"File processing completed. Created: {processing_result['processed_records']}, Failed: {processing_result['failed_records']}")
        
        return {
            'status': 'completed',
            'created_count': processing_result['processed_records'],
            'failed_count': processing_result['failed_records'],
            'validation_warnings': validation_result.get('warnings', []),
            'processing_errors': processing_result.get('errors', [])
        }
        
    except Exception as e:
        logger.error(f"File processing failed: {e}\n{traceback.format_exc()}")
        
        try:
            file_upload = FileUploadStatus.objects.get(id=file_upload_id)
            file_upload.status = 'failed'
            file_upload.error_log = json.dumps({
                'stage': 'unexpected_error',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            file_upload.processing_completed_at = timezone.now()
            file_upload.save()
        except:
            pass
        
        raise


@shared_task(bind=True)
def enhance_transactions_with_ml(self, transaction_ids: List[int], company_id: int):
    """
    Enhance transactions with ML-based analysis and categorization.
    """
    try:
        company = Company.objects.get(id=company_id)
        ml_manager = MLModelManager(company)
        
        # Get transactions
        transactions = BankTransaction.objects.filter(
            id__in=transaction_ids,
            company=company
        )
        
        enhanced_count = 0
        
        for transaction in transactions:
            try:
                # ML-based categorization and enhancement
                enhancement_data = ml_manager.enhance_transaction(transaction)
                
                if enhancement_data:
                    # Update transaction with ML insights
                    transaction.ml_category = enhancement_data.get('category')
                    transaction.ml_confidence = enhancement_data.get('confidence')
                    transaction.ml_features = enhancement_data.get('features', {})
                    transaction.save(update_fields=['ml_category', 'ml_confidence', 'ml_features'])
                    enhanced_count += 1
                
                # Update progress
                if enhanced_count % 50 == 0:
                    self.update_state(
                        state='PROGRESS',
                        meta={'enhanced': enhanced_count, 'total': len(transaction_ids)}
                    )
            
            except Exception as e:
                logger.error(f"Failed to enhance transaction {transaction.id}: {e}")
        
        logger.info(f"Enhanced {enhanced_count} transactions with ML analysis")
        
        return {
            'company': company.name,
            'enhanced_count': enhanced_count,
            'total_transactions': len(transaction_ids)
        }
        
    except Exception as e:
        logger.error(f"ML enhancement failed: {e}\n{traceback.format_exc()}")
        raise


@shared_task(bind=True)
def trigger_advanced_ml_matching(self, company_id: int, transaction_ids: Optional[List[int]] = None):
    """
    Advanced ML-based matching with deep learning models.
    """
    try:
        company = Company.objects.get(id=company_id)
        dl_engine = DeepLearningMatchingEngine(company)
        
        # Get transactions to match
        if transaction_ids:
            transactions = BankTransaction.objects.filter(
                id__in=transaction_ids,
                company=company,
                status='unmatched'
            )
        else:
            transactions = BankTransaction.objects.filter(
                company=company,
                status='unmatched'
            )
        
        matched_count = 0
        total_transactions = transactions.count()
        
        # Process in batches for better performance
        batch_size = 100
        
        for i in range(0, total_transactions, batch_size):
            batch = transactions[i:i + batch_size]
            
            for transaction in batch:
                try:
                    # Get ML matching suggestions
                    matches = dl_engine.find_intelligent_matches(
                        transaction,
                        include_confidence_threshold=0.7
                    )
                    
                    if matches:
                        # Create reconciliation logs for high-confidence matches
                        for match in matches:
                            if match['confidence'] > 0.85:
                                dl_engine.create_auto_reconciliation(transaction, match)
                                matched_count += 1
                                break
                            else:
                                # Store as potential match for manual review
                                dl_engine.store_potential_match(transaction, match)
                
                except Exception as e:
                    logger.error(f"Failed to process transaction {transaction.id}: {e}")
            
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={
                    'processed': min(i + batch_size, total_transactions),
                    'total': total_transactions,
                    'matched': matched_count
                }
            )
        
        # Update ML model performance metrics
        update_ml_performance_metrics.delay(company_id, matched_count, total_transactions)
        
        logger.info(f"Advanced ML matching completed for {company.name}. Matched: {matched_count}/{total_transactions}")
        
        return {
            'company': company.name,
            'matched_count': matched_count,
            'total_processed': total_transactions,
            'match_rate': round((matched_count / total_transactions) * 100, 2) if total_transactions > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Advanced ML matching failed: {e}\n{traceback.format_exc()}")
        raise


@shared_task
def update_ml_performance_metrics(company_id: int, matched_count: int, total_processed: int):
    """
    Update ML model performance metrics and trigger retraining if needed.
    """
    try:
        company = Company.objects.get(id=company_id)
        
        # Get or create latest model version
        model_version = MLModelVersion.objects.filter(
            company=company,
            is_active=True
        ).first()
        
        if model_version:
            # Update performance metrics
            model_version.total_predictions += total_processed
            model_version.correct_predictions += matched_count
            
            # Calculate accuracy
            if model_version.total_predictions > 0:
                model_version.accuracy = (
                    model_version.correct_predictions / model_version.total_predictions
                ) * 100
            
            model_version.last_used_at = timezone.now()
            model_version.save()
            
            # Check if retraining is needed
            retrain_threshold = getattr(settings, 'ML_MODEL_RETRAIN_THRESHOLD', 1000)
            accuracy_threshold = getattr(settings, 'ML_MODEL_ACCURACY_THRESHOLD', 85.0)
            
            if (model_version.total_predictions >= retrain_threshold and
                model_version.accuracy < accuracy_threshold):
                
                logger.info(f"Triggering model retraining for {company.name} - Accuracy: {model_version.accuracy}%")
                retrain_deep_learning_model.delay(company_id)
        
        return {
            'company': company.name,
            'model_updated': True,
            'current_accuracy': model_version.accuracy if model_version else None
        }
        
    except Exception as e:
        logger.error(f"Failed to update ML performance metrics: {e}")
        raise


@shared_task(bind=True)
def retrain_deep_learning_model(self, company_id: int):
    """
    Retrain deep learning model with accumulated data.
    """
    try:
        company = Company.objects.get(id=company_id)
        dl_engine = DeepLearningMatchingEngine(company)
        
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'preparation', 'message': 'Preparing training data'}
        )
        
        # Prepare training data from reconciliation logs
        training_data = dl_engine.prepare_training_data()
        
        if len(training_data) < 100:  # Minimum training data threshold
            logger.warning(f"Insufficient training data for {company.name}: {len(training_data)} samples")
            return {
                'company': company.name,
                'status': 'skipped',
                'reason': 'insufficient_data',
                'samples': len(training_data)
            }
        
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'training', 'message': 'Training deep learning model'}
        )
        
        # Train new model
        training_result = dl_engine.train_model(
            training_data,
            validation_split=0.2,
            epochs=50,
            batch_size=32
        )
        
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'validation', 'message': 'Validating model performance'}
        )
        
        # Validate model performance
        validation_metrics = dl_engine.validate_model(training_data)
        
        # Create new model version if performance is good
        if validation_metrics['accuracy'] > 0.8:
            # Deactivate old model
            MLModelVersion.objects.filter(
                company=company,
                is_active=True
            ).update(is_active=False)
            
            # Create new active model version
            new_version = MLModelVersion.objects.create(
                company=company,
                version=f"v{timezone.now().strftime('%Y%m%d_%H%M%S')}",
                model_type='deep_learning',
                accuracy=validation_metrics['accuracy'] * 100,
                precision=validation_metrics['precision'] * 100,
                recall=validation_metrics['recall'] * 100,
                f1_score=validation_metrics['f1_score'] * 100,
                training_samples=len(training_data),
                is_active=True,
                trained_at=timezone.now()
            )
            
            # Save model to ONNX format
            dl_engine.export_to_onnx(new_version)
            
            logger.info(f"Successfully retrained model for {company.name} - Accuracy: {validation_metrics['accuracy']:.3f}")
            
            return {
                'company': company.name,
                'status': 'completed',
                'model_version': new_version.version,
                'accuracy': validation_metrics['accuracy'],
                'training_samples': len(training_data)
            }
        else:
            logger.warning(f"Model retraining failed for {company.name} - Low accuracy: {validation_metrics['accuracy']:.3f}")
            
            return {
                'company': company.name,
                'status': 'failed',
                'reason': 'low_accuracy',
                'accuracy': validation_metrics['accuracy'],
                'training_samples': len(training_data)
            }
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}\n{traceback.format_exc()}")
        raise


@shared_task(bind=True)
def batch_reconciliation_processing(self, company_id: int, date_range: Optional[Dict] = None):
    """
    Process batch reconciliation with comprehensive analysis.
    """
    try:
        company = Company.objects.get(id=company_id)
        
        # Determine date range
        if date_range:
            start_date = datetime.strptime(date_range['start'], '%Y-%m-%d').date()
            end_date = datetime.strptime(date_range['end'], '%Y-%m-%d').date()
        else:
            # Default to last 30 days
            end_date = timezone.now().date()
            start_date = end_date - timedelta(days=30)
        
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'analysis', 'message': 'Analyzing transaction patterns'}
        )
        
        # Get transactions in date range
        transactions = BankTransaction.objects.filter(
            company=company,
            transaction_date__range=(start_date, end_date)
        )
        
        total_transactions = transactions.count()
        unmatched_transactions = transactions.filter(status='unmatched')
        
        # Run ML matching on unmatched transactions
        if unmatched_transactions.exists():
            self.update_state(
                state='PROGRESS',
                meta={'stage': 'ml_matching', 'message': 'Running ML matching'}
            )
            
            ml_result = trigger_advanced_ml_matching.delay(
                company_id,
                list(unmatched_transactions.values_list('id', flat=True))
            )
        
        # Generate reconciliation insights
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'insights', 'message': 'Generating reconciliation insights'}
        )
        
        insights = generate_reconciliation_insights.delay(company_id, {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        })
        
        # Create comprehensive summary
        summary_data = {
            'period_start': start_date,
            'period_end': end_date,
            'total_transactions': total_transactions,
            'matched_transactions': transactions.filter(status='matched').count(),
            'unmatched_transactions': unmatched_transactions.count(),
            'total_amount': sum(t.amount for t in transactions),
            'matched_amount': sum(t.amount for t in transactions.filter(status='matched')),
            'processing_accuracy': 0.0,
            'generated_at': timezone.now()
        }
        
        if total_transactions > 0:
            summary_data['processing_accuracy'] = (
                summary_data['matched_transactions'] / total_transactions
            ) * 100
        
        logger.info(f"Batch reconciliation completed for {company.name}: {summary_data['matched_transactions']}/{total_transactions} matched")
        
        return {
            'company': company.name,
            'summary': summary_data,
            'status': 'completed'
        }
        
    except Exception as e:
        logger.error(f"Batch reconciliation processing failed: {e}\n{traceback.format_exc()}")
        raise


@shared_task
def generate_reconciliation_insights(company_id: int, date_range: Dict):
    """
    Generate advanced reconciliation insights and analytics.
    """
    try:
        company = Company.objects.get(id=company_id)
        start_date = datetime.strptime(date_range['start_date'], '%Y-%m-%d').date()
        end_date = datetime.strptime(date_range['end_date'], '%Y-%m-%d').date()
        
        # Get transactions and reconciliation logs
        transactions = BankTransaction.objects.filter(
            company=company,
            transaction_date__range=(start_date, end_date)
        )
        
        from .models import ReconciliationLog
        logs = ReconciliationLog.objects.filter(
            transaction__company=company,
            transaction__transaction_date__range=(start_date, end_date),
            is_active=True
        )
        
        # Calculate insights
        insights = {
            'matching_patterns': {},
            'ml_performance': {},
            'transaction_trends': {},
            'anomalies': [],
            'recommendations': []
        }
        
        # Matching patterns analysis
        auto_matches = logs.filter(matched_by='ml_auto').count()
        manual_matches = logs.filter(matched_by='manual').count()
        
        insights['matching_patterns'] = {
            'auto_match_rate': (auto_matches / logs.count() * 100) if logs.count() > 0 else 0,
            'manual_match_rate': (manual_matches / logs.count() * 100) if logs.count() > 0 else 0,
            'total_matches': logs.count()
        }
        
        # ML performance analysis
        if auto_matches > 0:
            avg_confidence = logs.filter(
                matched_by='ml_auto',
                confidence_score__isnull=False
            ).aggregate(avg_confidence=models.Avg('confidence_score'))['avg_confidence']
            
            insights['ml_performance'] = {
                'average_confidence': round(avg_confidence or 0, 3),
                'auto_matches': auto_matches,
                'high_confidence_matches': logs.filter(
                    matched_by='ml_auto',
                    confidence_score__gte=0.9
                ).count()
            }
        
        # Transaction trends
        daily_counts = transactions.extra(
            select={'day': 'date(transaction_date)'}
        ).values('day').annotate(count=models.Count('id')).order_by('day')
        
        insights['transaction_trends'] = {
            'daily_volumes': list(daily_counts),
            'peak_day': max(daily_counts, key=lambda x: x['count']) if daily_counts else None,
            'average_daily_volume': transactions.count() / ((end_date - start_date).days + 1)
        }
        
        # Anomaly detection
        anomalies = []
        
        # Large transactions
        large_threshold = transactions.aggregate(
            avg_amount=models.Avg('amount')
        )['avg_amount'] * 10 if transactions.exists() else 0
        
        large_transactions = transactions.filter(amount__gt=large_threshold)
        if large_transactions.exists():
            anomalies.append({
                'type': 'large_transactions',
                'count': large_transactions.count(),
                'threshold': float(large_threshold),
                'description': f'{large_transactions.count()} transactions exceed 10x average amount'
            })
        
        # Duplicate patterns
        duplicate_groups = transactions.values(
            'transaction_date', 'amount', 'description'
        ).annotate(count=models.Count('id')).filter(count__gt=1)
        
        if duplicate_groups.exists():
            anomalies.append({
                'type': 'potential_duplicates',
                'count': duplicate_groups.count(),
                'description': f'{duplicate_groups.count()} groups of potential duplicate transactions'
            })
        
        insights['anomalies'] = anomalies
        
        # Generate recommendations
        recommendations = []
        
        if insights['matching_patterns']['auto_match_rate'] < 70:
            recommendations.append({
                'priority': 'high',
                'category': 'ml_improvement',
                'message': 'Consider retraining ML model to improve automatic matching rate',
                'action': 'retrain_model'
            })
        
        if len(anomalies) > 0:
            recommendations.append({
                'priority': 'medium',
                'category': 'data_quality',
                'message': 'Review identified anomalies for data quality issues',
                'action': 'review_anomalies'
            })
        
        unmatched_rate = (transactions.filter(status='unmatched').count() / transactions.count() * 100) if transactions.count() > 0 else 0
        if unmatched_rate > 20:
            recommendations.append({
                'priority': 'medium',
                'category': 'process_improvement',
                'message': f'High unmatched rate ({unmatched_rate:.1f}%) - consider improving matching rules',
                'action': 'review_matching_rules'
            })
        
        insights['recommendations'] = recommendations
        
        # Store insights for future reference
        insights_json = json.dumps(insights, default=str)
        
        logger.info(f"Generated reconciliation insights for {company.name}")
        
        return {
            'company': company.name,
            'insights': insights,
            'period': f"{start_date} to {end_date}"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate reconciliation insights: {e}\n{traceback.format_exc()}")
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
def generate_reconciliation_report(summary_id, format_type, user_id):
    """
    Generate advanced reconciliation report with ML insights.
    """
    try:
        from django.contrib.auth.models import User
        
        summary = ReconciliationSummary.objects.get(id=summary_id)
        user = User.objects.get(id=user_id)
        
        # Enhanced report generation with ML insights
        report_data = {
            'summary': summary,
            'ml_insights': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
        # Get ML insights for the period
        insights_result = generate_reconciliation_insights(
            summary.company.id,
            {
                'start_date': summary.period_start.isoformat(),
                'end_date': summary.period_end.isoformat()
            }
        )
        
        if isinstance(insights_result, dict) and 'insights' in insights_result:
            report_data['ml_insights'] = insights_result['insights']
        
        # Generate report based on format
        if format_type == 'pdf':
            generator = PDFReportGenerator()
        elif format_type == 'xlsx':
            generator = ExcelReportGenerator()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        report_path = generator.generate_enhanced_report(report_data)
        
        # Send email with report attachment
        send_mail(
            subject=f'Advanced Reconciliation Report - {summary.period_start} to {summary.period_end}',
            message=f'Please find attached your enhanced reconciliation report with ML insights for {summary.company.name}.',
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            fail_silently=False,
        )
        
        return {
            'status': 'completed',
            'report_path': report_path,
            'insights_included': True
        }
        
    except Exception as e:
        logger.error(f"Enhanced report generation failed: {e}\n{traceback.format_exc()}")
        raise


@shared_task
def nightly_reconciliation_batch():
    """
    Enhanced nightly batch job with comprehensive processing.
    """
    try:
        companies = Company.objects.filter(is_active=True)
        results = []
        
        for company in companies:
            logger.info(f"Running nightly reconciliation for {company.name}")
            
            try:
                # 1. Run advanced ML matching
                ml_result = trigger_advanced_ml_matching.delay(company.id)
                
                # 2. Generate daily summary with insights
                summary_result = generate_enhanced_daily_summary.delay(company.id)
                
                # 3. Check if model retraining is needed
                check_model_retraining_need.delay(company.id)
                
                # 4. Generate daily insights
                insights_result = generate_reconciliation_insights.delay(
                    company.id,
                    {
                        'start_date': timezone.now().date().isoformat(),
                        'end_date': timezone.now().date().isoformat()
                    }
                )
                
                results.append({
                    'company': company.name,
                    'status': 'completed',
                    'tasks_triggered': 4
                })
                
            except Exception as e:
                logger.error(f"Nightly processing failed for {company.name}: {e}")
                results.append({
                    'company': company.name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Generate system-wide statistics
        system_stats = generate_system_statistics.delay()
        
        return {
            'status': 'completed',
            'companies_processed': len(results),
            'results': results,
            'system_stats_task': system_stats.id if system_stats else None
        }
        
    except Exception as e:
        logger.error(f"Nightly reconciliation batch failed: {e}\n{traceback.format_exc()}")
        raise


@shared_task
def generate_enhanced_daily_summary(company_id):
    """
    Generate enhanced daily reconciliation summary with ML metrics.
    """
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
                'average_confidence': Decimal('0'),
                'processing_time': Decimal('0'),
                'ml_accuracy': Decimal('0')
            }
        )
        
        if created or summary.updated_at < timezone.now() - timedelta(hours=1):
            # Calculate enhanced statistics
            transactions = BankTransaction.objects.filter(
                company=company,
                transaction_date=today
            )
            
            summary.total_transactions = transactions.count()
            summary.matched_transactions = transactions.filter(status='matched').count()
            summary.unmatched_transactions = transactions.filter(status='unmatched').count()
            summary.total_amount = sum(t.amount for t in transactions) if transactions.exists() else Decimal('0')
            
            # Calculate ML-specific metrics
            from .models import ReconciliationLog
            logs = ReconciliationLog.objects.filter(
                transaction__company=company,
                transaction__transaction_date=today,
                is_active=True
            )
            
            if logs.exists():
                summary.matched_amount = sum(log.amount_matched for log in logs)
                summary.ml_matches = logs.filter(matched_by='ml_auto').count()
                summary.manual_matches = logs.filter(matched_by='manual').count()
                
                # Calculate average confidence for ML matches
                ml_logs_with_confidence = logs.filter(
                    matched_by='ml_auto',
                    confidence_score__isnull=False
                )
                
                if ml_logs_with_confidence.exists():
                    summary.average_confidence = ml_logs_with_confidence.aggregate(
                        avg_confidence=models.Avg('confidence_score')
                    )['avg_confidence']
                
                # Calculate ML accuracy (successful matches / total ML attempts)
                ml_attempts = logs.filter(matched_by='ml_auto').count()
                ml_successful = logs.filter(matched_by='ml_auto', is_verified=True).count()
                
                if ml_attempts > 0:
                    summary.ml_accuracy = (ml_successful / ml_attempts) * 100
            
            # Calculate processing efficiency metrics
            file_uploads_today = FileUploadStatus.objects.filter(
                company=company,
                created_at__date=today,
                status='completed'
            )
            
            if file_uploads_today.exists():
                total_processing_time = sum([
                    (upload.processing_completed_at - upload.processing_started_at).total_seconds()
                    for upload in file_uploads_today
                    if upload.processing_completed_at and upload.processing_started_at
                ])
                
                summary.processing_time = total_processing_time / file_uploads_today.count()
            
            # Add metadata
            summary.metadata = {
                'generated_at': timezone.now().isoformat(),
                'ml_model_version': None,
                'data_quality_score': 0.0,
                'automation_rate': 0.0
            }
            
            # Get current ML model version
            active_model = MLModelVersion.objects.filter(
                company=company,
                is_active=True
            ).first()
            
            if active_model:
                summary.metadata['ml_model_version'] = active_model.version
            
            # Calculate automation rate
            if summary.total_transactions > 0:
                summary.metadata['automation_rate'] = (
                    summary.ml_matches / summary.total_transactions
                ) * 100
            
            # Calculate data quality score based on various factors
            quality_factors = []
            
            if transactions.exists():
                # Completeness score
                complete_transactions = transactions.exclude(
                    models.Q(description__isnull=True) |
                    models.Q(description__exact='') |
                    models.Q(amount__isnull=True)
                ).count()
                
                completeness_score = (complete_transactions / transactions.count()) * 100
                quality_factors.append(completeness_score)
                
                # Consistency score (no duplicate transactions)
                duplicates = transactions.values(
                    'transaction_date', 'amount', 'description'
                ).annotate(count=models.Count('id')).filter(count__gt=1).count()
                
                consistency_score = max(0, 100 - (duplicates / transactions.count() * 100))
                quality_factors.append(consistency_score)
                
                # Validity score (reasonable dates and amounts)
                future_dates = transactions.filter(
                    transaction_date__gt=timezone.now().date()
                ).count()
                
                zero_amounts = transactions.filter(amount=0).count()
                
                validity_issues = future_dates + zero_amounts
                validity_score = max(0, 100 - (validity_issues / transactions.count() * 100))
                quality_factors.append(validity_score)
            
            if quality_factors:
                summary.metadata['data_quality_score'] = sum(quality_factors) / len(quality_factors)
            
            summary.save()
            
            logger.info(f"Enhanced daily summary generated for {company.name}")
        
        return {
            'company': company.name,
            'summary_id': str(summary.id),
            'created': created,
            'metrics': {
                'total_transactions': summary.total_transactions,
                'match_rate': (summary.matched_transactions / summary.total_transactions * 100) if summary.total_transactions > 0 else 0,
                'ml_accuracy': float(summary.ml_accuracy),
                'automation_rate': summary.metadata.get('automation_rate', 0),
                'data_quality_score': summary.metadata.get('data_quality_score', 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced daily summary generation failed: {e}\n{traceback.format_exc()}")
        raise


@shared_task
def check_model_retraining_need(company_id):
    """
    Check if ML model retraining is needed based on performance metrics.
    """
    try:
        company = Company.objects.get(id=company_id)
        
        # Get current active model
        active_model = MLModelVersion.objects.filter(
            company=company,
            is_active=True
        ).first()
        
        if not active_model:
            logger.info(f"No active model found for {company.name}, triggering initial training")
            retrain_deep_learning_model.delay(company_id)
            return {
                'company': company.name,
                'action': 'initial_training_triggered',
                'reason': 'no_active_model'
            }
        
        # Check retraining criteria
        retrain_needed = False
        reasons = []
        
        # Criterion 1: Model age
        model_age_days = (timezone.now() - active_model.trained_at).days
        max_model_age = getattr(settings, 'ML_MODEL_MAX_AGE_DAYS', 30)
        
        if model_age_days > max_model_age:
            retrain_needed = True
            reasons.append(f'model_age_exceeded_{model_age_days}_days')
        
        # Criterion 2: Accuracy threshold
        min_accuracy = getattr(settings, 'ML_MODEL_MIN_ACCURACY', 85.0)
        
        if active_model.accuracy < min_accuracy:
            retrain_needed = True
            reasons.append(f'accuracy_below_threshold_{active_model.accuracy}%')
        
        # Criterion 3: Sufficient new training data
        from .models import ReconciliationLog
        new_logs_count = ReconciliationLog.objects.filter(
            transaction__company=company,
            created_at__gt=active_model.trained_at,
            is_verified=True
        ).count()
        
        min_new_samples = getattr(settings, 'ML_MODEL_MIN_NEW_SAMPLES', 100)
        
        if new_logs_count >= min_new_samples:
            retrain_needed = True
            reasons.append(f'sufficient_new_data_{new_logs_count}_samples')
        
        # Criterion 4: Performance degradation
        recent_logs = ReconciliationLog.objects.filter(
            transaction__company=company,
            matched_by='ml_auto',
            created_at__gte=timezone.now() - timedelta(days=7),
            confidence_score__isnull=False
        )
        
        if recent_logs.exists():
            recent_avg_confidence = recent_logs.aggregate(
                avg_confidence=models.Avg('confidence_score')
            )['avg_confidence']
            
            confidence_threshold = getattr(settings, 'ML_MODEL_MIN_CONFIDENCE', 0.8)
            
            if recent_avg_confidence < confidence_threshold:
                retrain_needed = True
                reasons.append(f'confidence_degraded_{recent_avg_confidence:.3f}')
        
        if retrain_needed:
            logger.info(f"Triggering model retraining for {company.name}: {reasons}")
            retrain_deep_learning_model.delay(company_id)
            
            return {
                'company': company.name,
                'action': 'retraining_triggered',
                'reasons': reasons,
                'model_version': active_model.version
            }
        else:
            return {
                'company': company.name,
                'action': 'no_retraining_needed',
                'model_version': active_model.version,
                'model_age_days': model_age_days,
                'current_accuracy': active_model.accuracy
            }
        
    except Exception as e:
        logger.error(f"Failed to check model retraining need: {e}\n{traceback.format_exc()}")
        raise


@shared_task
def generate_system_statistics():
    """
    Generate system-wide statistics and performance metrics.
    """
    try:
        stats = {
            'timestamp': timezone.now().isoformat(),
            'companies': {},
            'system_totals': {},
            'performance_metrics': {},
            'ml_metrics': {}
        }
        
        # Company-level statistics
        companies = Company.objects.filter(is_active=True)
        
        for company in companies:
            company_stats = {
                'total_transactions': BankTransaction.objects.filter(company=company).count(),
                'matched_transactions': BankTransaction.objects.filter(company=company, status='matched').count(),
                'active_ml_model': None,
                'last_processing': None
            }
            
            # Get active ML model info
            active_model = MLModelVersion.objects.filter(company=company, is_active=True).first()
            if active_model:
                company_stats['active_ml_model'] = {
                    'version': active_model.version,
                    'accuracy': active_model.accuracy,
                    'trained_at': active_model.trained_at.isoformat()
                }
            
            # Get last file processing
            last_upload = FileUploadStatus.objects.filter(
                company=company,
                status='completed'
            ).order_by('-created_at').first()
            
            if last_upload:
                company_stats['last_processing'] = last_upload.created_at.isoformat()
            
            company_stats['match_rate'] = (
                company_stats['matched_transactions'] / company_stats['total_transactions'] * 100
            ) if company_stats['total_transactions'] > 0 else 0
            
            stats['companies'][company.name] = company_stats
        
        # System totals
        stats['system_totals'] = {
            'total_companies': companies.count(),
            'total_transactions': BankTransaction.objects.count(),
            'total_matched': BankTransaction.objects.filter(status='matched').count(),
            'total_file_uploads': FileUploadStatus.objects.count(),
            'successful_uploads': FileUploadStatus.objects.filter(status='completed').count()
        }
        
        # Performance metrics
        today = timezone.now().date()
        this_week = timezone.now() - timedelta(days=7)
        
        stats['performance_metrics'] = {
            'transactions_today': BankTransaction.objects.filter(
                transaction_date=today
            ).count(),
            'transactions_this_week': BankTransaction.objects.filter(
                transaction_date__gte=this_week.date()
            ).count(),
            'uploads_today': FileUploadStatus.objects.filter(
                created_at__date=today
            ).count(),
            'average_processing_time': 0.0
        }
        
        # Calculate average processing time
        recent_uploads = FileUploadStatus.objects.filter(
            created_at__gte=this_week,
            status='completed',
            processing_started_at__isnull=False,
            processing_completed_at__isnull=False
        )
        
        if recent_uploads.exists():
            total_time = sum([
                (upload.processing_completed_at - upload.processing_started_at).total_seconds()
                for upload in recent_uploads
            ])
            stats['performance_metrics']['average_processing_time'] = total_time / recent_uploads.count()
        
        # ML metrics
        from .models import ReconciliationLog
        
        ml_logs_this_week = ReconciliationLog.objects.filter(
            created_at__gte=this_week,
            matched_by='ml_auto'
        )
        
        stats['ml_metrics'] = {
            'ml_matches_this_week': ml_logs_this_week.count(),
            'average_confidence': 0.0,
            'high_confidence_matches': ml_logs_this_week.filter(confidence_score__gte=0.9).count(),
            'active_models': MLModelVersion.objects.filter(is_active=True).count()
        }
        
        if ml_logs_this_week.filter(confidence_score__isnull=False).exists():
            stats['ml_metrics']['average_confidence'] = ml_logs_this_week.filter(
                confidence_score__isnull=False
            ).aggregate(avg_confidence=models.Avg('confidence_score'))['avg_confidence']
        
        logger.info("System statistics generated successfully")
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to generate system statistics: {e}\n{traceback.format_exc()}")
        raise
