
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from reconciliation.models import ReconciliationLog, BankTransaction, Invoice
from ml_engine.models import TrainingData, ModelPrediction
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Train the deep learning model for bank reconciliation'

    def add_arguments(self, parser):
        parser.add_argument(
            '--epochs',
            type=int,
            default=10,
            help='Number of training epochs (default: 10)'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=16,
            help='Batch size for training (default: 16)'
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=2e-5,
            help='Learning rate (default: 2e-5)'
        )
        parser.add_argument(
            '--company-id',
            type=str,
            help='Train model for specific company only (UUID string)'
        )
        parser.add_argument(
            '--min-samples',
            type=int,
            default=100,
            help='Minimum number of samples required for training (default: 100)'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force training even with few samples'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('Starting Deep Learning Model Training...')
        )

        try:

            self._check_dependencies()

            training_data = self._prepare_training_data(options['company_id'])

            if len(training_data[0]) < options['min_samples'] and not options['force']:
                raise CommandError(
                    f"Insufficient training data: {len(training_data[0])} samples. "
                    f"Minimum required: {options['min_samples']}. "
                    f"Use --force to train anyway or collect more reconciliation data."
                )

            self._train_model(training_data, options)

            self.stdout.write(
                self.style.SUCCESS('Model training completed successfully!')
            )

        except ImportError as e:
            raise CommandError(
                f"Missing ML dependencies. Please install: pip install -r requirements.txt\n"
                f"Error: {e}"
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise CommandError(f"Training failed: {e}")

    def _check_dependencies(self):
        try:
            import torch
            import transformers
            import sklearn
            import numpy
            import pandas

            self.stdout.write("✓ All ML dependencies available")

            if torch.cuda.is_available():
                self.stdout.write(f"✓ CUDA available: {torch.cuda.get_device_name()}")
            else:
                self.stdout.write("ℹ Using CPU (consider GPU for faster training)")

        except ImportError as e:
            raise ImportError(f"Missing dependency: {e}")

    def _prepare_training_data(self, company_id=None):
        self.stdout.write("Preparing training data...")

        logs_query = ReconciliationLog.objects.select_related(
            'transaction', 'invoice', 'invoice__customer'
        ).filter(
            matched_by__isnull=False,
            transaction__isnull=False,
            invoice__isnull=False
        )

        if company_id:
            logs_query = logs_query.filter(transaction__company_id=company_id)

        logs = list(logs_query[:5000])

        if not logs:
            raise CommandError(
                "No reconciliation logs found. Please perform some reconciliations first."
            )

        self.stdout.write(f"Found {len(logs)} confirmed matches")

        from ml_engine.deep_learning_engine import create_training_data_from_logs, augment_training_data

        transactions, invoices, labels = create_training_data_from_logs(logs)

        transactions, invoices, labels = augment_training_data(
            transactions, invoices, labels, negative_ratio=1.5
        )

        self.stdout.write(f"Training dataset: {len(transactions)} samples")
        self.stdout.write(f"Positive samples: {sum(labels)}")
        self.stdout.write(f"Negative samples: {len(labels) - sum(labels)}")

        TrainingData.objects.create(
            company_id=company_id or 1,
            data_type='reconciliation_pairs',
            features={
                'total_samples': len(transactions),
                'positive_samples': sum(labels),
                'negative_samples': len(labels) - sum(labels),
                'data_source': 'reconciliation_logs',
                'training_date': datetime.now().isoformat()
            },
            file_path=f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        return transactions, invoices, labels

    def _train_model(self, training_data, options):
        self.stdout.write("Initializing model...")

        from ml_engine.deep_learning_engine import DeepLearningReconciliationEngine

        engine = DeepLearningReconciliationEngine()

        transactions, invoices, labels = training_data
        train_loader, val_loader = engine.prepare_training_data(transactions, invoices, labels)

        self.stdout.write(f"Training batches: {len(train_loader)}")
        self.stdout.write(f"Validation batches: {len(val_loader)}")

        self.stdout.write("Starting training...")
        history = engine.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=options['epochs'],
            learning_rate=options['learning_rate']
        )

        metrics = engine.get_model_metrics(val_loader)

        self.stdout.write(self.style.SUCCESS("\nTraining Results:"))
        self.stdout.write(f"✓ Final Accuracy: {metrics['accuracy']:.4f}")
        self.stdout.write(f"✓ Precision: {metrics['precision']:.4f}")
        self.stdout.write(f"✓ Recall: {metrics['recall']:.4f}")
        self.stdout.write(f"✓ F1-Score: {metrics['f1_score']:.4f}")

        ModelPrediction.objects.create(
            company_id=options['company_id'] or 1,
            model_type='siamese_neural_network',
            confidence_score=float(metrics['accuracy']),
            predictions={
                'training_metrics': metrics,
                'training_history': history,
                'model_config': {
                    'epochs': options['epochs'],
                    'batch_size': options['batch_size'],
                    'learning_rate': options['learning_rate']
                }
            },
            created_at=datetime.now()
        )

        self.stdout.write(self.style.SUCCESS(f"Model saved to: {engine.model_path}"))
