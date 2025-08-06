
from django.core.management.base import BaseCommand, CommandError
from reconciliation.models import BankTransaction, Invoice
from ml_engine.models import ModelPrediction
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Predict matches for unmatched transactions using trained model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--transaction-id',
            type=int,
            help='Predict matches for specific transaction ID'
        )
        parser.add_argument(
            '--top-k',
            type=int,
            default=5,
            help='Number of top matches to return (default: 5)'
        )
        parser.add_argument(
            '--min-confidence',
            type=float,
            default=0.3,
            help='Minimum confidence threshold (default: 0.3)'
        )
        parser.add_argument(
            '--company-id',
            type=int,
            help='Filter by company ID'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('Starting Reconciliation Predictions...')
        )

        try:

            self._check_model_availability()

            transactions = self._get_unmatched_transactions(
                options['transaction_id'], options['company_id']
            )

            if not transactions:
                self.stdout.write(
                    self.style.WARNING('No unmatched transactions found')
                )
                return

            self._predict_matches(transactions, options)

        except ImportError as e:
            raise CommandError(
                f"Missing ML dependencies. Please install: pip install -r requirements.txt\n"
                f"Error: {e}"
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise CommandError(f"Prediction failed: {e}")

    def _check_model_availability(self):
        try:
            import os
            model_path = "ml_models/reconciliation_model.pth"

            if not os.path.exists(model_path):
                raise CommandError(
                    f"Trained model not found at {model_path}. "
                    f"Please run 'python manage.py train_reconciliation_model' first."
                )

            self.stdout.write("✓ Trained model found")

        except ImportError as e:
            raise ImportError(f"Missing dependency: {e}")

    def _get_unmatched_transactions(self, transaction_id=None, company_id=None):
        transactions_query = BankTransaction.objects.filter(
            reconciliationlog__isnull=True,
            is_reconciled=False
        )

        if transaction_id:
            transactions_query = transactions_query.filter(id=transaction_id)

        if company_id:
            transactions_query = transactions_query.filter(company_id=company_id)

        transactions = list(transactions_query[:100])

        self.stdout.write(f"Found {len(transactions)} unmatched transactions")
        return transactions

    def _predict_matches(self, transactions, options):

        from ml_engine.deep_learning_engine import DeepLearningReconciliationEngine

        engine = DeepLearningReconciliationEngine()
        engine.load_model()

        self.stdout.write("Model loaded successfully")

        invoices = Invoice.objects.filter(
            reconciliationlog__isnull=True,
            is_paid=False
        )[:1000]

        invoice_list = []
        for invoice in invoices:
            invoice_data = {
                'id': str(invoice.id),
                'invoice_number': invoice.invoice_number,
                'description': invoice.description or '',
                'total_amount': float(invoice.total_amount),
                'due_date': invoice.due_date.isoformat() if invoice.due_date else '',
                'customer_name': invoice.customer.name if invoice.customer else ''
            }
            invoice_list.append(invoice_data)

        self.stdout.write(f"Evaluating against {len(invoice_list)} candidate invoices")

        total_matches = 0

        for transaction in transactions:
            transaction_data = {
                'id': str(transaction.id),
                'description': transaction.description or '',
                'amount': float(transaction.amount),
                'reference_number': transaction.reference_number or '',
                'transaction_date': transaction.transaction_date.isoformat(),
                'transaction_type': transaction.transaction_type or ''
            }

            matches = engine.find_best_matches(
                transaction_data,
                invoice_list,
                top_k=options['top_k'],
                min_confidence=options['min_confidence']
            )

            if matches:
                total_matches += len(matches)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"\nTransaction {transaction.id}: {transaction.description[:50]}..."
                    )
                )
                self.stdout.write(f"Amount: ${transaction.amount}")

                for i, match in enumerate(matches, 1):
                    invoice = match['invoice']
                    confidence = match['confidence']

                    self.stdout.write(
                        f"  {i}. Invoice {invoice['invoice_number']} "
                        f"(Confidence: {confidence:.3f})"
                    )
                    self.stdout.write(f"     Amount: ${invoice['total_amount']}")
                    self.stdout.write(f"     Customer: {invoice['customer_name']}")

                    features = match['features']
                    if features.get('is_exact_match'):
                        self.stdout.write("     ✓ Exact amount match")
                    elif features.get('is_close_match'):
                        self.stdout.write(f"     ≈ Close amount match ({features.get('amount_percentage_diff', 0):.1f}% diff)")

                    if features.get('is_same_day'):
                        self.stdout.write("     ✓ Same day transaction")
                    elif features.get('is_within_week'):
                        self.stdout.write("     ≈ Within 1 week")

            else:
                self.stdout.write(
                    self.style.WARNING(
                        f"No matches found for transaction {transaction.id}"
                    )
                )

        self.stdout.write(
            self.style.SUCCESS(
                f"\nPrediction Summary:"
                f"\n- Processed: {len(transactions)} transactions"
                f"\n- Total matches found: {total_matches}"
                f"\n- Average matches per transaction: {total_matches/len(transactions):.1f}"
            )
        )
