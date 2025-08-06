import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime, timedelta
from decimal import Decimal
import logging

from django.conf import settings
from core.models import Invoice
from .models import ReconciliationLog, MLModelVersion
from .utils import calculate_similarity_score, extract_reference_numbers
from ml_engine.models import TrainingData, ModelPrediction

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from transactions and invoices for ML model."""
    
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def extract_transaction_features(self, transaction):
        """Extract features from a bank transaction."""
        features = {}
        
        # Text features
        description = transaction.description.lower()
        features['description_length'] = len(description)
        features['description_word_count'] = len(description.split())
        
        # Amount features
        features['amount'] = float(transaction.amount)
        features['amount_abs'] = abs(float(transaction.amount))
        features['amount_log'] = np.log(abs(float(transaction.amount)) + 1)
        
        # Date features
        features['transaction_day'] = transaction.transaction_date.day
        features['transaction_month'] = transaction.transaction_date.month
        features['transaction_weekday'] = transaction.transaction_date.weekday()
        
        # Reference features
        references = extract_reference_numbers(description)
        features['has_reference'] = len(references) > 0
        features['reference_count'] = len(references)
        
        # Transaction type features
        features['is_debit'] = transaction.amount < 0
        features['is_credit'] = transaction.amount > 0
        
        return features
    
    def extract_invoice_features(self, invoice):
        """Extract features from an invoice."""
        features = {}
        
        # Amount features
        features['invoice_amount'] = float(invoice.total_amount)
        features['invoice_amount_log'] = np.log(float(invoice.total_amount) + 1)
        
        # Date features
        features['issue_day'] = invoice.issue_date.day
        features['issue_month'] = invoice.issue_date.month
        features['due_day'] = invoice.due_date.day
        features['due_month'] = invoice.due_date.month
        
        # Customer features
        customer_name = invoice.customer.name.lower()
        features['customer_name_length'] = len(customer_name)
        features['customer_name_word_count'] = len(customer_name.split())
        
        # Invoice features
        features['invoice_number_length'] = len(invoice.invoice_number)
        features['has_reference_number'] = bool(invoice.reference_number)
        
        return features
    
    def extract_pair_features(self, transaction, invoice):
        """Extract features from transaction-invoice pairs."""
        features = {}
        
        # Amount similarity
        amount_diff = abs(float(transaction.amount) - float(invoice.total_amount))
        features['amount_diff'] = amount_diff
        features['amount_ratio'] = min(
            float(transaction.amount), float(invoice.total_amount)
        ) / max(float(transaction.amount), float(invoice.total_amount))
        features['amount_exact_match'] = float(transaction.amount) == float(invoice.total_amount)
        
        # Date proximity
        date_diff = abs((transaction.transaction_date - invoice.due_date).days)
        features['date_diff'] = date_diff
        features['date_within_week'] = date_diff <= 7
        features['date_within_month'] = date_diff <= 30
        
        # Text similarity
        transaction_text = transaction.description.lower()
        invoice_text = f"{invoice.customer.name} {invoice.invoice_number} {invoice.description or ''}".lower()
        
        features['text_similarity'] = calculate_similarity_score(transaction_text, invoice_text)
        
        # Reference matching
        transaction_refs = extract_reference_numbers(transaction.description)
        invoice_refs = []
        if invoice.reference_number:
            invoice_refs.extend(extract_reference_numbers(invoice.reference_number))
        invoice_refs.extend(extract_reference_numbers(invoice.invoice_number))
        
        features['reference_overlap'] = len(set(transaction_refs) & set(invoice_refs))
        features['has_reference_match'] = features['reference_overlap'] > 0
        
        return features


class MLMatchingEngine:
    """Machine learning engine for automatic transaction matching."""
    
    def __init__(self, company):
        self.company = company
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.feature_columns = None
        self.model_version = None
        
        # Load existing model if available
        self.load_model()
    
    def load_model(self):
        """Load the latest trained model for the company."""
        try:
            latest_model = MLModelVersion.objects.filter(
                company=self.company,
                is_active=True
            ).first()
            
            if latest_model and os.path.exists(latest_model.model_path):
                model_data = joblib.load(latest_model.model_path)
                self.model = model_data['model']
                self.feature_columns = model_data['feature_columns']
                self.model_version = latest_model.version
                logger.info(f"Loaded ML model version {self.model_version} for {self.company.name}")
            else:
                logger.info(f"No trained model found for {self.company.name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def prepare_training_data(self):
        """Prepare training data from historical reconciliations."""
        # Get reconciliation logs
        reconciliation_logs = ReconciliationLog.objects.filter(
            transaction__company=self.company,
            is_active=True
        ).select_related('transaction', 'invoice')
        
        if reconciliation_logs.count() < 50:
            raise ValueError("Insufficient training data. Need at least 50 reconciled transactions.")
        
        # Prepare positive examples (actual matches)
        positive_examples = []
        for log in reconciliation_logs:
            features = self._extract_all_features(log.transaction, log.invoice)
            features['is_match'] = 1
            positive_examples.append(features)
        
        # Generate negative examples (non-matches)
        negative_examples = []
        transactions = [log.transaction for log in reconciliation_logs[:100]]  # Limit for performance
        invoices = Invoice.objects.filter(
            customer__company=self.company,
            status__in=['sent', 'overdue']
        )[:500]  # Limit for performance
        
        for transaction in transactions:
            # Get invoices that are NOT matched with this transaction
            matched_invoice_ids = set(
                ReconciliationLog.objects.filter(
                    transaction=transaction,
                    is_active=True
                ).values_list('invoice_id', flat=True)
            )
            
            unmatched_invoices = [inv for inv in invoices if inv.id not in matched_invoice_ids]
            
            # Sample some negative examples
            for invoice in unmatched_invoices[:5]:  # Limit negative examples per transaction
                features = self._extract_all_features(transaction, invoice)
                features['is_match'] = 0
                negative_examples.append(features)
        
        # Combine and balance the dataset
        all_examples = positive_examples + negative_examples
        
        # Convert to DataFrame
        df = pd.DataFrame(all_examples)
        return df
    
    def train_model(self):
        """Train the ML model with current data."""
        try:
            logger.info(f"Training ML model for {self.company.name}")
            
            # Prepare training data
            training_df = self.prepare_training_data()
            
            # Separate features and target
            feature_columns = [col for col in training_df.columns if col != 'is_match']
            X = training_df[feature_columns]
            y = training_df['is_match']
            
            # Handle missing values
            X = X.fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model training completed. Accuracy: {accuracy:.3f}")
            
            # Save model
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(
                settings.ML_MODEL_PATH,
                f"{self.company.id}_{model_version}.joblib"
            )
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model and metadata
            model_data = {
                'model': model,
                'feature_columns': feature_columns,
                'training_date': datetime.now(),
                'accuracy': accuracy
            }
            joblib.dump(model_data, model_path)
            
            # Deactivate old models
            MLModelVersion.objects.filter(
                company=self.company,
                is_active=True
            ).update(is_active=False)
            
            # Create new model version record
            model_version_obj = MLModelVersion.objects.create(
                company=self.company,
                version=model_version,
                model_path=model_path,
                training_data_count=len(training_df),
                accuracy_score=accuracy,
                is_active=True,
                training_metadata={
                    'feature_count': len(feature_columns),
                    'positive_examples': sum(y),
                    'negative_examples': len(y) - sum(y)
                }
            )
            
            # Update instance variables
            self.model = model
            self.feature_columns = feature_columns
            self.model_version = model_version
            
            return model_version_obj
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def find_matches(self, transaction, confidence_threshold=0.5):
        """Find potential invoice matches for a transaction."""
        if not self.model:
            logger.warning("No trained model available for matching")
            return []
        
        # Get candidate invoices
        candidate_invoices = Invoice.objects.filter(
            customer__company=self.company,
            status__in=['sent', 'overdue'],
            total_amount__gte=float(transaction.amount) * 0.5,  # Amount filter
            total_amount__lte=float(transaction.amount) * 2.0
        )
        
        matches = []
        
        for invoice in candidate_invoices:
            try:
                # Extract features
                features = self._extract_all_features(transaction, invoice)
                
                # Prepare feature vector
                feature_vector = pd.DataFrame([features])[self.feature_columns].fillna(0)
                
                # Make prediction
                probability = self.model.predict_proba(feature_vector)[0][1]  # Probability of match
                
                if probability >= confidence_threshold:
                    matches.append({
                        'invoice': invoice,
                        'confidence': probability,
                        'features': features
                    })
                
                # Store prediction for later analysis
                ModelPrediction.objects.create(
                    company=self.company,
                    transaction_id=transaction.id,
                    invoice_id=invoice.id,
                    model_version=self.model_version,
                    prediction_score=probability,
                    features_used=features
                )
                
            except Exception as e:
                logger.error(f"Error predicting match for invoice {invoice.id}: {e}")
        
        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return matches
    
    def create_reconciliation_logs(self, transaction, matches):
        """Create reconciliation logs for ML matches."""
        for match in matches:
            if match['confidence'] >= 0.8:  # High confidence threshold for auto-matching
                ReconciliationLog.objects.create(
                    transaction=transaction,
                    invoice=match['invoice'],
                    matched_by='ml_auto',
                    confidence_score=match['confidence'],
                    amount_matched=min(transaction.amount, match['invoice'].total_amount),
                    metadata={
                        'ml_features': match['features'],
                        'model_version': self.model_version
                    }
                )
                
                # Update transaction status
                transaction.status = 'matched'
                transaction.save()
                
                logger.info(
                    f"Auto-matched transaction {transaction.id} with invoice "
                    f"{match['invoice'].invoice_number} (confidence: {match['confidence']:.3f})"
                )
    
    def retrain_model(self):
        """Retrain the model with new data."""
        return self.train_model()
    
    def _extract_all_features(self, transaction, invoice):
        """Extract all features for a transaction-invoice pair."""
        features = {}
        
        # Individual features
        transaction_features = self.feature_extractor.extract_transaction_features(transaction)
        invoice_features = self.feature_extractor.extract_invoice_features(invoice)
        pair_features = self.feature_extractor.extract_pair_features(transaction, invoice)
        
        # Combine all features
        features.update(transaction_features)
        features.update({f"invoice_{k}": v for k, v in invoice_features.items()})
        features.update(pair_features)
        
        return features
