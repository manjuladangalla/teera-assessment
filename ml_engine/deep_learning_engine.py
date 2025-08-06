
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class TransactionInvoiceDataset(Dataset):

    def __init__(self, transactions: List[Dict], invoices: List[Dict], 
                 labels: List[int], tokenizer, max_length: int = 128):
        self.transactions = transactions
        self.invoices = invoices
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.transactions)

    def __getitem__(self, idx):
        transaction = self.transactions[idx]
        invoice = self.invoices[idx]
        label = self.labels[idx]

        trans_text = f"{transaction.get('description', '')} {transaction.get('reference_number', '')}"
        invoice_text = f"{invoice.get('description', '')} {invoice.get('invoice_number', '')} {invoice.get('customer_name', '')}"

        trans_tokens = self.tokenizer(
            trans_text, max_length=self.max_length, 
            padding='max_length', truncation=True, return_tensors='pt'
        )
        invoice_tokens = self.tokenizer(
            invoice_text, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )

        trans_amount = float(transaction.get('amount', 0))
        invoice_amount = float(invoice.get('total_amount', 0))
        amount_diff = abs(trans_amount - invoice_amount)
        amount_ratio = min(trans_amount, invoice_amount) / max(trans_amount, invoice_amount) if max(trans_amount, invoice_amount) > 0 else 0

        trans_date = self._parse_date(transaction.get('transaction_date'))
        invoice_date = self._parse_date(invoice.get('due_date'))
        date_diff = abs((trans_date - invoice_date).days) if trans_date and invoice_date else 999

        numerical_features = torch.tensor([
            trans_amount, invoice_amount, amount_diff, amount_ratio, date_diff
        ], dtype=torch.float32)

        return {
            'trans_input_ids': trans_tokens['input_ids'].squeeze(),
            'trans_attention_mask': trans_tokens['attention_mask'].squeeze(),
            'invoice_input_ids': invoice_tokens['input_ids'].squeeze(),
            'invoice_attention_mask': invoice_tokens['attention_mask'].squeeze(),
            'numerical_features': numerical_features,
            'label': torch.tensor(label, dtype=torch.float32)
        }

    def _parse_date(self, date_str):
        if not date_str:
            return None
        try:
            if isinstance(date_str, str):
                return datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
            return date_str
        except:
            return None

class TransformerEmbedding(nn.Module):

    def __init__(self, model_name: str = 'distilbert-base-uncased', 
                 embedding_dim: int = 256):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.transformer.config.hidden_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(pooled_output)
        return self.dropout(projected)

class SiameseNetwork(nn.Module):

    def __init__(self, transformer_model: str = 'distilbert-base-uncased',
                 embedding_dim: int = 256, numerical_dim: int = 5):
        super().__init__()

        self.text_embedding = TransformerEmbedding(transformer_model, embedding_dim)

        self.numerical_net = nn.Sequential(
            nn.Linear(numerical_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim + 16, num_heads=8, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear((embedding_dim + 16) * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, trans_input_ids, trans_attention_mask,
                invoice_input_ids, invoice_attention_mask, numerical_features):

        trans_text_emb = self.text_embedding(trans_input_ids, trans_attention_mask)
        invoice_text_emb = self.text_embedding(invoice_input_ids, invoice_attention_mask)

        numerical_emb = self.numerical_net(numerical_features)

        trans_combined = torch.cat([trans_text_emb, numerical_emb], dim=1)
        invoice_combined = torch.cat([invoice_text_emb, numerical_emb], dim=1)

        trans_attended, _ = self.attention(
            trans_combined.unsqueeze(1), 
            trans_combined.unsqueeze(1), 
            trans_combined.unsqueeze(1)
        )
        invoice_attended, _ = self.attention(
            invoice_combined.unsqueeze(1),
            invoice_combined.unsqueeze(1),
            invoice_combined.unsqueeze(1)
        )

        trans_attended = trans_attended.squeeze(1)
        invoice_attended = invoice_attended.squeeze(1)

        similarity_features = torch.cat([trans_attended, invoice_attended], dim=1)

        output = self.classifier(similarity_features)
        return output.squeeze()

class FeatureExtractor:

    def __init__(self):
        self.text_patterns = {
            'reference_patterns': [
                r'INV[-_]?\d+',
                r'REF[-_]?\d+',
                r'ORDER[-_]?\d+',
                r'\b\d{6,}\b',
            ],
            'company_patterns': [
                r'LTD\.?$',
                r'LLC\.?$',
                r'CORP\.?$',
                r'INC\.?$',
            ]
        }

    def extract_text_features(self, text: str) -> Dict[str, Any]:
        if not text:
            return self._default_text_features()

        text = text.upper().strip()

        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'has_reference': any(re.search(pattern, text) for pattern in self.text_patterns['reference_patterns']),
            'has_company_suffix': any(re.search(pattern, text) for pattern in self.text_patterns['company_patterns']),
            'digit_ratio': sum(c.isdigit() for c in text) / len(text) if text else 0,
            'alpha_ratio': sum(c.isalpha() for c in text) / len(text) if text else 0,
            'special_char_ratio': sum(not c.isalnum() and not c.isspace() for c in text) / len(text) if text else 0,
            'has_payment_keywords': any(keyword in text for keyword in [
                'PAYMENT', 'TRANSFER', 'DEPOSIT', 'WIRE', 'ACH', 'CHECK'
            ]),
            'has_invoice_keywords': any(keyword in text for keyword in [
                'INVOICE', 'BILL', 'CHARGE', 'FEE', 'SERVICE'
            ])
        }

        return features

    def extract_amount_features(self, amount1: float, amount2: float) -> Dict[str, float]:
        if amount1 == 0 or amount2 == 0:
            return {
                'amount_diff': abs(amount1 - amount2),
                'amount_ratio': 0.0,
                'amount_percentage_diff': 100.0,
                'is_exact_match': False,
                'is_close_match': False
            }

        diff = abs(amount1 - amount2)
        ratio = min(amount1, amount2) / max(amount1, amount2)
        percentage_diff = (diff / max(amount1, amount2)) * 100

        return {
            'amount_diff': diff,
            'amount_ratio': ratio,
            'amount_percentage_diff': percentage_diff,
            'is_exact_match': diff < 0.01,
            'is_close_match': percentage_diff < 5.0
        }

    def extract_temporal_features(self, date1, date2) -> Dict[str, Any]:
        if not date1 or not date2:
            return self._default_temporal_features()

        if isinstance(date1, str):
            date1 = datetime.strptime(date1.split('T')[0], '%Y-%m-%d')
        if isinstance(date2, str):
            date2 = datetime.strptime(date2.split('T')[0], '%Y-%m-%d')

        diff_days = (date1 - date2).days

        return {
            'date_diff_days': abs(diff_days),
            'is_same_day': diff_days == 0,
            'is_within_week': abs(diff_days) <= 7,
            'is_within_month': abs(diff_days) <= 30,
            'is_future_payment': diff_days > 0,
            'weekday_match': date1.weekday() == date2.weekday()
        }

    def _default_text_features(self) -> Dict[str, Any]:
        return {
            'text_length': 0,
            'word_count': 0,
            'has_reference': False,
            'has_company_suffix': False,
            'digit_ratio': 0.0,
            'alpha_ratio': 0.0,
            'special_char_ratio': 0.0,
            'has_payment_keywords': False,
            'has_invoice_keywords': False
        }

    def _default_temporal_features(self) -> Dict[str, Any]:
        return {
            'date_diff_days': 999,
            'is_same_day': False,
            'is_within_week': False,
            'is_within_month': False,
            'is_future_payment': False,
            'weekday_match': False
        }

class DeepLearningReconciliationEngine:

    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.model_path = model_path or "ml_models/reconciliation_model.pth"

        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        logger.info(f"Deep Learning Engine initialized with device: {self.device}")

    def prepare_training_data(self, transactions: List[Dict], 
                            invoices: List[Dict], 
                            labels: List[int]) -> Tuple[DataLoader, DataLoader]:

        train_trans, val_trans, train_inv, val_inv, train_labels, val_labels = train_test_split(
            transactions, invoices, labels, test_size=0.2, random_state=42, stratify=labels
        )

        train_dataset = TransactionInvoiceDataset(
            train_trans, train_inv, train_labels, self.tokenizer
        )
        val_dataset = TransactionInvoiceDataset(
            val_trans, val_inv, val_labels, self.tokenizer
        )

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        return train_loader, val_loader

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                   epochs: int = 10, learning_rate: float = 2e-5) -> Dict[str, List[float]]:

        self.model = SiameseNetwork().to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.BCELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

        best_val_accuracy = 0.0

        for epoch in range(epochs):

            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()

                outputs = self.model(
                    batch['trans_input_ids'],
                    batch['trans_attention_mask'],
                    batch['invoice_input_ids'],
                    batch['invoice_attention_mask'],
                    batch['numerical_features']
                )

                loss = criterion(outputs, batch['label'])
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            val_loss, val_accuracy = self._validate_model(val_loader, criterion)

            scheduler.step(val_loss)

            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model()
                logger.info(f"New best model saved with accuracy: {best_val_accuracy:.4f}")

        return history

    def _validate_model(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    batch['trans_input_ids'],
                    batch['trans_attention_mask'],
                    batch['invoice_input_ids'],
                    batch['invoice_attention_mask'],
                    batch['numerical_features']
                )

                loss = criterion(outputs, batch['label'])
                val_loss += loss.item()

                predictions = (outputs > 0.5).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(batch['label'].cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)

        return avg_val_loss, accuracy

    def predict_match_probability(self, transaction: Dict, invoice: Dict) -> float:
        if not self.model:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")

        self.model.eval()

        dataset = TransactionInvoiceDataset(
            [transaction], [invoice], [0], self.tokenizer
        )

        batch = dataset[0]
        batch = {k: v.unsqueeze(0).to(self.device) for k, v in batch.items() if k != 'label'}

        with torch.no_grad():
            output = self.model(
                batch['trans_input_ids'],
                batch['trans_attention_mask'],
                batch['invoice_input_ids'],
                batch['invoice_attention_mask'],
                batch['numerical_features']
            )

            probability = output.item()

        return probability

    def find_best_matches(self, transaction: Dict, candidate_invoices: List[Dict],
                         top_k: int = 5, min_confidence: float = 0.3) -> List[Dict]:
        matches = []

        for invoice in candidate_invoices:
            probability = self.predict_match_probability(transaction, invoice)

            if probability >= min_confidence:
                matches.append({
                    'invoice': invoice,
                    'confidence': probability,
                    'features': self._extract_match_features(transaction, invoice)
                })

        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches[:top_k]

    def _extract_match_features(self, transaction: Dict, invoice: Dict) -> Dict:
        text_features = self.feature_extractor.extract_text_features(
            f"{transaction.get('description', '')} {invoice.get('description', '')}"
        )

        amount_features = self.feature_extractor.extract_amount_features(
            float(transaction.get('amount', 0)),
            float(invoice.get('total_amount', 0))
        )

        temporal_features = self.feature_extractor.extract_temporal_features(
            transaction.get('transaction_date'),
            invoice.get('due_date')
        )

        return {
            **text_features,
            **amount_features,
            **temporal_features
        }

    def save_model(self, path: str = None):
        if not self.model:
            raise ValueError("No model to save")

        save_path = path or self.model_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'transformer_model': 'distilbert-base-uncased',
                'embedding_dim': 256,
                'numerical_dim': 5
            }
        }, save_path)

        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: str = None):
        load_path = path or self.model_path

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)
        config = checkpoint['model_config']

        self.model = SiameseNetwork(
            transformer_model=config['transformer_model'],
            embedding_dim=config['embedding_dim'],
            numerical_dim=config['numerical_dim']
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Model loaded from {load_path}")

    def get_model_metrics(self, test_loader: DataLoader) -> Dict[str, float]:
        if not self.model:
            raise ValueError("No model loaded")

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    batch['trans_input_ids'],
                    batch['trans_attention_mask'],
                    batch['invoice_input_ids'],
                    batch['invoice_attention_mask'],
                    batch['numerical_features']
                )

                predictions = (outputs > 0.5).cpu().numpy()
                probabilities = outputs.cpu().numpy()

                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
                all_labels.extend(batch['label'].cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_samples': len(all_labels)
        }

def create_training_data_from_logs(reconciliation_logs) -> Tuple[List[Dict], List[Dict], List[int]]:
    transactions = []
    invoices = []
    labels = []

    for log in reconciliation_logs:
        transaction_data = {
            'id': str(log.transaction.id),
            'description': log.transaction.description,
            'amount': float(log.transaction.amount),
            'reference_number': log.transaction.reference_number,
            'transaction_date': log.transaction.transaction_date.isoformat(),
            'transaction_type': log.transaction.transaction_type
        }

        invoice_data = {
            'id': str(log.invoice.id),
            'invoice_number': log.invoice.invoice_number,
            'description': log.invoice.description,
            'total_amount': float(log.invoice.total_amount),
            'due_date': log.invoice.due_date.isoformat(),
            'customer_name': log.invoice.customer.name
        }

        transactions.append(transaction_data)
        invoices.append(invoice_data)
        labels.append(1)

    return transactions, invoices, labels

def augment_training_data(transactions: List[Dict], invoices: List[Dict], 
                         labels: List[int], negative_ratio: float = 1.0) -> Tuple[List[Dict], List[Dict], List[int]]:
    import random

    positive_count = sum(labels)
    negative_count = int(positive_count * negative_ratio)

    augmented_transactions = transactions.copy()
    augmented_invoices = invoices.copy()
    augmented_labels = labels.copy()

    all_transactions = [t for t, l in zip(transactions, labels) if l == 1]
    all_invoices = [i for i, l in zip(invoices, labels) if l == 1]

    for _ in range(negative_count):

        trans = random.choice(all_transactions)
        invoice = random.choice(all_invoices)

        while (trans in transactions and invoice in invoices and 
               transactions.index(trans) == invoices.index(invoice)):
            trans = random.choice(all_transactions)
            invoice = random.choice(all_invoices)

        augmented_transactions.append(trans)
        augmented_invoices.append(invoice)
        augmented_labels.append(0)

    return augmented_transactions, augmented_invoices, augmented_labels
