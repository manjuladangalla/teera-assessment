# Deep Learning Bank Reconciliation System

## Overview

This advanced deep learning system provides AI-powered bank reconciliation capabilities using state-of-the-art neural networks. The system combines multiple AI techniques to achieve high-accuracy transaction-invoice matching.

## Architecture

### 🧠 Neural Network Components

1. **Siamese Neural Network**

   - Learns similarity between transaction and invoice pairs
   - Transformer-based text embeddings (DistilBERT)
   - Multi-head attention mechanisms
   - Numerical feature processing

2. **Feature Engineering**

   - Text similarity analysis with regex patterns
   - Amount comparison algorithms
   - Temporal pattern recognition
   - Multi-modal feature fusion

3. **Training Pipeline**
   - Automatic positive/negative sample generation
   - Balanced dataset creation
   - Cross-validation and metrics tracking
   - Model versioning and persistence

## 🚀 Quick Start

### 1. One-Command Setup & Launch

```bash
# Start server with automatic ML setup
./start_server.sh
```

This unified script will:

- Install all dependencies (including ML)
- Check system capabilities (GPU/CPU)
- Set up database migrations
- Verify ML training data
- Check for existing trained models
- Start the server with ML endpoints

### 2. Manual Installation (if needed)

```bash
# Install all dependencies including deep learning requirements
pip install -r requirements.txt

# Optional: Install CUDA support for GPU acceleration
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Train the Model

```bash
# Train with default settings (after server is running)
python manage.py train_reconciliation_model

# Advanced training options
python manage.py train_reconciliation_model \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 1e-5 \
    --company-id 1
```

### 3. Test Predictions

```bash
# Get predictions for unmatched transactions
python manage.py predict_matches

# Predict for specific transaction
python manage.py predict_matches --transaction-id 123

# Adjust confidence thresholds
python manage.py predict_matches --min-confidence 0.7 --top-k 10
```

## 📊 Model Performance

### Metrics Tracked

- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate (avoiding false matches)
- **Recall**: Coverage of actual matches
- **F1-Score**: Harmonic mean of precision and recall

### Expected Performance

- **Accuracy**: 85-95% on well-trained models
- **Precision**: 90%+ (minimal false positives)
- **Recall**: 80-90% (finds most true matches)
- **Training Time**: 10-30 minutes depending on data size

## 🔧 API Integration

### Get ML Suggestions

```bash
# Get AI-powered match suggestions
curl -X GET "http://localhost:8000/api/v1/bank/transactions/123/ml_suggestions/?top_k=5" \
     -H "Authorization: Bearer YOUR_TOKEN"
```

### Response Format

```json
{
  "suggestions": [
    {
      "invoice_id": "456",
      "confidence": 0.95,
      "invoice_details": {
        "invoice_number": "INV-2024-001",
        "customer_name": "Acme Corp",
        "total_amount": 1500.0,
        "due_date": "2024-01-15",
        "description": "Consulting services"
      },
      "match_features": {
        "is_exact_match": true,
        "is_same_day": false,
        "amount_percentage_diff": 0.0,
        "has_reference": true
      }
    }
  ]
}
```

## 🧪 Feature Engineering Details

### Text Features

- **Reference Pattern Matching**: Invoice numbers, order IDs, reference codes
- **Company Name Detection**: LTD, LLC, CORP suffixes
- **Payment Keywords**: PAYMENT, TRANSFER, DEPOSIT, WIRE, ACH
- **Text Statistics**: Length, word count, character ratios

### Amount Features

- **Exact Matching**: Penny-perfect amount matches
- **Close Matching**: Within 5% tolerance
- **Ratio Analysis**: Proportional amount relationships
- **Difference Metrics**: Absolute and percentage differences

### Temporal Features

- **Date Proximity**: Same day, within week/month
- **Future Payments**: Transaction after invoice date
- **Weekday Patterns**: Day-of-week matching
- **Seasonal Analysis**: Month/quarter patterns

## 🎯 Model Training Details

### Dataset Preparation

```python
# Automatic positive samples from existing reconciliation logs
positive_samples = ReconciliationLog.objects.filter(matched_by__isnull=False)

# Generated negative samples with smart pairing
negative_samples = generate_negative_pairs(transactions, invoices, ratio=1.5)
```

### Training Process

1. **Data Preprocessing**: Text tokenization, numerical normalization
2. **Model Initialization**: Siamese network with transformer embeddings
3. **Training Loop**: Adam optimizer, BCE loss, learning rate scheduling
4. **Validation**: Hold-out validation with early stopping
5. **Model Persistence**: PyTorch state dict saving

### Hyperparameters

```python
{
    "embedding_dim": 256,
    "transformer_model": "distilbert-base-uncased",
    "learning_rate": 2e-5,
    "batch_size": 16,
    "dropout": 0.1,
    "attention_heads": 8
}
```

## 🔍 Advanced Usage

### Custom Training Data

```python
from ml_engine.deep_learning_engine import DeepLearningReconciliationEngine

# Initialize engine
engine = DeepLearningReconciliationEngine()

# Prepare custom data
transactions = [{"id": "1", "description": "Payment", "amount": 100.0}]
invoices = [{"id": "1", "description": "Invoice", "total_amount": 100.0}]
labels = [1]  # 1 for match, 0 for no match

# Train
train_loader, val_loader = engine.prepare_training_data(transactions, invoices, labels)
history = engine.train_model(train_loader, val_loader, epochs=10)
```

### Batch Predictions

```python
# Get multiple predictions
for transaction in unmatched_transactions:
    matches = engine.find_best_matches(
        transaction_data,
        candidate_invoices,
        top_k=5,
        min_confidence=0.3
    )
    process_matches(matches)
```

## 📈 Monitoring and Maintenance

### Model Retraining

- **Trigger**: Monthly or when accuracy drops below threshold
- **Data**: Use new reconciliation logs as additional training data
- **Versioning**: Keep multiple model versions for rollback

### Performance Monitoring

- **Accuracy Tracking**: Monitor prediction accuracy over time
- **Confidence Distribution**: Ensure healthy confidence score spread
- **Feature Importance**: Track which features contribute most to predictions

### Model Deployment

```bash
# Check model status
python manage.py predict_matches --transaction-id 1

# Retrain if needed
python manage.py train_reconciliation_model --force

# Monitor performance
curl -X GET "http://localhost:8000/api/v1/ml/performance/" \
     -H "Authorization: Bearer YOUR_TOKEN"
```

## 🛠️ Troubleshooting

### Common Issues

#### "Model not trained" Error

```bash
# Solution: Train the model first
python manage.py train_reconciliation_model --force
```

#### "Insufficient training data" Error

```bash
# Solution: Create more reconciliation logs or use --force flag
python manage.py train_reconciliation_model --min-samples 50 --force
```

#### "CUDA out of memory" Error

```bash
# Solution: Reduce batch size or use CPU
python manage.py train_reconciliation_model --batch-size 8
```

#### Poor Accuracy

1. **More Training Data**: Collect more reconciliation logs
2. **Better Features**: Improve text preprocessing
3. **Hyperparameter Tuning**: Adjust learning rate, epochs
4. **Data Quality**: Ensure clean, consistent data

### Performance Optimization

#### GPU Acceleration

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Management

```python
# Reduce batch size for large datasets
engine.train_model(train_loader, val_loader, batch_size=8)

# Use gradient accumulation
engine.train_model(train_loader, val_loader, accumulation_steps=4)
```

## 🚦 Production Considerations

### Model Serving

- **Load Balancing**: Multiple model instances
- **Caching**: Redis cache for predictions
- **Monitoring**: Prometheus metrics
- **Logging**: Structured prediction logs

### Security

- **Model Isolation**: Company-specific models
- **Access Control**: API authentication required
- **Data Privacy**: No PII in training logs
- **Audit Trail**: Track all predictions

### Scalability

- **Horizontal Scaling**: Multiple worker processes
- **Async Processing**: Celery task queues
- **Database Optimization**: Indexed queries
- **CDN Caching**: Static model artifacts

## 📚 Technical References

### Deep Learning Papers

- "Siamese Neural Networks for One-shot Image Recognition" (Koch et al., 2015)
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)

### Implementation Details

- **Framework**: PyTorch 2.0+
- **Transformers**: Hugging Face Transformers 4.30+
- **Text Model**: DistilBERT (66M parameters)
- **Training Strategy**: Supervised learning with data augmentation

### Model Architecture

```
Input: [Transaction Text, Invoice Text, Numerical Features]
    ↓
[DistilBERT Embeddings] → [Text Features (256D)]
[Numerical Network] → [Numerical Features (16D)]
    ↓
[Multi-Head Attention] → [Fused Features (272D)]
    ↓
[Classification Network] → [Match Probability (0-1)]
```

## 🤝 Contributing

### Adding New Features

1. **Text Features**: Add patterns to `FeatureExtractor.text_patterns`
2. **Numerical Features**: Extend `extract_amount_features()`
3. **Model Architecture**: Modify `SiameseNetwork` class
4. **Training Logic**: Update `train_model()` method

### Testing

```bash
# Run model tests
python manage.py test ml_engine

# Test predictions
python manage.py predict_matches --transaction-id 1

# Validate training
python manage.py train_reconciliation_model --epochs 1 --force
```

---

## 📞 Support

For technical support or questions about the deep learning system:

1. **Documentation**: Check this README and code comments
2. **Logs**: Review Django logs for error details
3. **Performance**: Monitor model metrics and training history
4. **Issues**: Report bugs with reproduction steps

The deep learning reconciliation system is designed to learn and improve over time. The more reconciliation data you provide, the better the model becomes at finding accurate matches!
