from django.db import models
from core.models import TimestampedModel, Company
import uuid

class TrainingData(TimestampedModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='training_data')
    transaction_features = models.JSONField()
    invoice_features = models.JSONField()
    is_match = models.BooleanField()
    match_confidence = models.FloatField(null=True, blank=True)
    source = models.CharField(max_length=50, default='manual')

    class Meta:
        db_table = 'ml_engine_trainingdata'
        indexes = [
            models.Index(fields=['company', 'is_match']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"Training data for {self.company.name} - Match: {self.is_match}"

class FeatureExtraction(TimestampedModel):
    FEATURE_TYPE_CHOICES = [
        ('text', 'Text Features'),
        ('numeric', 'Numeric Features'),
        ('categorical', 'Categorical Features'),
        ('temporal', 'Temporal Features'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    feature_name = models.CharField(max_length=100)
    feature_type = models.CharField(max_length=20, choices=FEATURE_TYPE_CHOICES)
    extraction_config = models.JSONField()
    is_active = models.BooleanField(default=True)
    importance_score = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = 'ml_engine_featureextraction'
        unique_together = ['company', 'feature_name']

    def __str__(self):
        return f"{self.feature_name} ({self.feature_type})"

class ModelPrediction(TimestampedModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    transaction_id = models.UUIDField()
    invoice_id = models.UUIDField()
    model_version = models.CharField(max_length=50)
    prediction_score = models.FloatField()
    features_used = models.JSONField()
    prediction_metadata = models.JSONField(default=dict)
    actual_outcome = models.BooleanField(null=True, blank=True)

    class Meta:
        db_table = 'ml_engine_modelprediction'
        indexes = [
            models.Index(fields=['company', 'transaction_id']),
            models.Index(fields=['prediction_score']),
            models.Index(fields=['model_version']),
        ]

    def __str__(self):
        return f"Prediction: {self.prediction_score:.3f} for {self.transaction_id}"
