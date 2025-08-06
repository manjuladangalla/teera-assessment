"""
Custom API Root view for Bank Reconciliation System
Provides unauthenticated access to API schema and endpoint discovery
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.reverse import reverse
from rest_framework.permissions import AllowAny
from rest_framework import status
from django.conf import settings

class APIRootView(APIView):
    """
    Custom API root view that provides unauthenticated access to API discovery.
    Returns a list of available endpoints and their descriptions.
    """
    permission_classes = [AllowAny]  # Allow unauthenticated access
    
    def get(self, request, format=None):
        """
        Return the API root with all available endpoints.
        """
        api_urls = {
            # Authentication endpoints
            'authentication': {
                'obtain_token': reverse('token_obtain_pair', request=request, format=format),
                'refresh_token': reverse('token_refresh', request=request, format=format),
                'verify_token': reverse('token_verify', request=request, format=format),
            },
            
            # Documentation endpoints
            'documentation': {
                'api_schema': reverse('schema', request=request, format=format),
                'swagger_ui': reverse('swagger-ui', request=request, format=format),
                'redoc': reverse('redoc', request=request, format=format),
            },
            
            # Simplified Bank Reconciliation endpoints
            'bank_reconciliation_simplified': {
                'upload_file': request.build_absolute_uri('/api/v1/bank/upload/'),
                'unmatched_transactions': request.build_absolute_uri('/api/v1/bank/unmatched/'),
                'bulk_reconcile': request.build_absolute_uri('/api/v1/bank/reconcile/'),
                'reconciliation_logs': request.build_absolute_uri('/api/v1/bank/logs/'),
                'summary_report': request.build_absolute_uri('/api/v1/bank/summary/'),
            },
            
            # Detailed Bank Transaction endpoints
            'bank_transactions_detailed': {
                'transactions': request.build_absolute_uri('/api/v1/bank/transactions/'),
                'uploads': request.build_absolute_uri('/api/v1/bank/uploads/'),
                'logs': request.build_absolute_uri('/api/v1/bank/logs/'),
                'rules': request.build_absolute_uri('/api/v1/bank/rules/'),
                'summaries': request.build_absolute_uri('/api/v1/bank/summaries/'),
            },
            
            # ML Engine endpoints
            'machine_learning': {
                'models': request.build_absolute_uri('/api/v1/ml/models/'),
            },
            
            # Report Generation endpoints
            'reports': {
                'reports': request.build_absolute_uri('/api/v1/reports/'),
            },
            
            # Core application endpoints
            'core': {
                'core_api': request.build_absolute_uri('/api/v1/core/'),
            }
        }
        
        # Add system information
        system_info = {
            'api_version': '1.0.0',
            'system_name': 'Bank Reconciliation System',
            'description': 'Advanced reconciliation system with ML-powered matching',
            'authentication': 'JWT Bearer Token required for most endpoints',
            'docs_available': True,
            'environment': 'development' if settings.DEBUG else 'production',
        }
        
        # Add usage instructions
        usage_info = {
            'getting_started': {
                'step_1': 'Obtain JWT token from /api/auth/token/ with username/password',
                'step_2': 'Include token in Authorization header: "Bearer <your_token>"',
                'step_3': 'Upload bank statement via POST /api/v1/bank/upload/',
                'step_4': 'View unmatched transactions via GET /api/v1/bank/unmatched/',
                'step_5': 'Reconcile transactions via POST /api/v1/bank/reconcile/',
            },
            'api_documentation': {
                'interactive_docs': request.build_absolute_uri('/api/docs/'),
                'schema_download': request.build_absolute_uri('/api/schema/'),
                'redoc_docs': request.build_absolute_uri('/api/redoc/'),
            }
        }
        
        return Response({
            'system': system_info,
            'endpoints': api_urls,
            'usage': usage_info,
            '_note': 'Most endpoints require JWT authentication. Use /api/auth/token/ to obtain tokens.'
        })
