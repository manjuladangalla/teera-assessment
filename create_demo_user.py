#!/usr/bin/env python
"""
Create demo user and company for testing
"""
import os
import sys
import django

# Add the project directory to the path
sys.path.append('/Users/mdangallage/teera-assessment')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

django.setup()

from django.contrib.auth.models import User
from core.models import Company, UserProfile

def create_demo_user():
    """Create demo user and company for testing."""
    
    # Create or get demo company
    company, created = Company.objects.get_or_create(
        name='Demo Company',
        defaults={
            'industry': 'Technology',
            'contact_email': 'demo@democompany.com',
            'is_active': True
        }
    )
    
    if created:
        print("✅ Created demo company: Demo Company")
    else:
        print("✅ Using existing demo company: Demo Company")
    
    # Create or get demo user
    user, created = User.objects.get_or_create(
        username='demo',
        defaults={
            'email': 'demo@example.com',
            'first_name': 'Demo',
            'last_name': 'User',
            'is_staff': True,
            'is_superuser': True
        }
    )
    
    if created:
        user.set_password('demo123')  # Simple password for testing
        user.save()
        print("✅ Created demo user: demo / demo123")
    else:
        print("✅ Using existing demo user: demo")
    
    # Create or get user profile
    profile, created = UserProfile.objects.get_or_create(
        user=user,
        defaults={
            'company': company,
            'employee_id': 'EMP-001',
            'department': 'Finance',
            'is_admin': True
        }
    )
    
    if created:
        print("✅ Created user profile for demo user")
    else:
        print("✅ Using existing user profile")
    
    print(f"\n🎯 Test Credentials:")
    print(f"   Username: demo")
    print(f"   Password: demo123")
    print(f"   Company: {company.name}")
    print(f"   API Root: http://127.0.0.1:8000/api/v1/")
    print(f"   Admin: http://127.0.0.1:8000/admin/")

if __name__ == "__main__":
    create_demo_user()
