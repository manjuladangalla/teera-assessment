from rest_framework import permissions
from .utils import get_user_company


class IsCompanyMember(permissions.BasePermission):
    """Permission to check if user belongs to the same company as the resource."""
    
    def has_permission(self, request, view):
        """Check if user has a company profile."""
        try:
            get_user_company(request.user)
            return True
        except ValueError:
            return False
    
    def has_object_permission(self, request, view, obj):
        """Check if user belongs to the same company as the object."""
        try:
            user_company = get_user_company(request.user)
            
            # Check if object has a company field
            if hasattr(obj, 'company'):
                return obj.company == user_company
            
            # Check if object has a company through a related field
            if hasattr(obj, 'customer') and hasattr(obj.customer, 'company'):
                return obj.customer.company == user_company
            
            if hasattr(obj, 'transaction') and hasattr(obj.transaction, 'company'):
                return obj.transaction.company == user_company
            
            return True  # Default to allowing access if no company relation found
        except ValueError:
            return False


class IsCompanyAdmin(permissions.BasePermission):
    """Permission to check if user is an admin of their company."""
    
    def has_permission(self, request, view):
        """Check if user is a company admin."""
        if not request.user.is_authenticated:
            return False
        
        try:
            user_company = get_user_company(request.user)
            return request.user.profile.is_admin
        except (ValueError, AttributeError):
            return False


class IsOwnerOrCompanyAdmin(permissions.BasePermission):
    """Permission to check if user is the owner of the object or a company admin."""
    
    def has_object_permission(self, request, view, obj):
        """Check if user is the owner or a company admin."""
        # Check if user is the owner
        if hasattr(obj, 'user') and obj.user == request.user:
            return True
        
        # Check if user is a company admin
        try:
            user_company = get_user_company(request.user)
            if request.user.profile.is_admin:
                # Ensure the object belongs to the same company
                if hasattr(obj, 'company'):
                    return obj.company == user_company
                return True
        except (ValueError, AttributeError):
            pass
        
        return False
