from rest_framework import permissions
from .utils import get_user_company

class IsCompanyMember(permissions.BasePermission):

    def has_permission(self, request, view):
        try:
            get_user_company(request.user)
            return True
        except ValueError:
            return False

    def has_object_permission(self, request, view, obj):
        try:
            user_company = get_user_company(request.user)

            if hasattr(obj, 'company'):
                return obj.company == user_company

            if hasattr(obj, 'customer') and hasattr(obj.customer, 'company'):
                return obj.customer.company == user_company

            if hasattr(obj, 'transaction') and hasattr(obj.transaction, 'company'):
                return obj.transaction.company == user_company

            return True
        except ValueError:
            return False

class IsCompanyAdmin(permissions.BasePermission):

    def has_permission(self, request, view):
        if not request.user.is_authenticated:
            return False

        try:
            user_company = get_user_company(request.user)
            return request.user.profile.is_admin
        except (ValueError, AttributeError):
            return False

class IsOwnerOrCompanyAdmin(permissions.BasePermission):

    def has_object_permission(self, request, view, obj):

        if hasattr(obj, 'user') and obj.user == request.user:
            return True

        try:
            user_company = get_user_company(request.user)
            if request.user.profile.is_admin:

                if hasattr(obj, 'company'):
                    return obj.company == user_company
                return True
        except (ValueError, AttributeError):
            pass

        return False
