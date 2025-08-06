from django.core.management.base import BaseCommand
from core.models import Company
from reconciliation.ml_matching import MLMatchingEngine

class Command(BaseCommand):
    help = 'Train ML models for all companies'

    def add_arguments(self, parser):
        parser.add_argument(
            '--company-id',
            type=str,
            help='Train model for specific company ID'
        )

    def handle(self, *args, **options):
        if options['company_id']:
            try:
                company = Company.objects.get(id=options['company_id'])
                companies = [company]
            except Company.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'Company with ID {options["company_id"]} not found')
                )
                return
        else:
            companies = Company.objects.filter(is_active=True)

        for company in companies:
            self.stdout.write(f'Training ML model for {company.name}...')

            try:
                ml_engine = MLMatchingEngine(company)
                model_version = ml_engine.train_model()

                self.stdout.write(
                    self.style.SUCCESS(
                        f'Successfully trained model {model_version.version} for {company.name}'
                    )
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Failed to train model for {company.name}: {e}')
                )

        self.stdout.write(
            self.style.SUCCESS('ML model training completed')
        )
