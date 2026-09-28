from django.core.management.base import BaseCommand
from django.contrib.auth.models import User


class Command(BaseCommand):
    help = 'Initializes demo superuser and student account for presentation.'

    def handle(self, *args, **options):
        # Create Superuser
        if not User.objects.filter(username='admin').exists():
            admin_user = User.objects.create_superuser(
                username='admin',
                email='admin@example.com',
                password='adminpassword'
            )
            self.stdout.write(self.style.SUCCESS("Superuser 'admin' created (Password: adminpassword)"))
        else:
            self.stdout.write(self.style.WARNING("Superuser 'admin' already exists."))

        # Create Demo Student User
        if not User.objects.filter(username='student').exists():
            student_user = User.objects.create_user(
                username='student',
                email='student@example.com',
                password='studentpassword',
                first_name='Academic',
                last_name='Student'
            )
            self.stdout.write(self.style.SUCCESS("Demo Student user 'student' created (Password: studentpassword)"))
        else:
            self.stdout.write(self.style.WARNING("User 'student' already exists."))
