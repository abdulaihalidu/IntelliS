from django.db import models
from django.contrib.auth.models import User

# Create your models here.
GENDER_CHOICES = (
    ('Male', 'Male'),
    ('Female', 'Female'),
)


class Patient(models.Model):
    user = models.OneToOneField(
        User, null=True, blank=True, on_delete=models.CASCADE)
    username = models.CharField(max_length=20, null=False)
    first_name = models.CharField(max_length=30, null=False)
    last_name = models.CharField(max_length=30, null=False)
    email = models.EmailField(null=True)
    gender = models.CharField(max_length=8, choices=GENDER_CHOICES, null=True)
    age = models.IntegerField(null=True)

    def __str__(self):
        return self.username


class Disease(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.SET_NULL, null=True)
    title = models.CharField(max_length=150, null=False, blank=False)
    department = models.CharField(max_length=200, null=False, blank=False)
    date = models.DateTimeField(auto_now_add=True, null=False, blank=False)

    def __str__(self):
        return f"{self.title} {(self.date)}"


UNIT_CHOICES = (
    ('Cardiology', 'Cardiology'),
    ('Childhood diseaes', 'Childhood diseaes'),
    ('Dermatology', 'Dermatology'),
    ('Ear Nose and Throat (ent)', 'Ear Nose and Throat (ent)'),
    ('Family doctor', 'Family doctor'),
    ('Gastroenterology', 'Gastroenterology'),
    ('General Surgery', 'General Surgery'),
    ('Hematology', 'Hematology'),
    ('Hepatology', 'Hepatology'),
    ('Infectious diseases', 'Infectious diseases'),
    ('Internal diseaes', 'Internal diseaes'),
    ('Neurology', 'Neurology'),
    ('Orthopedic', 'Orthopedic'),
    ('Pulmonology', 'Pulmonology'),
    ('Urology', 'Urology'),
    ('infectious diseases and clinical microbiology',
     'infectious diseases and clinical microbiology')
)


class Doctor(models.Model):
    user = models.OneToOneField(
        User, null=True, blank=True, on_delete=models.CASCADE)
    first_name = models.CharField(max_length=30, null=False)
    last_name = models.CharField(max_length=30, null=False)
    username = models.CharField(max_length=20, null=False)
    email = models.EmailField(null=True)
    gender = models.CharField(max_length=8, choices=GENDER_CHOICES, null=True)
    age = models.IntegerField(null=True)
    department = models.CharField(
        max_length=200, choices=UNIT_CHOICES, null=False)

    def __str__(self):
        return f"Dr. {self.first_name} {self.last_name}"
