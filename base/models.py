from django.db import models
from django.contrib.auth.models import User

# Create your models here.


class Patient(models.Model):
    user = models.OneToOneField(
        User, null=True, blank=True, on_delete=models.CASCADE)
    username = models.CharField(max_length=20, null=True)
    first_name = models.CharField(max_length=30, null=True)
    last_name = models.CharField(max_length=30, null=True)
    email = models.EmailField(null=True)
    gender = models.CharField(max_length=8, null=True)
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
