from django.forms import ModelForm
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import Patient


class PatientForm(ModelForm):
    class meta:
        model = Patient
        fields = '__all__'


class createUserForm(UserCreationForm):
    username = forms.CharField(widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'Username'}))
    first_name = forms.CharField(widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'First Name'}), max_length=32, help_text='First name')
    last_name = forms.CharField(widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'Last Name'}), max_length=32, help_text='Last name')
    email = forms.EmailField(widget=forms.EmailInput(attrs={
                             'class': 'form-control', 'placeholder': 'Email'}), help_text='Enter a valid email address')
    password1 = forms.CharField(widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Password'}))
    password2 = forms.CharField(widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Password Again'}))
    gender = forms.CharField(widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'Gender'}), max_length=32, help_text='Enter gender')
    age = forms.IntegerField(widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'Age'}), help_text='Enter age')

    class meta:
        model = User
        fields = UserCreationForm.Meta.fields + \
            ('first_name', 'last_name', 'email', 'gender', 'age')
