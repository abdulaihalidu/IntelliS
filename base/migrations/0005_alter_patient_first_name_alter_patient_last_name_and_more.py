# Generated by Django 4.1.7 on 2023-06-09 12:02

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('base', '0004_alter_patient_age_alter_patient_first_name_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='patient',
            name='first_name',
            field=models.CharField(max_length=30),
        ),
        migrations.AlterField(
            model_name='patient',
            name='last_name',
            field=models.CharField(max_length=30),
        ),
        migrations.AlterField(
            model_name='patient',
            name='username',
            field=models.CharField(max_length=20),
        ),
        migrations.CreateModel(
            name='Doctor',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('first_name', models.CharField(max_length=30)),
                ('last_name', models.CharField(max_length=30)),
                ('username', models.CharField(max_length=20)),
                ('email', models.EmailField(max_length=254, null=True)),
                ('gender', models.CharField(max_length=8, null=True)),
                ('age', models.IntegerField(null=True)),
                ('department', models.CharField(max_length=50)),
                ('is_doctor', models.BooleanField(default=True)),
                ('user', models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]