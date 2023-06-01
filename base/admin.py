from django.contrib import admin
from .models import *

# Register your models here.


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('username', 'email')
    list_filter = ('username', 'email',)
    search_fields = ("username__icontains",)


@admin.register(Disease)
class DiseaseAdmin(admin.ModelAdmin):
    list_display = ('patient', 'title', 'date')
    list_filter = ('patient', 'title', 'date')
    search_fields = ("title__icontains",)


# Customize the default Django admin site
admin.site.site_header = "IntelliS Administration"
admin.site.site_title = "IntelliS Admin Site"
admin.site.index_title = "IntelliS Admin Panel"
