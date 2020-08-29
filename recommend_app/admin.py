from django.contrib import admin
from .models import Images


@admin.register(Images)
class Images(admin.ModelAdmin):
    pass