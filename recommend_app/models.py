from django.db import models

class Images(models.Model):
    image1 = models.ImageField(upload_to='images/', blank=True, null=True)
    image2 = models.ImageField(upload_to='images/', blank=True, null=True)
    image3 = models.ImageField(upload_to='images/', blank=True, null=True)
    image4 = models.ImageField(upload_to='images/', blank=True, null=True)
    image5 = models.ImageField(upload_to='images/', blank=True, null=True)

    out_image1 = models.ImageField(upload_to='images/', blank=True, null=True)
    out_image2 = models.ImageField(upload_to='images/', blank=True, null=True)
    out_image3 = models.ImageField(upload_to='images/', blank=True, null=True)
    out_image4 = models.ImageField(upload_to='images/', blank=True, null=True)
    out_image5 = models.ImageField(upload_to='images/', blank=True, null=True)

    out_name1 = models.CharField(max_length=128)
    out_name2 = models.CharField(max_length=128)
    out_name3 = models.CharField(max_length=128)
    out_name4 = models.CharField(max_length=128)
    out_name5 = models.CharField(max_length=128)