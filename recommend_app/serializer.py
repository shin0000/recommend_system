from rest_framework import serializers
from .models import Images


class ImagesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Images
        fields = ('image1', 'image2', "image3", "image4", "image5", "out_image1", "out_image2", "out_image3", "out_image4", "out_image5", "out_name1", "out_name2", "out_name3", "out_name4", "out_name5")