import django_filters
from rest_framework import viewsets, filters
from .models import Images
from .serializer import ImagesSerializer
from PIL import Image
import numpy as np
from .autoencoder.model import decode_img, seek_suggest, autoencoder, crop_center, train_enc, X_train, train_name, mse_similarity, cos_similarity, model, encoder
import json
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status
from io import BytesIO
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files.base import ContentFile

class ImagesViewSet(viewsets.ModelViewSet):
    queryset = Images.objects.all()
    serializer_class = ImagesSerializer

    @action(detail=True, methods=["get"])
    def suggest(self, request, pk=None):
        imagess = Images.objects.get(pk=pk)

        image1 = imagess.image1
        image2 = imagess.image2
        image3 = imagess.image3
        image4 = imagess.image4
        image5 = imagess.image5

        image1 = Image.open(image1)
        image2 = Image.open(image2)
        image3 = Image.open(image3)
        image4 = Image.open(image4)
        image5 = Image.open(image5)

        image1 = crop_center(np.array(image1)) / 255
        image2 = crop_center(np.array(image2)) / 255
        image3 = crop_center(np.array(image3)) / 255
        image4 = crop_center(np.array(image4)) / 255
        image5 = crop_center(np.array(image5)) / 255

        images = np.zeros((5, *image1.shape))

        images[0] = image1
        images[1] = image2
        images[2] = image3
        images[3] = image4
        images[4] = image5

        encs = encoder.predict(images)

        suggest_list = []
        suggest_name_list = []
        for enc in encs:
            suggest, name = seek_suggest(enc, train_enc, X_train, train_name, mse_similarity)
            suggest_list.append(suggest.tolist())
            suggest_name_list.append(name.tolist())
        suggest_list = np.array(suggest_list)
        suggest_name_list = np.array(suggest_name_list)

        img1 = decode_img(suggest_list[0])
        img2 = decode_img(suggest_list[1])
        img3 = decode_img(suggest_list[2])
        img4 = decode_img(suggest_list[3])
        img5 = decode_img(suggest_list[4])

        img1.flags.writeable = True
        img2.flags.writeable = True
        img3.flags.writeable = True
        img4.flags.writeable = True
        img5.flags.writeable = True

        name1 = suggest_name_list[0]
        name2 = suggest_name_list[1]
        name3 = suggest_name_list[2]
        name4 = suggest_name_list[3]
        name5 = suggest_name_list[4]

        imagess.out_name1 = name1
        imagess.out_name2 = name2
        imagess.out_name3 = name3
        imagess.out_name4 = name4
        imagess.out_name5 = name5

        imagess.save()

        buffer1 = BytesIO()
        Image.fromarray(img1).save(fp=buffer1, format="JPEG")
        img1 = ContentFile(buffer1.getvalue())
        buffer2 = BytesIO()
        Image.fromarray(img2).save(fp=buffer2, format="JPEG")
        img2 = ContentFile(buffer2.getvalue())
        buffer3 = BytesIO()
        Image.fromarray(img3).save(fp=buffer3, format="JPEG")
        img3 = ContentFile(buffer3.getvalue())
        buffer4 = BytesIO()
        Image.fromarray(img4).save(fp=buffer4, format="JPEG")
        img4 = ContentFile(buffer4.getvalue())
        buffer5 = BytesIO()
        Image.fromarray(img5).save(fp=buffer5, format="JPEG")
        img5 = ContentFile(buffer5.getvalue())

        imagess.out_image1.save("./img1.jpg", InMemoryUploadedFile(img1, None, "./img1.jpg", 'image/jpeg', img1.tell, None))
        imagess.out_image2.save("./img2.jpg", InMemoryUploadedFile(img2, None, "./img2.jpg", 'image/jpeg', img2.tell, None))
        imagess.out_image3.save("./img3.jpg", InMemoryUploadedFile(img3, None, "./img3.jpg", 'image/jpeg', img3.tell, None))
        imagess.out_image4.save("./img4.jpg", InMemoryUploadedFile(img4, None, "./img4.jpg", 'image/jpeg', img4.tell, None))
        imagess.out_image5.save("./img5.jpg", InMemoryUploadedFile(img5, None, "./img5.jpg", 'image/jpeg', img5.tell, None))

        return Response("success", status=status.HTTP_200_OK)