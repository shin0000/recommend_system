from rest_framework import routers
from .views import ImagesViewSet

router = routers.DefaultRouter()
router.register(r'imagess', ImagesViewSet)