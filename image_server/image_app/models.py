from django.db import models

class ImageModel(models.Model):
    image = models.ImageField(upload_to='images/')
    # 다른 필드들을 필요에 따라 추가할 수 있습니다.
