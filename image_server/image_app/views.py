import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os
import json
import csv
import datetime
import torch.nn.functional as F
import numpy as np
import PIL
import matplotlib.pyplot as plt
import matplotlib
from google.cloud import storage
import io
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['actinic keratoses', 'basal cell carcinoma', 'benign keratosis-like lesions', 'dermatofibroma', 'melanoma', 'melanocytic nevi', 'vascular lesions']
norm_mean = (0.4914, 0.4822, 0.4465)
norm_std = (0.2023, 0.1994, 0.2010)
model_path = "C:/Users/82107/Downloads/model1.pth"
net = models.resnet18(pretrained=True)
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, len(class_names))
net.load_state_dict(torch.load(model_path, map_location=device))
net = net.to(device)
net.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

from collections import OrderedDict, Sequence

class _BaseWrapper(object):
    """
    Please modify forward() and backward() according to your task.
    """

    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        """
        Class-specific backpropagation
        Either way works:
        1. self.logits.backward(gradient=one_hot, retain_graph=True)
        2. (self.logits * one_hot).sum().backward(retain_graph=True)
        """

        one_hot = self._encode_one_hot(ids)
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers  # list

        def forward_hook(key):
            def forward_hook_(module, input, output):
                # Save featuremaps
                self.fmap_pool[key] = output.detach()

            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                # Save the gradients correspond to the featuremaps
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook_

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(module.register_backward_hook(backward_hook(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def _compute_grad_weights(self, grads):
        return F.adaptive_avg_pool2d(grads, 1)

    def forward(self, image):
        self.image_shape = image.shape[2:]
        return super(GradCAM, self).forward(image)

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = self._compute_grad_weights(grads)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

def demo2(image, label, model):
    """
    Generate Grad-CAM
    """
    # Model
    model = model
    model.to(device)
    model.eval()

    # The layers
    target_layers = ["layer4"]
    target_class = label

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(image)
    ids_ = torch.LongTensor([[target_class]] * len(image)).to(device)
    gcam.backward(ids=ids_)

    gradcam_images = []  # Grad-CAM 이미지들을 저장할 리스트

    for target_layer in target_layers:
        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)
        for j in range(len(image)):
            gcam_image = regions[j, 0]
            gradcam_images.append(gcam_image.cpu())  # Grad-CAM 이미지를 리스트에 추가

    return gradcam_images

def generate_diagnosis_id():
    # 결과 저장 디렉토리
    results_directory = "C:\\Users\\82107\\Downloads\\test_server"
    result_csv_path = os.path.join(results_directory, 'ResultList.csv')

    # ResultList.csv 파일이 존재하는 경우 마지막 진단 ID 확인
    if os.path.exists(result_csv_path):
        with open(result_csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            if len(rows) > 1:
                last_id = rows[-1][0]
                last_number = int(last_id)
            else:
                last_number = 0
    else:
        last_number = 0

    # 새로운 진단 ID 생성
    new_number = last_number + 1
    diagnosis_id = str(new_number)
    return diagnosis_id

@csrf_exempt
def predict_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')

        result_csv_path = r'C:\Users\82107\Downloads\test_server'

        # 진단 ID 생성
        diagnosis_id = generate_diagnosis_id()

        # 입력 이미지 저장
        client = storage.Client()
        image_bucket = client.get_bucket('test1014')
        image_blob = image_bucket.blob(f'test_server/original_images/{diagnosis_id}.jpg')
        image_blob.upload_from_file(image_file)

        # 테스트 이미지에 대한 예측 수행
        image = Image.open(image_file).convert('RGB')
        test_image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = net(test_image)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

        # 예측된 클래스 출력
        predicted_class = class_names[predicted.item()]

        # 클래스별 확률 값을 추출하여 딕셔너리에 저장
        class_probabilities = {}
        for i, probability in enumerate(probabilities[0]):
            class_name = class_names[i]
            class_probabilities[class_name] = f"{probability.item() * 100:.2f}%"

        # 결과 저장
        result_csv_path = os.path.join(result_csv_path, 'ResultList.csv')

        # 분류 결과 CSV 파일에 저장
        result_row = [diagnosis_id, predicted_class, str(datetime.datetime.now())]
        if not os.path.exists(result_csv_path):
            with open(result_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['ID', 'Prediction', 'Date'])
        with open(result_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(result_row)


        # GradCAM 이미지 생성 및 저장
        gcam_images = demo2(test_image, predicted.item(), net)
        for i, gcam_image in enumerate(gcam_images):
            gcam_image_blob = image_bucket.blob(f'test_server/gradcam_images/{diagnosis_id}.jpg')
            with io.BytesIO() as output:
                plt.imsave(output, gcam_image, cmap='jet')
                output.seek(0)  # 파일 포인터를 시작으로 이동
                gcam_image_blob.upload_from_file(output, content_type='image/jpeg')

        # 응답 데이터 준비
        response_data = {
            'id': diagnosis_id,
            'input_image_url': image_blob.public_url,
            'gradcam_image_urls': [image_bucket.blob(f'test_server/gradcam_images/{diagnosis_id}.jpg').public_url for i in range(len(test_image))],
            'prediction': predicted_class,
            'class_probabilities': class_probabilities
        }

        return JsonResponse(response_data)

    return JsonResponse({'message': 'Invalid request method.'})




@csrf_exempt
def doctor_feedback(request):
    if request.method == 'POST':
        try:
            feedback_list = json.loads(request.body)  # JSON 데이터 파싱
        except json.JSONDecodeError:
            response_data = {
                'message': 'Invalid JSON format',
                'status': 400
            }
            return JsonResponse(response_data, status=400)

        result_csv_path = r'C:\Users\82107\Downloads\test_server\ResultList.csv'

        # 최신 진단 ID 가져오기
        if os.path.exists(result_csv_path):
            with open(result_csv_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                last_row = None
                for row in reader:
                    last_row = row
                if last_row:
                    latest_diagnosis_id = int(last_row[0])
                else:
                    latest_diagnosis_id = 0
        else:
            latest_diagnosis_id = 0

        response_data = {
            'message': 'success',
            'status': 200
        }

        feedback_id = latest_diagnosis_id

        for feedback_data in feedback_list:
            response_data.setdefault('ids', []).append(feedback_id)

            file_name = f'{feedback_id}.txt'
            file_path = os.path.join(r'C:\Users\82107\Downloads\test_server\feedback', file_name)
            with open(file_path, 'a') as f:
                f.write(str(feedback_data) + '|')

        return JsonResponse(response_data, status=200)
    else:
        response_data = {
            'message': 'Invalid request method',
            'status': 400
        }
        return JsonResponse(response_data, status=400)




    
@csrf_exempt   
def developer_feedback(request):
    if request.method == 'GET':
        feedback_id = request.GET.get('id')
        if feedback_id:
            feedback_file = os.path.join(r'C:\Users\82107\Downloads\test_server\feedback', f"{feedback_id}.txt")
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    feedback_data = f.read()
                feedback_list = feedback_data.split('|')
                response_data = {
                    'feedback': feedback_list,
                    'message': 'success',
                    'status': 200
                }
                return JsonResponse(response_data, status=200)
            else:
                response_data = {
                    'message': 'Feedback not found',
                    'status': 404
                }
                return JsonResponse(response_data, status=404)
        else:
            response_data = {
                'message': 'Invalid feedback ID',
                'status': 400
            }
            return JsonResponse(response_data, status=400)
    else:
        response_data = {
            'message': 'Invalid request method',
            'status': 400
        }
        return JsonResponse(response_data, status=400)





@csrf_exempt
def result_list(request):
    if request.method == 'GET':
        result_csv_path = r'C:\Users\82107\Downloads\test_server\ResultList.csv'

        # 결과 목록 초기화
        result_list = []

        # 결과 CSV 파일 읽기
        if os.path.exists(result_csv_path):
            with open(result_csv_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # 헤더 라인 건너뛰기
                for row in reader:
                    diagnosis_id, prediction, date = row

                    # 입력 이미지 URL 가져오기
                    client = storage.Client()
                    image_bucket = client.get_bucket('test1014')
                    input_image_blob = image_bucket.blob(f'test_server/original_images/{diagnosis_id}.jpg')
                    input_image_url = input_image_blob.public_url

                    # GradCAM 이미지 URL 가져오기
                    gradcam_image_blob = image_bucket.blob(f'test_server/gradcam_images/{diagnosis_id}.jpg')
                    gradcam_image_url = gradcam_image_blob.public_url

                    result = {
                        'id': int(diagnosis_id),
                        'date': date,
                        'image': input_image_url,
                        'gradcam_image': gradcam_image_url
                    }

                    result_list.append(result)

        response_data = {
            'results': result_list
        }
        return JsonResponse(response_data, status=200)

    response_data = {
        'message': 'Invalid request method.'
    }
    return JsonResponse(response_data, status=400)


@csrf_exempt
def result(request):
    if request.method == 'GET':
        # 진단 ID 가져오기
        diagnosis_id = request.GET.get('id')

        # 입력 이미지 URL 가져오기
        client = storage.Client()
        image_bucket = client.get_bucket('test1014')
        input_image_blob = image_bucket.blob(f'test_server/original_images/{diagnosis_id}.jpg')
        input_image_url = input_image_blob.public_url

        # GradCAM 이미지 URL 가져오기
        gradcam_image_blob = image_bucket.blob(f'test_server/gradcam_images/{diagnosis_id}.jpg')
        gradcam_image_url = gradcam_image_blob.public_url

        # 응답 데이터 준비
        response_data = {
            'id': diagnosis_id,
            'image': input_image_url,
            'gradcam_image': gradcam_image_url
        }

        return JsonResponse(response_data)

    return JsonResponse({'message': 'Invalid request method.'})    