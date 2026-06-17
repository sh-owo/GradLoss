# GradLoss : GradCAM을 활용한 손실함수

기본적인 CE Loss에 GradCAM을 적용하여 모델이 특징을 잘 학습하도록 도와주는 손실함수를 구현한 프로젝트입니다.


## GradLoss 적용 예제

```python
import torch
import torchvision.models as models
from gradloss import GradLoss

model = models.resnet18(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# GradCAM 대상 레이어 지정 후 손실 초기화
target_layers = [model.layer4[-1]]
criterion = GradLoss(
    target_layers=target_layers,
    lambda_obj=1.0,     # 객체 반응 보상 가중치
    lambda_noobj=1.0,   # 배경 억제 가중치
    alpha=0.5,          # attention threshold
    tau=10.0,           # soft-threshold 날카로움
    ce_weight=1.0,      # CE loss 가중치
    attn_weight=0.5,    # attention loss 가중치
)

logits = model(images)
loss = criterion(model, images, logits, labels, masks)
loss.backward()
```