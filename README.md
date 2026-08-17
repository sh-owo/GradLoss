# GradLoss : GradCAM을 활용한 손실함수

기본적인 CE Loss에 GradCAM++을 적용하여 모델이 특징을 잘 학습하도록 도와주는 손실함수를 구현한 프로젝트입니다.


## Example

```python
criterion = GradLoss(
    target_layer=model.layer4[-1], # Attention 추적 레이어 직접 지정 (필수, None이면 에러)
    tau=5.0,                       # GradLoss내부 σ 민감도 (default: 5)
    ce_weight=1.0,                 # CE loss 가중치
    attn_weight=0.5,               # attention loss 가중치
    eps=1e-6,                      # 수치 안정화 항 (default: 1e-6)
)

loss = criterion(model, images, labels, masks) # segmentation mask 필요 (B,H,W, 0/1)
loss.backward()
```