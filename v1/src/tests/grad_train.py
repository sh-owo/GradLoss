import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torch.amp import autocast, GradScaler
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, src_dir)

from v1.src.GradLoss.gradloss import GradLoss


class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())

    def create_mask(self, img_id, height, width):
        mask = torch.zeros((height, width), dtype=torch.float32)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            mask_ann = self.coco.annToMask(ann)
            mask = torch.maximum(mask, torch.tensor(mask_ann, dtype=torch.float32))
        return mask

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        mask = self.create_mask(img_id, img_info['height'], img_info['width'])

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        category_id = anns[0]['category_id'] if anns else 0

        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((224, 224))(mask.unsqueeze(0)).squeeze(0)

        return image, category_id, mask

    def __len__(self):
        return len(self.ids)


def calculate_acc(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total


def train_model(num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter('../Models/run_logs/grad_loss')
    scaler = GradScaler()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_root = os.path.abspath('COCO_datasets/coco2017/train2017')
    train_annot = os.path.abspath('COCO_datasets/coco2017/annotations/instances_train2017.json')

    full_dataset = COCODataset(
        root_dir=train_root,
        annotation_file=train_annot,
        transform=transform
    )

    train_loader = DataLoader(
        full_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_classes = 91
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = GradLoss(
        target_layers=[model.layer4[-1]],
        initial_lambda_ce=1.0,
        final_lambda_ce=0.7,
        initial_lambda_grad=0.0,
        final_lambda_grad=0.7,
        grad_interval=20
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.25
    )

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_grad_loss = 0.0
        epoch_acc = 0.0

        criterion.update_lambda(epoch, num_epochs)
        current_lambdas = criterion.get_current_lambdas()

        for name, value in current_lambdas.items():
            writer.add_scalar(f'Lambda/{name}', value, epoch)

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for images, labels, masks in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(images)
                total_loss, ce_loss, cur_grad_loss = criterion(
                    model=model,
                    inputs=images,
                    outputs=outputs,
                    targets=labels,
                    masks=masks,
                    global_step=global_step
                )

            accuracy = calculate_acc(outputs, labels)

            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += total_loss.item()
            epoch_ce_loss += ce_loss.item()
            epoch_grad_loss += cur_grad_loss.item()
            epoch_acc += accuracy

            progress_bar.set_postfix({
                'total_loss': f'{total_loss.item():.4f}',
                'ce_loss': f'{ce_loss.item():.4f}',
                'grad_loss': f'{cur_grad_loss.item():.4f}',
                'acc': f'{accuracy:.4f}'
            })

            if global_step % 100 == 0:
                writer.add_scalar('Training/Loss', total_loss.item(), global_step)
                writer.add_scalar('Training/CE_Loss', ce_loss.item(), global_step)
                writer.add_scalar('Training/GradCAM_Loss', cur_grad_loss.item(), global_step)
                writer.add_scalar('Training/Accuracy', accuracy, global_step)
                writer.add_scalar('Training/Learning Rate',
                                  optimizer.param_groups[0]['lr'], global_step)

            global_step += 1

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_ce_loss = epoch_ce_loss / len(train_loader)
        avg_epoch_grad_loss = epoch_grad_loss / len(train_loader)
        avg_epoch_acc = epoch_acc / len(train_loader)

        writer.add_scalar('Training/Epoch Loss', avg_epoch_loss, epoch)
        writer.add_scalar('Training/Epoch_CE_Loss', avg_epoch_ce_loss, epoch)
        writer.add_scalar('Training/Epoch_GradCAM_Loss', avg_epoch_grad_loss, epoch)
        writer.add_scalar('Training/Epoch Accuracy', avg_epoch_acc, epoch)

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5
        writer.add_scalar('Gradients/L2_Norm', grad_norm, epoch)

        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.abspath(f'Models/grad_checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'total_loss': total_loss.item(),
                'ce_loss': ce_loss.item(),
                'grad_loss': cur_grad_loss.item(),
                'accuracy': accuracy,
                'current_lambdas': current_lambdas
            }, checkpoint_path)

    writer.close()


if __name__ == '__main__':
    train_model()