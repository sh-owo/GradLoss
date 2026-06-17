import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torch.amp import autocast, GradScaler
from pycocotools.coco import COCO
from PIL import Image
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        category_id = anns[0]['category_id'] if anns else 0

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, category_id


def calculate_acc(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total


def train_model(num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(os.path.abspath('Models/run_logs/basic_loss'))
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
        batch_size=128,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_classes = 91
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
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
        epoch_acc = 0.0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            accuracy = calculate_acc(outputs, labels)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += accuracy

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}'
            })

            if global_step % 100 == 0:
                writer.add_scalar('Training/Loss', loss.item(), global_step)
                writer.add_scalar('Training/Accuracy', accuracy, global_step)
                writer.add_scalar('Training/Learning Rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_histogram('conv1/weights', model.conv1.weight.data, global_step)

            global_step += 1

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_acc = epoch_acc / len(train_loader)
        writer.add_scalar('Training/Epoch Loss', avg_epoch_loss, epoch)
        writer.add_scalar('Training/Epoch Accuracy', avg_epoch_acc, epoch)

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5
        writer.add_scalar('Gradients/L2_Norm', grad_norm, epoch)

        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.abspath(f'Models/checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                'accuracy': accuracy
            }, checkpoint_path)


    writer.close()


if __name__ == '__main__':
    train_model()