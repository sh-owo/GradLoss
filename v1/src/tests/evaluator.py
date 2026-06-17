import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns

Model_pth = os.path.abspath('')



class COCOValDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.class_names = {cat['id']: cat['name'] for cat in self.categories}

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

        return image, category_id, img_id


class ModelEvaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.writer =SummaryWriter(os.path.abspath('Models/evaluate/basic'))

    def load_model(self, model_path):
        model = resnet50()
        num_classes = 91
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model

    def log_metrics(self, class_metrics, class_names):
        """Visualize top 20 classes by sample count"""
        n_classes = 20
        class_samples = class_metrics['support']
        top_classes = np.argsort(class_samples)[-n_classes:]

        metrics_text = "Top 20 Classes by Sample Count:\n\n"
        for idx in top_classes:
            class_name = class_names.get(idx, f'Class {idx}')
            metrics_text += f"Class: {class_name}\n"
            metrics_text += f"Support: {int(class_metrics['support'][idx])}\n"
            metrics_text += f"Precision: {class_metrics['precision'][idx]:.4f}\n"
            metrics_text += f"Recall: {class_metrics['recall'][idx]:.4f}\n"
            metrics_text += f"F1: {class_metrics['f1'][idx]:.4f}\n\n"

        self.writer.add_text('Metrics/Top20_Performance', metrics_text)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
        class_labels = [class_names.get(i, f'Class {i}') for i in top_classes]

        x = np.arange(len(class_labels))
        width = 0.25
        metrics = ['precision', 'recall', 'f1']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            values = [class_metrics[metric][idx] for idx in top_classes]
            ax1.bar(x + i * width, values, width, label=metric.capitalize(), color=color)

        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics for Top 20 Classes by Sample Count')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(class_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        supports = [class_metrics['support'][idx] for idx in top_classes]
        ax2.bar(class_labels, supports, color='#17becf')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Sample Distribution for Top 20 Classes')
        ax2.set_xticklabels(class_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self.writer.add_figure('Metrics/Top20_Overview', fig)
        plt.close(fig)

    def log_confusion_matrix(self, conf_matrix, class_names):
        """Confusion matrix visualization for top 20 classes"""
        n_classes = 20
        class_samples = conf_matrix.sum(axis=1)
        top_classes = np.argsort(class_samples)[-n_classes:]
        reduced_matrix = conf_matrix[top_classes][:, top_classes]
        normalized_matrix = reduced_matrix / (reduced_matrix.sum(axis=1, keepdims=True) + 1e-6)
        class_labels = [class_names.get(i, f'Class {i}') for i in top_classes]

        conf_text = "Confusion Matrix for Top 20 Classes:\n\n"
        for i, true_label in enumerate(class_labels):
            conf_text += f"True Class: {true_label}\n"
            for j, pred_label in enumerate(class_labels):
                count = reduced_matrix[i, j]
                percentage = normalized_matrix[i, j] * 100
                if count > 0:
                    conf_text += f"  -> {pred_label}: {count} ({percentage:.1f}%)\n"
            conf_text += "\n"

        self.writer.add_text('Confusion_Matrix/Top20_Details', conf_text)

        fig, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(
            normalized_matrix,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax
        )
        plt.title('Normalized Confusion Matrix (Top 20 Classes by Sample Count)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()

        self.writer.add_figure('Confusion_Matrix/Top20_Normalized', fig)
        plt.close(fig)

    def evaluate(self, val_loader, class_names):
        all_preds = []
        all_labels = []
        all_image_ids = []
        incorrect_predictions = []

        with torch.no_grad():
            for images, labels, image_ids in tqdm(val_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                incorrect_mask = preds != labels
                for i in range(len(incorrect_mask)):
                    if incorrect_mask[i]:
                        incorrect_predictions.append({
                            'image_id': image_ids[i].item(),
                            'true_label': labels[i].item(),
                            'pred_label': preds[i].item()
                        })

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_image_ids.extend(image_ids.numpy())

        report = classification_report(
            all_labels,
            all_preds,
            output_dict=True,
            zero_division=0
        )

        class_metrics = {
            'precision': np.zeros(len(class_names)),
            'recall': np.zeros(len(class_names)),
            'f1': np.zeros(len(class_names)),
            'support': np.zeros(len(class_names))
        }

        for class_idx in range(len(class_names)):
            if str(class_idx) in report:
                class_metrics['precision'][class_idx] = report[str(class_idx)]['precision']
                class_metrics['recall'][class_idx] = report[str(class_idx)]['recall']
                class_metrics['f1'][class_idx] = report[str(class_idx)]['f1-score']
                class_metrics['support'][class_idx] = report[str(class_idx)]['support']

        conf_matrix = confusion_matrix(all_labels, all_preds)
        self.log_confusion_matrix(conf_matrix, class_names)
        self.log_metrics(class_metrics, class_names)

        metrics_summary = {
            'Accuracy': (np.array(all_preds) == np.array(all_labels)).mean(),
            'Macro_Precision': report['macro avg']['precision'],
            'Macro_Recall': report['macro avg']['recall'],
            'Macro_F1': report['macro avg']['f1-score'],
            'Weighted_Precision': report['weighted avg']['precision'],
            'Weighted_Recall': report['weighted avg']['recall'],
            'Weighted_F1': report['weighted avg']['f1-score']
        }

        print("\nPerformance Metrics:")
        print("-" * 50)
        for metric, value in metrics_summary.items():
            print(f"{metric}: {value:.4f}")
        print("-" * 50)

        summary_text = "Performance Metrics:\n\n"
        summary_text += "-" * 30 + "\n"
        for metric, value in metrics_summary.items():
            summary_text += f"{metric}: {value:.4f}\n"
        summary_text += "-" * 30 + "\n"

        self.writer.add_text('Summary/Overall_Performance', summary_text)

        self.writer.close()

def main():
    model_path = Model_pth
    val_root = os.path.abspath('COCO_datasets/coco2017/val2017')
    val_annot = os.path.abspath('COCO_datasets/coco2017/annotations/instances_val2017.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_dataset = COCOValDataset(
        root_dir=val_root,
        annotation_file=val_annot,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    evaluator = ModelEvaluator(model_path, device)
    evaluator.evaluate(val_loader, val_dataset.class_names)


if __name__ == '__main__':
    main()