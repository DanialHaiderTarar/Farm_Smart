"""
Train Vision Transformer for wheat grain classification
Classifies grains as bad_seed, healthy_seed, or impurity
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import timm
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import warnings
import time
import gc
import multiprocessing as mp
warnings.filterwarnings('ignore')

class WheatGrainDataset(Dataset):
   def __init__(self, data_dir, transform=None, target_size=(192, 192), subset_ratio=1.0):
       """Dataset for wheat grain classification"""
       self.data_dir = Path(data_dir)
       self.transform = transform
       self.target_size = target_size
       
       # Class mapping
       self.class_to_idx = {'bad_seed': 0, 'healthy_seed': 1, 'impurity': 2}
       self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
       
       # Load all images
       self.samples = []
       for class_name, class_idx in self.class_to_idx.items():
           class_dir = self.data_dir / class_name
           if class_dir.exists():
               for img_path in class_dir.glob("*.jpg"):
                   self.samples.append((str(img_path), class_idx))
       
       # Apply subset if needed
       if subset_ratio < 1.0:
           subset_size = int(len(self.samples) * subset_ratio)
           np.random.seed(42)
           indices = np.random.choice(len(self.samples), subset_size, replace=False)
           self.samples = [self.samples[i] for i in indices]
       
       print(f"Loaded {len(self.samples)} grain images from {data_dir}")
       self._print_class_distribution()
   
   def _print_class_distribution(self):
       """Print distribution of classes"""
       dist = {0: 0, 1: 0, 2: 0}
       for _, label in self.samples:
           dist[label] += 1
       
       print("Class distribution:")
       for idx, count in dist.items():
           class_name = self.idx_to_class[idx]
           print(f"  {class_name}: {count} samples")
   
   def __len__(self):
       return len(self.samples)
   
   def __getitem__(self, idx):
       img_path, label = self.samples[idx]
       
       try:
           # Load image with error handling
           image = cv2.imread(img_path)
           if image is None:
               # Return dummy data if image fails to load
               image = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
           else:
               image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
               # Resize
               image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
       except Exception as e:
           print(f"Error loading {img_path}: {e}")
           image = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
       
       if self.transform:
           image = self.transform(image)
       
       return image, label

def create_model(model_name='vit_small_patch16_224', num_classes=3, pretrained=True):
   """Create optimized model for classification"""
   print(f"Creating model: {model_name}")
   
   if model_name.startswith('vit'):
       # Vision Transformer - use smaller model for speed
       model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
       
   elif model_name.startswith('efficientnet'):
       # EfficientNet
       model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
       
   elif model_name.startswith('convnext'):
       # ConvNeXt
       model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
       
   else:
       raise ValueError(f"Unsupported model: {model_name}")
   
   return model

class EarlyStopping:
   def __init__(self, patience=7, min_delta=0.001):
       self.patience = patience
       self.min_delta = min_delta
       self.counter = 0
       self.best_loss = None
       
   def __call__(self, val_loss):
       if self.best_loss is None:
           self.best_loss = val_loss
       elif val_loss < self.best_loss - self.min_delta:
           self.best_loss = val_loss
           self.counter = 0
       else:
           self.counter += 1
           
       return self.counter >= self.patience

def train_classifier(
   data_dir="../data/classification",
   model_name="vit_small_patch16_224",
   epochs=30,
   batch_size=256,
   lr=0.001,
   img_size=192,
   output_dir="../models",
   subset_ratio=1.0,
   mixed_precision=True,
   num_workers=12
):
   """Train the optimized Vision Transformer classifier"""
   
   # Device configuration
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")
   print(f"Model: {model_name}")
   print(f"Image size: {img_size}x{img_size}")
   print(f"Batch size: {batch_size}")
   print(f"Mixed precision: {mixed_precision}")
   print(f"Number of workers: {num_workers}")
   print(f"Subset ratio: {subset_ratio}")
   
   # Optimized data transforms
   train_transform = transforms.Compose([
       transforms.ToPILImage(),
       transforms.Resize((img_size + 32, img_size + 32)),
       transforms.RandomCrop((img_size, img_size)),
       # Essential augmentations only
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.RandomVerticalFlip(p=0.3),
       transforms.RandomRotation(degrees=10),
       transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
       # Convert to tensor early
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       # Light augmentation
       transforms.RandomErasing(p=0.1, scale=(0.02, 0.2))
   ])
   
   val_transform = transforms.Compose([
       transforms.ToPILImage(),
       transforms.Resize((img_size, img_size)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   
   # Create datasets
   train_dataset = WheatGrainDataset(
       data_dir=Path(data_dir) / "train",
       transform=train_transform,
       target_size=(img_size, img_size),
       subset_ratio=subset_ratio
   )
   
   val_dataset = WheatGrainDataset(
       data_dir=Path(data_dir) / "val",
       transform=val_transform,
       target_size=(img_size, img_size),
       subset_ratio=subset_ratio
   )
   
   test_dataset = WheatGrainDataset(
       data_dir=Path(data_dir) / "test",
       transform=val_transform,
       target_size=(img_size, img_size),
       subset_ratio=subset_ratio
   )
   
   # Optimized data loaders
   train_loader = DataLoader(
       train_dataset, 
       batch_size=batch_size, 
       shuffle=True, 
       num_workers=num_workers, 
       pin_memory=True, 
       persistent_workers=True,
       prefetch_factor=2,
       drop_last=True
   )
   
   val_loader = DataLoader(
       val_dataset, 
       batch_size=batch_size, 
       shuffle=False,
       num_workers=num_workers, 
       pin_memory=True,
       persistent_workers=True,
       prefetch_factor=2
   )
   
   test_loader = DataLoader(
       test_dataset, 
       batch_size=batch_size, 
       shuffle=False,
       num_workers=num_workers, 
       pin_memory=True,
       persistent_workers=True,
       prefetch_factor=2
   )
   
   # Create model
   model = create_model(model_name, num_classes=3, pretrained=True)
   model = model.to(device)
   
   # Enable optimizations
   #if hasattr(torch, 'compile'):
       #model = torch.compile(model)
   
   # Model summary
   total_params = sum(p.numel() for p in model.parameters())
   trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   print(f"Total parameters: {total_params:,}")
   print(f"Trainable parameters: {trainable_params:,}")
   
   # Loss function and optimizer
   criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
   
   # Optimized optimizer settings
   optimizer = optim.AdamW(
       model.parameters(), 
       lr=lr, 
       weight_decay=0.05, 
       betas=(0.9, 0.999),
       eps=1e-8
   )
   
   # OneCycle scheduler for faster convergence
   scheduler = optim.lr_scheduler.OneCycleLR(
       optimizer,
       max_lr=lr * 2,
       epochs=epochs,
       steps_per_epoch=len(train_loader),
       pct_start=0.1,
       anneal_strategy='cos'
   )
   
   # Mixed precision scaler
   scaler = GradScaler() if mixed_precision else None
   
   # Early stopping
   early_stopping = EarlyStopping(patience=7)
   
   # Training history
   history = {
       'train_loss': [], 'train_acc': [],
       'val_loss': [], 'val_acc': []
   }
   
   best_val_acc = 0
   model_save_path = Path(output_dir) / f"wheat_classifier_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
   model_save_path.parent.mkdir(parents=True, exist_ok=True)
   
   # Training loop
   print(f"\nStarting optimized training...")
   print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}")
   
   total_start_time = time.time()
   
   for epoch in range(epochs):
       epoch_start_time = time.time()
       
       # Training phase
       model.train()
       train_loss = 0
       train_correct = 0
       train_total = 0
       
       pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
       for batch_idx, (images, labels) in enumerate(pbar):
           images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
           
           optimizer.zero_grad()
           
           if mixed_precision:
               with autocast():
                   outputs = model(images)
                   loss = criterion(outputs, labels)
               
               scaler.scale(loss).backward()
               scaler.unscale_(optimizer)
               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
               scaler.step(optimizer)
               scaler.update()
           else:
               outputs = model(images)
               loss = criterion(outputs, labels)
               loss.backward()
               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
               optimizer.step()
           
           scheduler.step()
           
           train_loss += loss.item() * images.size(0)
           _, predicted = torch.max(outputs, 1)
           train_total += labels.size(0)
           train_correct += (predicted == labels).sum().item()
           
           # Update progress bar every 50 batches
           if batch_idx % 50 == 0:
               current_lr = optimizer.param_groups[0]['lr']
               pbar.set_postfix({
                   'loss': f'{loss.item():.4f}',
                   'acc': f'{100 * train_correct / train_total:.1f}%',
                   'lr': f'{current_lr:.6f}'
               })
       
       # Validation phase
       model.eval()
       val_loss = 0
       val_correct = 0
       val_total = 0
       
       with torch.no_grad():
           for images, labels in val_loader:
               images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
               
               if mixed_precision:
                   with autocast():
                       outputs = model(images)
                       loss = criterion(outputs, labels)
               else:
                   outputs = model(images)
                   loss = criterion(outputs, labels)
               
               val_loss += loss.item() * images.size(0)
               _, predicted = torch.max(outputs, 1)
               val_total += labels.size(0)
               val_correct += (predicted == labels).sum().item()
       
       # Calculate epoch metrics
       train_loss = train_loss / len(train_dataset)
       train_acc = train_correct / train_total
       val_loss = val_loss / len(val_dataset)
       val_acc = val_correct / val_total
       
       # Store history
       history['train_loss'].append(train_loss)
       history['train_acc'].append(train_acc)
       history['val_loss'].append(val_loss)
       history['val_acc'].append(val_acc)
       
       epoch_time = time.time() - epoch_start_time
       
       print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
       print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
       print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
       
       # Save best model
       if val_acc > best_val_acc:
           best_val_acc = val_acc
           torch.save({
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'epoch': epoch,
               'val_acc': val_acc,
               'model_name': model_name,
               'class_to_idx': train_dataset.class_to_idx,
               'config': {
                   'img_size': img_size,
                   'num_classes': 3,
                   'model_name': model_name,
                   'class_names': ['bad_seed', 'healthy_seed', 'impurity']
               }
           }, model_save_path)
           print(f"  Saved best model with validation accuracy: {val_acc:.4f}")
       
       # Early stopping check
       if early_stopping(val_loss):
           print(f"Early stopping triggered at epoch {epoch+1}")
           break
       
       # Memory cleanup
       if (epoch + 1) % 5 == 0:
           gc.collect()
           torch.cuda.empty_cache()
   
   total_training_time = time.time() - total_start_time
   print(f"\nTraining completed in {total_training_time/3600:.2f} hours")
   
   # Test evaluation
   print(f"\nEvaluating on test set...")
   model.load_state_dict(torch.load(model_save_path)['model_state_dict'])
   model.eval()
   
   test_predictions = []
   test_labels = []
   test_probabilities = []
   
   with torch.no_grad():
       for images, labels in tqdm(test_loader, desc="Testing"):
           images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
           
           if mixed_precision:
               with autocast():
                   outputs = model(images)
           else:
               outputs = model(images)
               
           probabilities = torch.softmax(outputs, dim=1)
           _, predicted = torch.max(outputs, 1)
           
           test_predictions.extend(predicted.cpu().numpy())
           test_labels.extend(labels.cpu().numpy())
           test_probabilities.extend(probabilities.cpu().numpy())
  
   # Classification report
   class_names = ['bad_seed', 'healthy_seed', 'impurity']
   test_accuracy = accuracy_score(test_labels, test_predictions)
   
   print(f"\nTest Accuracy: {test_accuracy:.4f}")
   print("\nClassification Report:")
   print(classification_report(test_labels, test_predictions, target_names=class_names))
  
   # Confusion Matrix
   cm = confusion_matrix(test_labels, test_predictions)
   plt.figure(figsize=(10, 8))
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
   plt.title(f'Confusion Matrix - {model_name}')
   plt.ylabel('True Label')
   plt.xlabel('Predicted Label')
   plt.savefig(f'confusion_matrix_{model_name}.png', dpi=150, bbox_inches='tight')
   plt.close()
   
   # Plot training history
   plt.figure(figsize=(15, 5))
   
   plt.subplot(1, 3, 1)
   plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
   plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()
   plt.title('Training and Validation Loss')
   plt.grid(True, alpha=0.3)
   
   plt.subplot(1, 3, 2)
   plt.plot(history['train_acc'], label='Train Acc', linewidth=2)
   plt.plot(history['val_acc'], label='Val Acc', linewidth=2)
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.legend()
   plt.title('Training and Validation Accuracy')
   plt.grid(True, alpha=0.3)
   
   # Learning curve
   plt.subplot(1, 3, 3)
   epochs_range = range(1, len(history['train_acc']) + 1)
   plt.plot(epochs_range, history['train_acc'], 'b-', label='Training Accuracy')
   plt.plot(epochs_range, history['val_acc'], 'r-', label='Validation Accuracy')
   plt.fill_between(epochs_range, history['train_acc'], alpha=0.1, color='blue')
   plt.fill_between(epochs_range, history['val_acc'], alpha=0.1, color='red')
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.title('Model Performance')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.savefig(f'training_history_{model_name}.png', dpi=150, bbox_inches='tight')
   plt.close()
   
   # Per-class accuracy analysis
   per_class_acc = {}
   for i, class_name in enumerate(class_names):
       class_mask = np.array(test_labels) == i
       if np.sum(class_mask) > 0:
           class_predictions = np.array(test_predictions)[class_mask]
           class_true = np.array(test_labels)[class_mask]
           per_class_acc[class_name] = accuracy_score(class_true, class_predictions)
   
   print("\nPer-class Accuracy:")
   for class_name, acc in per_class_acc.items():
       print(f"  {class_name}: {acc:.4f}")
   
   # Save training results
   results = {
       'model_name': model_name,
       'best_val_accuracy': float(best_val_acc),
       'test_accuracy': float(test_accuracy),
       'per_class_accuracy': per_class_acc,
       'training_time_hours': total_training_time / 3600,
       'training_history': history,
       'model_path': str(model_save_path),
       'config': {
           'epochs': epochs,
           'batch_size': batch_size,
           'learning_rate': lr,
           'img_size': img_size,
           'mixed_precision': mixed_precision,
           'subset_ratio': subset_ratio
       }
   }
   
   results_path = Path(output_dir) / f"training_results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
   with open(results_path, 'w') as f:
       json.dump(results, f, indent=2, default=str)
   
   print(f"\nTraining completed successfully!")
   print(f"Best validation accuracy: {best_val_acc:.4f}")
   print(f"Test accuracy: {test_accuracy:.4f}")
   print(f"Training time: {total_training_time/3600:.2f} hours")
   print(f"Model saved: {model_save_path}")
   print(f"Results saved: {results_path}")
   
   return model, history

if __name__ == "__main__":
   mp.set_start_method('spawn',force=True)
   parser = argparse.ArgumentParser(description='Train optimized Vision Transformer for wheat grain classification')
   parser.add_argument('--data-dir', type=str, default='../data/classification',
                      help='Path to classification data directory')
   parser.add_argument('--model', type=str, default='vit_small_patch16_224',
                      choices=['vit_small_patch16_224', 'vit_base_patch16_224', 
                              'efficientnet_b0', 'efficientnet_b1', 
                              'convnext_tiny', 'convnext_small'],
                      help='Model architecture')
   parser.add_argument('--epochs', type=int, default=30,
                      help='Number of training epochs')
   parser.add_argument('--batch-size', type=int, default=256,
                      help='Batch size (optimized for 32GB RAM + 16GB GPU)')
   parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
   parser.add_argument('--img-size', type=int, default=192,
                      help='Input image size (192 for speed)')
   parser.add_argument('--output-dir', type=str, default='../models',
                      help='Output directory for saved models')
   parser.add_argument('--subset-ratio', type=float, default=1.0,
                      help='Use subset of data for testing (0.1 = 10%)')
   parser.add_argument('--mixed-precision', action='store_true', default=True,
                      help='Use mixed precision training')
   parser.add_argument('--num-workers', type=int, default=12,
                      help='Number of data loader workers')
  
   args = parser.parse_args()
  
   print("Starting Optimized Wheat Grain Classification Training")
   print(f"Model: {args.model}")
   print(f"Data: {args.data_dir}")
   print(f"Config: {args.epochs} epochs, batch size {args.batch_size}")
   print(f"Hardware optimized for: 32GB RAM + 16GB GPU")
   
   train_classifier(
       data_dir=args.data_dir,
       model_name=args.model,
       epochs=args.epochs,
       batch_size=args.batch_size,
       lr=args.lr,
       img_size=args.img_size,
       output_dir=args.output_dir,
       subset_ratio=args.subset_ratio,
       mixed_precision=args.mixed_precision,
       num_workers=args.num_workers
   )