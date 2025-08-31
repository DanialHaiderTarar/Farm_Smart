"""
Test Vision Transformer for wheat grain classification
Loads trained model and evaluates on test dataset
Classifies grains as bad_seed, healthy_seed, or impurity
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
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
import os
warnings.filterwarnings('ignore')

class WheatGrainDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(224, 224)):
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
                print(f"Warning: Failed to load {img_path}, using dummy image")
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

def load_trained_model(model_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    model_name = checkpoint.get('model_name', 'vit_small_patch16_224')
    config = checkpoint.get('config', {})
    num_classes = config.get('num_classes', 3)
    
    print(f"Model architecture: {model_name}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Training epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'Unknown'):.4f}")
    
    return model, checkpoint

def evaluate_model(model, data_loader, device, mixed_precision=True):
    """Evaluate model on dataset"""
    model.eval()
    
    predictions = []
    true_labels = []
    probabilities = []
    image_paths = []
    
    total_time = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Testing")):
            batch_start = time.time()
            
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Forward pass
            if mixed_precision:
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Store results
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            
            batch_time = time.time() - batch_start
            total_time += batch_time
    
    avg_time_per_batch = total_time / len(data_loader)
    avg_time_per_image = total_time / len(true_labels)
    
    print(f"Inference completed!")
    print(f"Average time per batch: {avg_time_per_batch:.4f}s")
    print(f"Average time per image: {avg_time_per_image:.4f}s")
    print(f"Total inference time: {total_time:.2f}s")
    
    return predictions, true_labels, probabilities

def generate_detailed_report(predictions, true_labels, probabilities, class_names, output_dir):
    """Generate comprehensive evaluation report"""
    
    # Calculate metrics
    test_accuracy = accuracy_score(true_labels, predictions)
    
    print(f"\n{'='*60}")
    print(f"WHEAT GRAIN CLASSIFICATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Overall Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Classification report
    print(f"\n{'-'*40}")
    print("DETAILED CLASSIFICATION REPORT:")
    print(f"{'-'*40}")
    report = classification_report(true_labels, predictions, target_names=class_names, digits=4)
    print(report)
    
    # Per-class accuracy
    print(f"\n{'-'*30}")
    print("PER-CLASS ACCURACY:")
    print(f"{'-'*30}")
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        class_mask = np.array(true_labels) == i
        if np.sum(class_mask) > 0:
            class_predictions = np.array(predictions)[class_mask]
            class_true = np.array(true_labels)[class_mask]
            class_acc = accuracy_score(class_true, class_predictions)
            per_class_acc[class_name] = class_acc
            print(f"  {class_name:<15}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    
    print(f"\n{'-'*25}")
    print("CONFUSION MATRIX:")
    print(f"{'-'*25}")
    print("Format: Predicted → | True ↓")
    print(f"{'':>15}", end="")
    for name in class_names:
        print(f"{name[:8]:>10}", end="")
    print()
    
    for i, true_name in enumerate(class_names):
        print(f"{true_name[:12]:>15}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>10}", end="")
        print()
    
    # Error analysis
    print(f"\n{'-'*30}")
    print("ERROR ANALYSIS:")
    print(f"{'-'*30}")
    
    misclassified = []
    for i in range(len(true_labels)):
        if true_labels[i] != predictions[i]:
            confidence = probabilities[i][predictions[i]]
            misclassified.append({
                'true_class': class_names[true_labels[i]],
                'predicted_class': class_names[predictions[i]],
                'confidence': confidence
            })
    
    print(f"Total misclassified samples: {len(misclassified)}")
    print(f"Misclassification rate: {len(misclassified)/len(true_labels)*100:.2f}%")
    
    # Most common misclassifications
    if misclassified:
        misclass_types = {}
        for error in misclassified:
            error_type = f"{error['true_class']} → {error['predicted_class']}"
            if error_type not in misclass_types:
                misclass_types[error_type] = []
            misclass_types[error_type].append(error['confidence'])
        
        print(f"\nMost common misclassification patterns:")
        sorted_errors = sorted(misclass_types.items(), key=lambda x: len(x[1]), reverse=True)
        for error_type, confidences in sorted_errors[:5]:
            avg_conf = np.mean(confidences)
            print(f"  {error_type}: {len(confidences)} cases (avg confidence: {avg_conf:.3f})")
    
    # Confidence analysis
    print(f"\n{'-'*35}")
    print("PREDICTION CONFIDENCE ANALYSIS:")
    print(f"{'-'*35}")
    
    confidences = []
    correct_confidences = []
    incorrect_confidences = []
    
    for i in range(len(predictions)):
        pred_confidence = probabilities[i][predictions[i]]
        confidences.append(pred_confidence)
        
        if predictions[i] == true_labels[i]:
            correct_confidences.append(pred_confidence)
        else:
            incorrect_confidences.append(pred_confidence)
    
    print(f"Average prediction confidence: {np.mean(confidences):.4f}")
    print(f"Average confidence (correct predictions): {np.mean(correct_confidences):.4f}")
    if incorrect_confidences:
        print(f"Average confidence (incorrect predictions): {np.mean(incorrect_confidences):.4f}")
    
    # Confidence distribution
    high_conf_correct = sum(1 for c in correct_confidences if c > 0.9)
    low_conf_correct = sum(1 for c in correct_confidences if c < 0.7)
    high_conf_incorrect = sum(1 for c in incorrect_confidences if c > 0.9)
    
    print(f"\nConfidence distribution:")
    print(f"  High confidence (>0.9) correct predictions: {high_conf_correct}")
    print(f"  Low confidence (<0.7) correct predictions: {low_conf_correct}")
    print(f"  High confidence (>0.9) incorrect predictions: {high_conf_incorrect}")
    
    # Create visualizations
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    plt.title('Confusion Matrix - Test Results', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    confusion_matrix_path = output_dir / 'confusion_matrix_test.png'
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved: {confusion_matrix_path}")
    
    # 2. Per-class accuracy bar plot
    plt.figure(figsize=(10, 6))
    classes = list(per_class_acc.keys())
    accuracies = list(per_class_acc.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = plt.bar(classes, accuracies, color=colors, alpha=0.8, edgecolor='black')
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.ylim(0, 1.1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    accuracy_plot_path = output_dir / 'per_class_accuracy_test.png'
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class accuracy plot saved: {accuracy_plot_path}")
    
    # 3. Confidence distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(correct_confidences, bins=30, alpha=0.7, color='green', label='Correct', density=True)
    if incorrect_confidences:
        plt.hist(incorrect_confidences, bins=30, alpha=0.7, color='red', label='Incorrect', density=True)
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Density')
    plt.title('Confidence Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    confidence_ranges = ['0.0-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']
    correct_counts = [
        sum(1 for c in correct_confidences if 0.0 <= c < 0.5),
        sum(1 for c in correct_confidences if 0.5 <= c < 0.7),
        sum(1 for c in correct_confidences if 0.7 <= c < 0.9),
        sum(1 for c in correct_confidences if 0.9 <= c <= 1.0)
    ]
    incorrect_counts = [
        sum(1 for c in incorrect_confidences if 0.0 <= c < 0.5),
        sum(1 for c in incorrect_confidences if 0.5 <= c < 0.7),
        sum(1 for c in incorrect_confidences if 0.7 <= c < 0.9),
        sum(1 for c in incorrect_confidences if 0.9 <= c <= 1.0)
    ]
    
    x = np.arange(len(confidence_ranges))
    width = 0.35
    
    plt.bar(x - width/2, correct_counts, width, label='Correct', color='green', alpha=0.7)
    plt.bar(x + width/2, incorrect_counts, width, label='Incorrect', color='red', alpha=0.7)
    plt.xlabel('Confidence Range')
    plt.ylabel('Count')
    plt.title('Predictions by Confidence Range')
    plt.xticks(x, confidence_ranges)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    confidence_plot_path = output_dir / 'confidence_analysis_test.png'
    plt.savefig(confidence_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confidence analysis plot saved: {confidence_plot_path}")
    
    # Save detailed results to JSON
    results = {
        'timestamp': datetime.now().isoformat(),
        'overall_accuracy': float(test_accuracy),
        'per_class_accuracy': per_class_acc,
        'total_samples': len(true_labels),
        'misclassified_samples': len(misclassified),
        'misclassification_rate': len(misclassified)/len(true_labels),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'confidence_stats': {
            'average_confidence': float(np.mean(confidences)),
            'correct_predictions_avg_confidence': float(np.mean(correct_confidences)),
            'incorrect_predictions_avg_confidence': float(np.mean(incorrect_confidences)) if incorrect_confidences else 0.0,
            'high_confidence_correct': int(high_conf_correct),
            'low_confidence_correct': int(low_conf_correct),
            'high_confidence_incorrect': int(high_conf_incorrect)
        }
    }
    
    results_path = output_dir / f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detailed results saved: {results_path}")
    print(f"\n{'='*60}")
    print("TESTING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    
    return results

def test_wheat_classifier(
    model_path,
    test_data_dir,
    batch_size=128,
    img_size=224,
    mixed_precision=True,
    num_workers=0,  # Set to 0 to avoid multiprocessing issues
    output_dir="./test_results"
):
    """Test the trained wheat grain classifier"""
    
    print("="*60)
    print("WHEAT GRAIN CLASSIFIER - TEST MODE")
    print("="*60)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"Test data directory: {test_data_dir}")
    print(f"Model path: {model_path}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Mixed precision: {mixed_precision}")
    print(f"Output directory: {output_dir}")
    
    # Load trained model
    model, checkpoint = load_trained_model(model_path, device)
    
    # Get class names from checkpoint or use default
    class_names = checkpoint.get('config', {}).get('class_names', ['bad_seed', 'healthy_seed', 'impurity'])
    print(f"Class names: {class_names}")
    
    # Data transforms (same as validation transforms)
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    test_dataset = WheatGrainDataset(
        data_dir=test_data_dir,
        transform=test_transform,
        target_size=(img_size, img_size)
    )
    
    if len(test_dataset) == 0:
        raise ValueError(f"No test images found in {test_data_dir}")
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nStarting evaluation...")
    start_time = time.time()
    
    # Run evaluation
    predictions, true_labels, probabilities = evaluate_model(
        model, test_loader, device, mixed_precision
    )
    
    evaluation_time = time.time() - start_time
    print(f"Total evaluation time: {evaluation_time:.2f} seconds")
    
    # Generate comprehensive report
    results = generate_detailed_report(
        predictions, true_labels, probabilities, class_names, output_dir
    )
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained Vision Transformer for wheat grain classification')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--test-data-dir', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for testing')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                       help='Use mixed precision inference')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loader workers (0 to avoid multiprocessing issues)')
    parser.add_argument('--output-dir', type=str, default='./test_results',
                       help='Output directory for test results')
    
    args = parser.parse_args()
    
    try:
        # Run testing
        results = test_wheat_classifier(
            model_path=args.model_path,
            test_data_dir=args.test_data_dir,
            batch_size=args.batch_size,
            img_size=args.img_size,
            mixed_precision=args.mixed_precision,
            num_workers=args.num_workers,
            output_dir=args.output_dir
        )
        
        print(f"\nTesting completed successfully!")
        print(f"Results saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()