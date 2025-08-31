import os
import cv2
import numpy as np
import argparse
import json
import shutil
import gc
from pathlib import Path
from typing import Dict, Any, Tuple, List
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


class WheatDatasetPreparer:
   def __init__(self):
       self.class_names = {
           0: 'bad_seed',
           1: 'healthy_seed', 
           2: 'impurity'
       }
       self.max_image_size = 4096
       self.grain_size = 224
       self.memory_batch_size = 50
       
   def extract_and_save_grains(self, 
                              data_dir: str, 
                              labels_dir: str, 
                              output_dir: str,
                              train_split: float = 0.7,
                              val_split: float = 0.2,
                              batch_size: int = 50) -> Dict[str, Any]:
       
       print(f"Starting dataset preparation with {batch_size} images per batch...")
       print(f"RAM: 32GB detected - using optimized settings")
       
       grain_count = defaultdict(int)
       total_grains = 0
       processed_images = 0
       skipped_images = 0
       
       for split in ['train', 'val', 'test']:
           for class_name in self.class_names.values():
               Path(output_dir, split, class_name).mkdir(parents=True, exist_ok=True)
       
       image_files = []
       for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
           image_files.extend(list(Path(data_dir).glob(ext)))
           image_files.extend(list(Path(data_dir).glob(ext.upper())))
       
       print(f"Found {len(image_files)} images to process")
       
       if not image_files:
           raise ValueError(f"No image files found in {data_dir}")
       
       total_batches = (len(image_files) + batch_size - 1) // batch_size
       
       for batch_idx in range(total_batches):
           batch_start = batch_idx * batch_size
           batch_end = min(batch_start + batch_size, len(image_files))
           batch_files = image_files[batch_start:batch_end]
           
           print(f"\nProcessing batch {batch_idx + 1}/{total_batches} ({len(batch_files)} images)")
           
           batch_images = {}
           successful_loads = 0
           
           for img_path in batch_files:
               try:
                   image = self._safe_read_image(str(img_path))
                   if image is not None:
                       batch_images[img_path] = image
                       successful_loads += 1
                   else:
                       skipped_images += 1
                       print(f"Skipped: {img_path.name}")
                       
               except Exception as e:
                   print(f"Error loading {img_path.name}: {str(e)}")
                   skipped_images += 1
                   continue
           
           print(f"Successfully loaded {successful_loads}/{len(batch_files)} images in batch")
           
           for img_path, image in batch_images.items():
               try:
                   label_path = Path(labels_dir) / f"{img_path.stem}.txt"
                   if not label_path.exists():
                       print(f"No label file for {img_path.name}")
                       continue
                   
                   grains_extracted = self._extract_grains_from_image(
                       image, label_path, img_path, output_dir, 
                       grain_count, train_split, val_split
                   )
                   
                   total_grains += grains_extracted
                   processed_images += 1
                   
                   if processed_images % 10 == 0:
                       print(f"Processed: {processed_images} images, Extracted: {total_grains} grains")
                       
               except Exception as e:
                   print(f"Error processing {img_path.name}: {str(e)}")
                   continue
           
           batch_images.clear()
           gc.collect()
           
           print(f"Batch {batch_idx + 1} completed, memory cleaned")
       
       dataset_info = self._generate_dataset_info(
           output_dir, grain_count, processed_images, skipped_images, total_grains
       )
       
       info_path = Path(output_dir) / "dataset_info.json"
       with open(info_path, 'w') as f:
           json.dump(dataset_info, f, indent=2)
       
       self._create_visualizations(output_dir, grain_count)
       
       print("\n" + "="*60)
       print("DATASET PREPARATION COMPLETED!")
       print(f"Total images processed: {processed_images}")
       print(f"Total grains extracted: {total_grains}")
       print(f"Grains per class: {dict(grain_count)}")
       print(f"Dataset saved to: {output_dir}")
       print("="*60)
       
       return dataset_info
   
   def _safe_read_image(self, image_path: str):
       try:
           image = cv2.imread(image_path)
           
           if image is None:
               return None
               
           h, w = image.shape[:2]
           
           if max(h, w) > self.max_image_size:
               if h > w:
                   new_h = self.max_image_size
                   new_w = int(w * self.max_image_size / h)
               else:
                   new_w = self.max_image_size
                   new_h = int(h * self.max_image_size / w)
               
               image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
               print(f"Resized large image from {w}x{h} to {new_w}x{new_h}")
           
           return image
           
       except Exception as e:
           print(f"Error reading image {image_path}: {str(e)}")
           return None
   
   def _extract_grains_from_image(self, image, label_path, img_path, output_dir, 
                                 grain_count, train_split, val_split):
       h, w = image.shape[:2]
       grains_extracted = 0
       
       try:
           with open(label_path, 'r') as f:
               lines = f.readlines()
       except Exception as e:
           print(f"Error reading label file {label_path}: {str(e)}")
           return 0
       
       for line_idx, line in enumerate(lines):
           parts = line.strip().split()
           if len(parts) < 5:
               continue
           
           try:
               class_id = int(parts[0])
               if class_id not in [0, 1, 2]:
                   continue
               
               x_center, y_center, width, height = map(float, parts[1:5])
               
               x1 = int((x_center - width/2) * w)
               y1 = int((y_center - height/2) * h)
               x2 = int((x_center + width/2) * w)
               y2 = int((y_center + height/2) * h)
               
               padding = 15
               x1 = max(0, x1 - padding)
               y1 = max(0, y1 - padding)
               x2 = min(w, x2 + padding)
               y2 = min(h, y2 + padding)
               
               if x2 <= x1 or y2 <= y1:
                   continue
               
               grain_img = image[y1:y2, x1:x2]
               
               if grain_img.size == 0:
                   continue
               
               if min(grain_img.shape[:2]) < 20:
                   continue
               
               grain_img = self._enhance_grain_image(grain_img)
               grain_img = cv2.resize(grain_img, (self.grain_size, self.grain_size), interpolation=cv2.INTER_LANCZOS4)
               
               class_name = self.class_names[class_id]
               grain_count[class_name] += 1
               
               split = self._determine_split(grain_count[class_name], train_split, val_split)
               
               save_dir = Path(output_dir) / split / class_name
               save_dir.mkdir(parents=True, exist_ok=True)
               
               filename = f"{img_path.stem}_grain_{line_idx}_{grain_count[class_name]:04d}.jpg"
               save_path = save_dir / filename
               
               success = cv2.imwrite(str(save_path), grain_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
               
               if success:
                   grains_extracted += 1
               else:
                   print(f"Failed to save grain: {save_path}")
                   
           except Exception as e:
               print(f"Error processing grain {line_idx} in {img_path.name}: {str(e)}")
               continue
       
       return grains_extracted
   
   def _enhance_grain_image(self, grain_img):
       try:
           lab = cv2.cvtColor(grain_img, cv2.COLOR_BGR2LAB)
           clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
           lab[:, :, 0] = clahe.apply(lab[:, :, 0])
           enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
           enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
           return enhanced
       except Exception:
           return grain_img
   
   def _determine_split(self, count: int, train_split: float, val_split: float) -> str:
       test_split = 1.0 - train_split - val_split
       
       if count % 10 < train_split * 10:
           return 'train'
       elif count % 10 < (train_split + val_split) * 10:
           return 'val'
       else:
           return 'test'
   
   def _generate_dataset_info(self, output_dir, grain_count, processed_images, skipped_images, total_grains):
       dataset_info = {
           'dataset_name': 'Wheat Grain Classification Dataset',
           'creation_date': str(Path().cwd()),
           'total_images_processed': processed_images,
           'total_images_skipped': skipped_images,
           'total_grains_extracted': total_grains,
           'classes': list(self.class_names.values()),
           'class_distribution': dict(grain_count),
           'grain_image_size': f"{self.grain_size}x{self.grain_size}",
           'splits': {}
       }
       
       for split in ['train', 'val', 'test']:
           split_counts = {}
           split_total = 0
           
           for class_name in self.class_names.values():
               split_dir = Path(output_dir) / split / class_name
               if split_dir.exists():
                   count = len(list(split_dir.glob('*.jpg')))
                   split_counts[class_name] = count
                   split_total += count
               else:
                   split_counts[class_name] = 0
           
           dataset_info['splits'][split] = {
               'total': split_total,
               'classes': split_counts
           }
       
       return dataset_info
   
   def _create_visualizations(self, output_dir, grain_count):
       try:
           plt.style.use('seaborn-v0_8')
       except:
           plt.style.use('default')
       
       fig, axes = plt.subplots(2, 2, figsize=(15, 12))
       fig.suptitle('Wheat Grain Dataset Analysis', fontsize=16, fontweight='bold')
       
       classes = list(grain_count.keys())
       counts = list(grain_count.values())
       colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
       
       axes[0, 0].bar(classes, counts, color=colors[:len(classes)])
       axes[0, 0].set_title('Class Distribution')
       axes[0, 0].set_ylabel('Number of Grains')
       for i, v in enumerate(counts):
           axes[0, 0].text(i, v + max(counts)*0.01, str(v), ha='center', va='bottom')
       
       axes[0, 1].pie(counts, labels=classes, autopct='%1.1f%%', colors=colors[:len(classes)])
       axes[0, 1].set_title('Class Percentage Distribution')
       
       split_data = {}
       for split in ['train', 'val', 'test']:
           split_counts = []
           for class_name in classes:
               split_dir = Path(output_dir) / split / class_name
               if split_dir.exists():
                   count = len(list(split_dir.glob('*.jpg')))
                   split_counts.append(count)
               else:
                   split_counts.append(0)
           split_data[split] = split_counts
       
       x = np.arange(len(classes))
       width = 0.25
       
       for i, (split, split_counts) in enumerate(split_data.items()):
           axes[1, 0].bar(x + i*width, split_counts, width, label=split, alpha=0.8)
       
       axes[1, 0].set_xlabel('Classes')
       axes[1, 0].set_ylabel('Number of Grains')
       axes[1, 0].set_title('Train/Val/Test Split Distribution')
       axes[1, 0].set_xticks(x + width)
       axes[1, 0].set_xticklabels(classes)
       axes[1, 0].legend()
       
       balance_scores = []
       for class_name in classes:
           class_count = grain_count[class_name]
           balance_score = class_count / max(counts) if max(counts) > 0 else 0
           balance_scores.append(balance_score * 100)
       
       bars = axes[1, 1].bar(classes, balance_scores, color=colors[:len(classes)], alpha=0.7)
       axes[1, 1].set_title('Class Balance Score')
       axes[1, 1].set_ylabel('Balance Score (%)')
       axes[1, 1].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Good Balance (80%)')
       axes[1, 1].legend()
       
       for bar, score in zip(bars, balance_scores):
           height = bar.get_height()
           axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                          f'{score:.1f}%', ha='center', va='bottom')
       
       plt.tight_layout()
       
       viz_path = Path(output_dir) / 'dataset_analysis.png'
       plt.savefig(viz_path, dpi=300, bbox_inches='tight')
       plt.close()
       
       self._create_sample_grid(output_dir)
       
       print(f"Visualizations saved to: {viz_path}")
   
   def _create_sample_grid(self, output_dir):
       try:
           fig, axes = plt.subplots(3, 6, figsize=(18, 9))
           fig.suptitle('Sample Grains from Each Class', fontsize=16, fontweight='bold')
           
           for class_idx, class_name in enumerate(self.class_names.values()):
               train_dir = Path(output_dir) / 'train' / class_name
               
               if train_dir.exists():
                   sample_files = list(train_dir.glob('*.jpg'))[:6]
                   
                   for sample_idx in range(6):
                       ax = axes[class_idx, sample_idx]
                       
                       if sample_idx < len(sample_files):
                           img = cv2.imread(str(sample_files[sample_idx]))
                           if img is not None:
                               img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                               ax.imshow(img_rgb)
                               ax.set_title(f'{class_name}_{sample_idx+1}', fontsize=8)
                           else:
                               ax.text(0.5, 0.5, 'Load Error', ha='center', va='center')
                       else:
                           ax.text(0.5, 0.5, 'No Sample', ha='center', va='center')
                       
                       ax.axis('off')
               else:
                   for sample_idx in range(6):
                       ax = axes[class_idx, sample_idx]
                       ax.text(0.5, 0.5, f'No {class_name}\nSamples', ha='center', va='center')
                       ax.axis('off')
           
           plt.tight_layout()
           
           sample_path = Path(output_dir) / 'sample_grains.png'
           plt.savefig(sample_path, dpi=300, bbox_inches='tight')
           plt.close()
           
           print(f"Sample grid saved to: {sample_path}")
           
       except Exception as e:
           print(f"Error creating sample grid: {str(e)}")


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Prepare wheat grain dataset for ViT training")
   parser.add_argument("--data-dir", required=True, help="Directory containing training images")
   parser.add_argument("--labels-dir", required=True, help="Directory containing YOLO label files")
   parser.add_argument("--output-dir", default="data/classification", help="Output directory for processed dataset")
   parser.add_argument("--train-split", type=float, default=0.7, help="Training split ratio")
   parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
   parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing (optimal for 32GB RAM)")
   
   args = parser.parse_args()
   
   if args.train_split + args.val_split >= 1.0:
       raise ValueError("train_split + val_split must be < 1.0")
   
   print("=" * 60)
   print("WHEAT GRAIN DATASET PREPARATION")
   print("=" * 60)
   print(f"Data directory: {args.data_dir}")
   print(f"Labels directory: {args.labels_dir}")
   print(f"Output directory: {args.output_dir}")
   print(f"Splits: Train={args.train_split}, Val={args.val_split}, Test={1-args.train_split-args.val_split}")
   print(f"Batch size: {args.batch_size} (optimized for 32GB RAM)")
   print("=" * 60)
   
   preparer = WheatDatasetPreparer()
   
   dataset_info = preparer.extract_and_save_grains(
       data_dir=args.data_dir,
       labels_dir=args.labels_dir,
       output_dir=args.output_dir,
       train_split=args.train_split,
       val_split=args.val_split,
       batch_size=args.batch_size
   )
   
   print("\nDataset preparation completed successfully!")
   print(f"You can now train the model using:")
   print(f"python scripts/train_classifier.py --data-dir {args.output_dir}")