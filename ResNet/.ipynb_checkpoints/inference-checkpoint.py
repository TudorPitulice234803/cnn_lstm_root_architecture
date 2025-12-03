import cv2
import numpy as np
import pandas as pd
from patchify import patchify, unpatchify
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from sklearn.metrics import jaccard_score, precision_score, recall_score
from sklearn.metrics import f1_score as f1_metric
import os

from preprocess import crop, padder, crop_to_coordinates
from helpers import f1
from root_measurement import segment_image, measuring_skeleton

def pipeline(image_dir, model_path, kernel_size=5, patch_size=256, view_masks=False, view_growth=True):
    if model_path.endswith('h5'):
        model = load_model(model_path, custom_objects={"f1": f1})
        
        all_f1 = []
        all_iou = []
        all_precision = []
        all_recall = []
        
        # Store root lengths for each plant across all images
        plant_root_lengths = {i: [] for i in range(5)}  # 5 plants
        image_names = []
        
        mask_dir = image_dir
        print(model_path)
    
        for file in sorted(os.listdir(image_dir)):
            if file.endswith('.png'):
                image_names.append(file)
                # Read and preprocess image
                img_path = os.path.join(image_dir, file)
                image = cv2.imread(img_path, 0)  # grayscale
                image_cropped, stats, centroids = crop(image, kernel_size=kernel_size)
                image_color = cv2.cvtColor(image_cropped, cv2.COLOR_GRAY2RGB)
                image_np = image_color / 255.0
    
                # Pad and patchify
                image_padded = padder(image_np, patch_size=patch_size)
                patches = patchify(image_padded, (patch_size, patch_size, 3), step=patch_size)
                expected_shape = (image_padded.shape[0] // patch_size, image_padded.shape[1] // patch_size)
                patches = patches.reshape(-1, patch_size, patch_size, 3)
    
                # Predict patches and unpatchify
                predicted_patches = model.predict(patches)
                predicted_patches = predicted_patches.reshape(expected_shape[0], expected_shape[1], patch_size, patch_size, predicted_patches.shape[-1])
                predicted_patches = predicted_patches[:, :, :, :, 0]
                prediction = unpatchify(predicted_patches, image_padded.shape[:2])
    
                # Threshold prediction to get binary mask
                prediction_bool = (prediction > 0.5).astype(np.uint8)
    
                # Load the actual mask and preprocess it to match prediction size
                mask_path = os.path.join(mask_dir, file).replace('.png', '.tif')
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, 0)  # grayscale mask
                    mask = crop_to_coordinates(mask, stats, centroids)
                    mask = padder(mask, patch_size=patch_size)
                    mask_bool = (mask > 0.5).astype(np.uint8)
    
                    if np.sum(mask_bool) > 0:
                        # Flatten masks for metrics
                        pred_flat = prediction_bool.flatten()
                        mask_flat = mask_bool.flatten()
        
                        # Calculate metrics
                        f1_val = f1_metric(mask_flat, pred_flat)
                        iou_val = jaccard_score(mask_flat, pred_flat)
                        precision_val = precision_score(mask_flat, pred_flat)
                        recall_val = recall_score(mask_flat, pred_flat)
        
                        # Store metrics
                        all_f1.append(f1_val)
                        all_iou.append(iou_val)
                        all_precision.append(precision_val)
                        all_recall.append(recall_val)
        
                        if view_masks:
                            # Plot side-by-side: original image, predicted mask, actual mask
                            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                            axs[0].imshow(image_np)
                            axs[0].set_title("Original Image")
                            axs[0].axis("off")
                
                            axs[1].imshow(prediction_bool, cmap='gray')
                            axs[1].set_title("Predicted Mask")
                            axs[1].axis("off")
                
                            axs[2].imshow(mask_bool, cmap='gray')
                            axs[2].set_title("Actual Mask")
                            axs[2].axis("off")
                
                            plt.tight_layout()
                            plt.show()
                    else:
                        print(f"Skipping frame {file} as ground truth has no positive pixels.")
                else:
                    print(f"Mask for {file} not found, skipping metrics...")
                
                # Segment into 5 plants and measure roots
                segments = segment_image(prediction_bool)
                
                for plant_idx, segment in enumerate(segments):
                    try:
                        if np.sum(segment) > 0:  # Check if there's any root detected
                            root_length = measuring_skeleton(segment)
                            plant_root_lengths[plant_idx].append(root_length)
                        else:
                            plant_root_lengths[plant_idx].append(0)
                    except Exception as e:
                        print(f"Error measuring plant {plant_idx} in {file}: {e}")
                        plant_root_lengths[plant_idx].append(0)
    
        # Print average metrics
        if all_f1:
            print("=== Average Metrics ===")
            print(f"Avg F1 Score: {np.mean(all_f1):.4f}")
            print(f"Avg mIoU: {np.mean(all_iou):.4f}")
            print(f"Avg Precision: {np.mean(all_precision):.4f}")
            print(f"Avg Recall: {np.mean(all_recall):.4f}")
        else:
            print("No valid images/masks were processed for metrics.")
        
        # Create separate line plots for each plant's root growth
        if view_growth:
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            for plant_idx in range(5):
                # Add 0 at the start if list has 14 measurements (missing day 1)
                root_data = plant_root_lengths[plant_idx]
                if len(root_data) == 14:
                    root_data = [0] + root_data
                
                x_values = range(1, len(root_data) + 1)
                axes[plant_idx].plot(x_values, 
                                    root_data, 
                                    marker='o', 
                                    color=f'C{plant_idx}')
                axes[plant_idx].set_xlabel('Day')
                axes[plant_idx].set_ylabel('Primary Root Length (pixels)')
                axes[plant_idx].set_title(f'Plant {plant_idx + 1}')
                axes[plant_idx].set_ylim(0, 1300)
                axes[plant_idx].set_xlim(1, 15)
                axes[plant_idx].set_xticks([1, 3, 6, 9, 12, 15])
                axes[plant_idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        return plant_root_lengths