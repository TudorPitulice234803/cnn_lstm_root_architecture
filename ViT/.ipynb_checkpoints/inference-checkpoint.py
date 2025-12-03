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

def pipeline(image_dir, model_path, kernel_size=5, patch_size=256, view=False):
    if model_path.endswith('h5'):
        model = load_model(model_path, custom_objects={"f1": f1})
    
        all_f1 = []
        all_iou = []
        all_precision = []
        all_recall = []
    
        # Assume masks are in a 'masks' subfolder inside image_dir, with matching filenames
        mask_dir = image_dir
    
        for file in os.listdir(image_dir):
            if file.endswith('.png'):
                # Read and preprocess image
                img_path = os.path.join(image_dir, file)
                image = cv2.imread(img_path, 0)  # grayscale
                image_cropped, stats, centroids = crop(image, kernel_size=kernel_size)
                image_color = cv2.cvtColor(image_cropped, cv2.COLOR_GRAY2RGB)
                image_np = image_color / 255.0
    
                # Pad and patchify
                image_padded = padder(image_np, patch_size=patch_size)
                patches = patchify(image_padded, (patch_size, patch_size, 1), step=patch_size)
                expected_shape = (image_padded.shape[0] // patch_size, image_padded.shape[1] // patch_size)
                patches = patches.reshape(-1, patch_size, patch_size, 1)
    
                # Predict patches and unpatchify
                predicted_patches = model.predict(patches)
                predicted_patches = predicted_patches.reshape(expected_shape[0], expected_shape[1], patch_size, patch_size, predicted_patches.shape[-1])
                predicted_patches = predicted_patches[:, :, :, :, 0]
                prediction = unpatchify(predicted_patches, image_padded.shape[:2])
    
                # Threshold prediction to get binary mask
                prediction_bool = (prediction > 0.5).astype(np.uint8)
    
                # Load the actual mask and preprocess it to match prediction size
                mask_path = os.path.join(mask_dir, file).replace('.png', '.tif')
                if not os.path.exists(mask_path):
                    print(f"Mask for {file} not found, skipping...")
                    continue
                mask = cv2.imread(mask_path, 0)  # grayscale mask
                mask = crop_to_coordinates(mask, stats, centroids)
                mask = padder(mask, patch_size=patch_size)
                mask_bool = (mask > 0.5).astype(np.uint8)
    
                if np.sum(mask_bool) == 0:
                    print(f"Skipping frame as ground truth has no positive pixels.")
                    continue
                
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
    
                # print(f"Metrics for {file}:")
                # print(f"  F1 Score: {f1_val:.4f} | mIoU: {iou_val:.4f} | Precision: {precision_val:.4f} | Recall: {recall_val:.4f}\n")
    
                if view:
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
    
        # After processing all images, print average metrics
        if all_f1:
            print("=== Average Metrics ===")
            print(f"Avg F1 Score: {np.mean(all_f1):.4f}")
            print(f"Avg mIoU: {np.mean(all_iou):.4f}")
            print(f"Avg Precision: {np.mean(all_precision):.4f}")
            print(f"Avg Recall: {np.mean(all_recall):.4f}")
        else:
            print("No valid images/masks were processed.")