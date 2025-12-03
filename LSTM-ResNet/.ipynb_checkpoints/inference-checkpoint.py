import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
from tensorflow.keras.models import load_model
from sklearn.metrics import jaccard_score, precision_score, recall_score
from sklearn.metrics import f1_score as f1_metric
import pandas as pd
from datetime import datetime
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
import networkx as nx

from preprocess import crop, padder, crop_to_coordinates
from helpers import f1

def find_longest_path(graph):
    """Finds the longest path in a graph."""
    longest_path = []
    for start_node in graph.nodes():
        for end_node in graph.nodes():
            try:
                path = nx.shortest_path(graph, source=start_node, target=end_node)
                if len(path) > len(longest_path):
                    longest_path = path
            except nx.NetworkXNoPath:
                pass
    return longest_path

def measuring_skeleton(individual_plant):
    """Measures the primary root length from a binary plant mask."""
    plant_contiguous = np.ascontiguousarray(individual_plant)
    root = skeletonize(plant_contiguous)
    root_skeleton = Skeleton(root)
    summary = summarize(root_skeleton)

    G = nx.from_pandas_edgelist(summary, source='node-id-src', target='node-id-dst', edge_attr='branch-distance')
    components = list(nx.connected_components(G))

    longest_path_component = None
    longest_path_length = 0
    for component in components:
        subgraph = G.subgraph(component)
        path = find_longest_path(subgraph)
        if len(path) > longest_path_length:
            longest_path_length = len(path)
            longest_path_component = subgraph

    largest_subgraph = longest_path_component
    
    pos = {}
    for _, row in summary.iterrows():
        if row['node-id-src'] not in pos:
            pos[row['node-id-src']] = (row['image-coord-src-0'], row['image-coord-src-1'])
        if row['node-id-dst'] not in pos:
            pos[row['node-id-dst']] = (row['image-coord-dst-0'], row['image-coord-dst-1'])
    
    start_node = min(largest_subgraph, key=lambda node: pos[node][0])
    end_node = max(largest_subgraph, key=lambda node: pos[node][0])
    path_length = nx.dijkstra_path_length(largest_subgraph, start_node, end_node, weight='branch-distance')
    
    return path_length

def segment_image(prediction, extension_threshold=5, max_extension=100):
    """Segments an image into 5 parts for individual plants."""
    if len(prediction.shape) == 2:
        h, w = prediction.shape
        is_2d = True
    else:
        h, w, c = prediction.shape
        is_2d = False
        
    base_segment_width = w // 5
    boundaries = []
    extended_segments = []
    
    for i in range(5):
        start = i * base_segment_width
        end = (i + 1) * base_segment_width if i < 4 else w
        
        segment = prediction[:, start:end] if is_2d else prediction[:, start:end, :]
        extended_left = 0
        extended_right = 0
        
        if i > 0:
            for ext in range(1, max_extension + 1):
                if start - ext < 0:
                    break
                left_column = prediction[:, start - ext] if is_2d else prediction[:, start - ext, :].max(axis=1)
                if np.count_nonzero(left_column) > extension_threshold:
                    extended_left = ext
                else:
                    break
                    
        if i < 4:
            for ext in range(1, max_extension + 1):
                if end + ext >= w:
                    break
                right_column = prediction[:, end + ext] if is_2d else prediction[:, end + ext, :].max(axis=1)
                if np.count_nonzero(right_column) > extension_threshold:
                    extended_right = ext
                else:
                    break
        
        actual_start = max(0, start - extended_left)
        actual_end = min(w, end + extended_right)
        boundaries.append((actual_start, actual_end))
        
        segment = prediction[:, actual_start:actual_end] if is_2d else prediction[:, actual_start:actual_end, :]
        extended_segments.append(np.array(segment))
    
    return extended_segments

def inference_single_model(model, image_dir, time_steps=15, kernel_size=5, patch_size=256, 
                          view_masks=False, view_growth=True):
    """
    Run inference on a single LSTM model and calculate metrics plus root measurements.
    
    Args:
        model: Loaded Keras model
        image_dir: Directory containing test images
        time_steps: Number of time steps for LSTM
        kernel_size: Kernel size for preprocessing
        patch_size: Patch size for patchifying
        view_masks: Whether to display mask comparisons
        view_growth: Whether to display root growth plots
    
    Returns:
        Dictionary with metrics and root lengths
    """
    
    all_f1 = []
    all_iou = []
    all_precision = []
    all_recall = []
    
    # Store root lengths for each plant across all images
    plant_root_lengths = {i: [] for i in range(5)}  # 5 plants
    
    mask_dir = image_dir
    
    # Get sorted list of image files
    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    num_sequences = len(files) // time_steps
    
    for seq_idx in range(num_sequences):
        seq_files = files[seq_idx * time_steps:(seq_idx + 1) * time_steps]
        frames_padded = []
        stats, centroids = None, None

        # Preprocess each frame in the sequence
        for file in seq_files:
            img_path = os.path.join(image_dir, file)
            image = cv2.imread(img_path, 0)  # Grayscale
            image_cropped, stats, centroids = crop(image, kernel_size=kernel_size)
            image_color = cv2.cvtColor(image_cropped, cv2.COLOR_GRAY2RGB)
            image_np = image_color / 255.0
            image_padded = padder(image_np, patch_size=patch_size)
            frames_padded.append(image_padded)

        frames_padded = np.array(frames_padded)  # (time_steps, H, W, 3)
        H, W, _ = frames_padded[0].shape

        # Patchify all frames
        patches_seq = []
        for t in range(time_steps):
            patches = patchify(frames_padded[t], (patch_size, patch_size, 3), step=patch_size)
            patches_seq.append(patches)

        patches_seq = np.stack(patches_seq, axis=2)
        num_x, num_y = patches_seq.shape[:2]

        # Reshape to (num_patches, time_steps, patch_size, patch_size, 3)
        patches_seq = patches_seq.reshape(-1, time_steps, patch_size, patch_size, 3)

        # Predict masks for all time steps
        predicted_patches = model.predict(patches_seq, verbose=0)

        # Reshape back to (num_x, num_y, time_steps, patch_size, patch_size)
        predicted_patches = predicted_patches.reshape(num_x, num_y, time_steps, patch_size, patch_size)

        # Process each frame
        for t in range(time_steps):
            predicted_mask = unpatchify(predicted_patches[:, :, t, :, :], (H, W))
            prediction_bool = (predicted_mask > 0.5).astype(np.uint8)

            # Load corresponding ground truth mask
            mask_filename = seq_files[t].replace('.png', '.tif')
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, 0)
                mask = crop_to_coordinates(mask, stats, centroids)
                mask = padder(mask, patch_size=patch_size)
                mask_bool = (mask > 0.5).astype(np.uint8)

                if np.sum(mask_bool) > 0:
                    # Compute metrics
                    pred_flat = prediction_bool.flatten()
                    mask_flat = mask_bool.flatten()
                    f1_val = f1_metric(mask_flat, pred_flat)
                    iou_val = jaccard_score(mask_flat, pred_flat)
                    precision_val = precision_score(mask_flat, pred_flat, zero_division=0)
                    recall_val = recall_score(mask_flat, pred_flat, zero_division=0)

                    all_f1.append(f1_val)
                    all_iou.append(iou_val)
                    all_precision.append(precision_val)
                    all_recall.append(recall_val)
                    
                    if view_masks:
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
                    print(f"Skipping frame {seq_files[t]} as ground truth has no positive pixels.")
            else:
                print(f"Mask for {seq_files[t]} not found, skipping metrics...")
            
            # Segment into 5 plants and measure roots for every frame
            segments = segment_image(prediction_bool)
            
            for plant_idx, segment in enumerate(segments):
                try:
                    if np.sum(segment) > 0:
                        root_length = measuring_skeleton(segment)
                        plant_root_lengths[plant_idx].append(root_length)
                    else:
                        plant_root_lengths[plant_idx].append(0)
                except Exception as e:
                    print(f"Error measuring plant {plant_idx} in frame {seq_files[t]}: {e}")
                    plant_root_lengths[plant_idx].append(0)
    
    # Calculate average metrics
    metrics = {
        'f1': np.mean(all_f1) if all_f1 else np.nan,
        'iou': np.mean(all_iou) if all_iou else np.nan,
        'precision': np.mean(all_precision) if all_precision else np.nan,
        'recall': np.mean(all_recall) if all_recall else np.nan,
        'num_frames': len(all_f1)
    }
    
    # Print metrics
    if all_f1:
        print("=== Average Metrics ===")
        print(f"Avg F1 Score: {metrics['f1']:.4f}")
        print(f"Avg mIoU: {metrics['iou']:.4f}")
        print(f"Avg Precision: {metrics['precision']:.4f}")
        print(f"Avg Recall: {metrics['recall']:.4f}")
    else:
        print("No valid images/masks were processed for metrics.")
    
    # Create line plot for root growth
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
    
    return metrics, plant_root_lengths

def pipeline(models_dir, image_dir, time_steps=15, kernel_size=5, patch_size=256, 
            model_extensions=['.h5', '.keras'], save_results=True, 
            view_masks=False, view_growth=True):
    """
    Run inference on all LSTM models in a directory and compile results.
    
    Args:
        models_dir: Directory containing all model files
        image_dir: Directory containing test images
        time_steps: Number of time steps for LSTM models
        kernel_size: Kernel size for preprocessing
        patch_size: Patch size for patchifying
        model_extensions: List of model file extensions to look for
        save_results: Whether to save results to CSV
        view_masks: Whether to display mask comparisons
        view_growth: Whether to display root growth plots
    
    Returns:
        DataFrame with results for all models
    """
    
    # Find all model files
    model_files = []
    for ext in model_extensions:
        model_files.extend([f for f in os.listdir(models_dir) if f.endswith(ext)])
    
    if not model_files:
        print(f"No model files found in {models_dir} with extensions {model_extensions}")
        return None
    
    print(f"Found {len(model_files)} models to evaluate:")
    for mf in model_files:
        print(f"  - {mf}")
    print("\n" + "="*60 + "\n")
    
    # Store results
    results = []
    
    # Process each model
    for idx, model_file in enumerate(model_files, 1):
        model_path = os.path.join(models_dir, model_file)
        model_name = os.path.splitext(model_file)[0]
        
        print(f"Processing model {idx}/{len(model_files)}: {model_name}")
        
        try:
            # Load model
            model = load_model(model_path, custom_objects={"f1": f1})
            
            # Run inference
            metrics, root_lengths = inference_single_model(
                model, 
                image_dir, 
                time_steps=time_steps,
                kernel_size=kernel_size,
                patch_size=patch_size
            )
            
            # Add model info to metrics
            metrics['model_name'] = model_name
            metrics['model_file'] = model_file
            
            # Print results for this model
            print(f"\n  Results for {model_name}:")
            print(f"    F1 Score:  {metrics['f1']:.4f}")
            print(f"    mIoU:      {metrics['iou']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    Frames:    {metrics['num_frames']}")
            
            results.append(metrics)
            
            # Clear model from memory
            del model
            
        except Exception as e:
            print(f"  ERROR processing {model_name}: {str(e)}")
            results.append({
                'model_name': model_name,
                'model_file': model_file,
                'f1': np.nan,
                'iou': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'num_frames': 0,
                'error': str(e)
            })
        
        print("\n" + "="*60 + "\n")
    
    # Create DataFrame with results
    df_results = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = ['model_name', 'f1', 'iou', 'precision', 'recall', 'num_frames', 'model_file']
    if 'error' in df_results.columns:
        column_order.append('error')
    df_results = df_results[column_order]
    
    # Sort by F1 score (descending)
    df_results = df_results.sort_values('f1', ascending=False)
    
    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"lstm_resnet_results_{timestamp}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL MODELS")
    print("="*60)
    print(df_results.to_string(index=False))
    
    # Print best performing model
    if not df_results.empty and not df_results['f1'].isna().all():
        best_model = df_results.iloc[0]
        print("\n" + "="*60)
        print(f"BEST MODEL: {best_model['model_name']}")
        print(f"  F1 Score:  {best_model['f1']:.4f}")
        print(f"  mIoU:      {best_model['iou']:.4f}")
        print("="*60)
    
    return df_results