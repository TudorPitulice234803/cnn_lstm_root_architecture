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

# ── Reuse your existing helpers ──────────────────────────────────────────────

def find_longest_path(graph):
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
    plant_contiguous = np.ascontiguousarray(individual_plant)
    root = skeletonize(plant_contiguous)
    root_skeleton = Skeleton(root)
    summary = summarize(root_skeleton)

    G = nx.from_pandas_edgelist(
        summary, source='node-id-src', target='node-id-dst',
        edge_attr='branch-distance'
    )
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
    end_node   = max(largest_subgraph, key=lambda node: pos[node][0])
    path_length = nx.dijkstra_path_length(
        largest_subgraph, start_node, end_node, weight='branch-distance'
    )
    return path_length


def segment_image(prediction, extension_threshold=5, max_extension=100):
    if len(prediction.shape) == 2:
        h, w = prediction.shape
        is_2d = True
    else:
        h, w, c = prediction.shape
        is_2d = False

    base_segment_width = w // 5
    extended_segments = []

    for i in range(5):
        start = i * base_segment_width
        end = (i + 1) * base_segment_width if i < 4 else w

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
        actual_end   = min(w, end + extended_right)

        segment = prediction[:, actual_start:actual_end] if is_2d else prediction[:, actual_start:actual_end, :]
        extended_segments.append(np.array(segment))

    return extended_segments


# ── Measure root lengths from a binary mask ──────────────────────────────────

def measure_roots_from_mask(mask_bool):
    """Segment into 5 plants and return list of root lengths."""
    segments = segment_image(mask_bool)
    lengths = []
    for plant_idx, segment in enumerate(segments):
        try:
            if np.sum(segment) > 0:
                lengths.append(measuring_skeleton(segment))
            else:
                lengths.append(0)
        except Exception as e:
            print(f"  Error measuring plant {plant_idx}: {e}")
            lengths.append(0)
    return lengths


# ── Main inference with ground truth comparison ──────────────────────────────

def inference_with_gt_comparison(model, image_dir, time_steps=15,
                                 kernel_size=5, patch_size=256,
                                 view_masks=False, grayscale=False):
    """
    Run inference and measure root lengths from BOTH predictions and ground
    truth masks. Returns metrics + root lengths for each.

    Args:
        grayscale: If True, keep single channel (for ViT models expecting shape (T,256,256,1)).
    """

    all_f1, all_iou, all_precision, all_recall = [], [], [], []

    pred_root_lengths = {i: [] for i in range(5)}
    gt_root_lengths   = {i: [] for i in range(5)}

    n_channels = 1 if grayscale else 3
    mask_dir = image_dir
    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    num_sequences = len(files) // time_steps

    for seq_idx in range(num_sequences):
        seq_files = files[seq_idx * time_steps:(seq_idx + 1) * time_steps]
        frames_padded = []
        stats, centroids = None, None

        for file in seq_files:
            img_path = os.path.join(image_dir, file)
            image = cv2.imread(img_path, 0)
            image_cropped, stats, centroids = crop(image, kernel_size=kernel_size)

            if grayscale:
                image_np = (image_cropped / 255.0).astype(np.float32)
                image_np = np.expand_dims(image_np, axis=-1)
            else:
                image_color = cv2.cvtColor(image_cropped, cv2.COLOR_GRAY2RGB)
                image_np = image_color / 255.0

            image_padded = padder(image_np, patch_size=patch_size)

            if grayscale and len(image_padded.shape) == 2:
                image_padded = np.expand_dims(image_padded, axis=-1)

            frames_padded.append(image_padded)

        frames_padded = np.array(frames_padded)
        H, W = frames_padded[0].shape[0], frames_padded[0].shape[1]

        patches_seq = []
        for t in range(time_steps):
            patches = patchify(frames_padded[t], (patch_size, patch_size, n_channels), step=patch_size)
            patches_seq.append(patches)

        patches_seq = np.stack(patches_seq, axis=2)
        num_x, num_y = patches_seq.shape[:2]
        patches_seq = patches_seq.reshape(-1, time_steps, patch_size, patch_size, n_channels)

        predicted_patches = model.predict(patches_seq, verbose=0)
        predicted_patches = predicted_patches.reshape(num_x, num_y, time_steps, patch_size, patch_size)

        for t in range(time_steps):
            predicted_mask = unpatchify(predicted_patches[:, :, t, :, :], (H, W))
            prediction_bool = (predicted_mask > 0.5).astype(np.uint8)

            # ── Ground truth ──
            mask_filename = seq_files[t].replace('.png', '.tif')
            mask_path = os.path.join(mask_dir, mask_filename)

            mask_bool = None
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, 0)
                mask = crop_to_coordinates(mask, stats, centroids)
                mask = padder(mask, patch_size=patch_size)
                mask_bool = (mask > 0.5).astype(np.uint8)

                if np.sum(mask_bool) > 0:
                    pred_flat = prediction_bool.flatten()
                    mask_flat = mask_bool.flatten()
                    all_f1.append(f1_metric(mask_flat, pred_flat))
                    all_iou.append(jaccard_score(mask_flat, pred_flat))
                    all_precision.append(precision_score(mask_flat, pred_flat, zero_division=0))
                    all_recall.append(recall_score(mask_flat, pred_flat, zero_division=0))

                    if view_masks:
                        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                        axs[0].imshow(frames_padded[t])
                        axs[0].set_title("Original"); axs[0].axis("off")
                        axs[1].imshow(prediction_bool, cmap='gray')
                        axs[1].set_title("Predicted"); axs[1].axis("off")
                        axs[2].imshow(mask_bool, cmap='gray')
                        axs[2].set_title("Ground Truth"); axs[2].axis("off")
                        plt.tight_layout(); plt.show()

            # ── Measure predicted root lengths ──
            pred_lengths = measure_roots_from_mask(prediction_bool)
            for i, l in enumerate(pred_lengths):
                pred_root_lengths[i].append(l)

            # ── Measure ground truth root lengths ──
            if mask_bool is not None and np.sum(mask_bool) > 0:
                gt_lengths = measure_roots_from_mask(mask_bool)
            else:
                gt_lengths = [0] * 5
            for i, l in enumerate(gt_lengths):
                gt_root_lengths[i].append(l)

    metrics = {
        'f1':        np.mean(all_f1) if all_f1 else np.nan,
        'iou':       np.mean(all_iou) if all_iou else np.nan,
        'precision': np.mean(all_precision) if all_precision else np.nan,
        'recall':    np.mean(all_recall) if all_recall else np.nan,
        'num_frames': len(all_f1)
    }

    return metrics, pred_root_lengths, gt_root_lengths


# ── Static model inference with ground truth comparison ──────────────────────

def inference_static_with_gt_comparison(model, image_dir,
                                         kernel_size=5, patch_size=256,
                                         view_masks=False, grayscale=False):
    """
    Run inference on a static (non-LSTM) model frame-by-frame.
    Each image is fed independently — no temporal batching.
    Still collects root lengths in chronological order so TCS can be computed.

    Args:
        grayscale: If True, keep single channel (for ViT models expecting shape (256,256,1)).
                   If False, convert to RGB (for ResNet/U-Net expecting shape (256,256,3)).
    """

    all_f1, all_iou, all_precision, all_recall = [], [], [], []

    pred_root_lengths = {i: [] for i in range(5)}
    gt_root_lengths   = {i: [] for i in range(5)}

    n_channels = 1 if grayscale else 3
    mask_dir = image_dir
    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    for file in files:
        img_path = os.path.join(image_dir, file)
        image = cv2.imread(img_path, 0)
        image_cropped, stats, centroids = crop(image, kernel_size=kernel_size)

        if grayscale:
            image_np = (image_cropped / 255.0).astype(np.float32)
            image_np = np.expand_dims(image_np, axis=-1)  # (H, W, 1)
        else:
            image_color = cv2.cvtColor(image_cropped, cv2.COLOR_GRAY2RGB)
            image_np = image_color / 255.0

        image_padded = padder(image_np, patch_size=patch_size)

        if grayscale and len(image_padded.shape) == 2:
            image_padded = np.expand_dims(image_padded, axis=-1)

        H, W = image_padded.shape[0], image_padded.shape[1]

        # Patchify single frame
        patches = patchify(image_padded, (patch_size, patch_size, n_channels), step=patch_size)
        num_x, num_y = patches.shape[0], patches.shape[1]
        patches_flat = patches.reshape(-1, patch_size, patch_size, n_channels)

        # Predict
        predicted_patches = model.predict(patches_flat, verbose=0)
        predicted_patches = predicted_patches.reshape(num_x, num_y, patch_size, patch_size)

        # Reconstruct full mask
        predicted_mask = unpatchify(predicted_patches, (H, W))
        prediction_bool = (predicted_mask > 0.5).astype(np.uint8)

        # ── Ground truth ──
        mask_filename = file.replace('.png', '.tif')
        mask_path = os.path.join(mask_dir, mask_filename)

        mask_bool = None
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0)
            mask = crop_to_coordinates(mask, stats, centroids)
            mask = padder(mask, patch_size=patch_size)
            mask_bool = (mask > 0.5).astype(np.uint8)

            if np.sum(mask_bool) > 0:
                pred_flat = prediction_bool.flatten()
                mask_flat = mask_bool.flatten()
                all_f1.append(f1_metric(mask_flat, pred_flat))
                all_iou.append(jaccard_score(mask_flat, pred_flat))
                all_precision.append(precision_score(mask_flat, pred_flat, zero_division=0))
                all_recall.append(recall_score(mask_flat, pred_flat, zero_division=0))

                if view_masks:
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    axs[0].imshow(image_np)
                    axs[0].set_title("Original"); axs[0].axis("off")
                    axs[1].imshow(prediction_bool, cmap='gray')
                    axs[1].set_title("Predicted"); axs[1].axis("off")
                    axs[2].imshow(mask_bool, cmap='gray')
                    axs[2].set_title("Ground Truth"); axs[2].axis("off")
                    plt.tight_layout(); plt.show()

        # ── Measure predicted root lengths ──
        pred_lengths = measure_roots_from_mask(prediction_bool)
        for i, l in enumerate(pred_lengths):
            pred_root_lengths[i].append(l)

        # ── Measure ground truth root lengths ──
        if mask_bool is not None and np.sum(mask_bool) > 0:
            gt_lengths = measure_roots_from_mask(mask_bool)
        else:
            gt_lengths = [0] * 5
        for i, l in enumerate(gt_lengths):
            gt_root_lengths[i].append(l)

    metrics = {
        'f1':        np.mean(all_f1) if all_f1 else np.nan,
        'iou':       np.mean(all_iou) if all_iou else np.nan,
        'precision': np.mean(all_precision) if all_precision else np.nan,
        'recall':    np.mean(all_recall) if all_recall else np.nan,
        'num_frames': len(all_f1)
    }

    return metrics, pred_root_lengths, gt_root_lengths

def temporal_consistency_score(pred_lengths, gt_lengths):
    """
    Cascade-based temporal consistency penalty score.

    For each frame t >= 1, assign a penalty using strict priority
    (most severe checked first):

        4 — Catastrophic drop: prediction falls into the last 10% of
            GT max root length (root effectively disappeared).
        3 — Regression: prediction dropped >3% below its previous value
            while GT was still growing (GT[t] > GT[t-1]).
        2 — Stagnation: prediction stayed within ±3% of its previous
            value while GT grew by more than 3%.
        1 — Acceptable deviation: prediction is >5% off GT but still
            larger than its previous value (growing, just inaccurately).
        0 — Good: prediction is within 5% of GT length.

    The continuous metric per plant is:
        TCS = sum(penalties) / (N - 1)
    where N is the number of frames (first frame has no previous).
    Lower is better. 0 = perfect, 4 = worst possible.

    Returns dict with per-plant scores and an overall average.
    """

    THRESH_GOOD      = 0.05   # within 5% of GT  → score 0
    THRESH_STAGNATION = 0.03  # ±3% of previous   → stagnation check
    THRESH_REGRESSION = 0.03  # >3% below previous → regression check
    THRESH_CATASTROPHIC = 0.10  # last 10% of GT max → catastrophic

    results = {}
    all_scores = []

    for plant_idx in range(5):
        gt   = np.array(gt_lengths[plant_idx],   dtype=float)
        pred = np.array(pred_lengths[plant_idx],  dtype=float)

        n = len(gt)
        if n < 2:
            results[f'plant_{plant_idx+1}'] = {'tcs': 0.0, 'penalties': []}
            all_scores.append(0.0)
            continue

        gt_max = gt.max() if gt.max() > 0 else 1.0
        penalties = []

        for t in range(1, n):
            gt_val      = gt[t]
            pred_val    = pred[t]
            pred_prev   = pred[t - 1]
            gt_prev     = gt[t - 1]

            # Reference for percentage comparisons
            gt_ref = gt_val if gt_val > 0 else 1.0

            # ── Cascade (most severe first) ──

            # 4: Catastrophic drop — prediction near zero
            if pred_val <= THRESH_CATASTROPHIC * gt_max:
                penalties.append(4)
                continue

            # 3: Regression — pred dropped >3% below previous while GT grew
            pred_change_pct = (pred_val - pred_prev) / pred_prev if pred_prev > 0 else 0
            gt_is_growing = gt_val > gt_prev
            if pred_change_pct < -THRESH_REGRESSION and gt_is_growing:
                penalties.append(3)
                continue

            # 2: Stagnation — pred barely moved (±3%) while GT grew >3%
            pred_stagnant = abs(pred_change_pct) <= THRESH_STAGNATION
            gt_growth_pct = (gt_val - gt_prev) / gt_prev if gt_prev > 0 else 0
            if pred_stagnant and gt_growth_pct > THRESH_STAGNATION:
                penalties.append(2)
                continue

            # 1: Growing but inaccurate — >5% off GT but still above previous
            deviation_pct = abs(pred_val - gt_val) / gt_ref
            if deviation_pct > THRESH_GOOD and pred_val > pred_prev:
                penalties.append(1)
                continue

            # 0: Good — within 5% of GT
            penalties.append(0)

        tcs = sum(penalties) / len(penalties) if penalties else 0.0

        results[f'plant_{plant_idx+1}'] = {
            'tcs': tcs,
            'penalties': penalties
        }
        all_scores.append(tcs)

    results['overall_tcs'] = np.mean(all_scores)
    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_gt_vs_pred(pred_root_lengths, gt_root_lengths, model_name="Model",
                    save_path=None):
    """
    Side-by-side line plots: ground truth (dashed) vs prediction (solid)
    for each of the 5 plants.
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    for plant_idx in range(5):
        gt_data   = gt_root_lengths[plant_idx]
        pred_data = pred_root_lengths[plant_idx]

        # Pad to 15 if needed
        for data in [gt_data, pred_data]:
            if len(data) == 14:
                data.insert(0, 0)

        x = range(1, len(gt_data) + 1)

        axes[plant_idx].plot(x, gt_data,   '--o', color='green',
                             label='Ground Truth', markersize=4)
        axes[plant_idx].plot(x, pred_data, '-s',  color='blue',
                             label=model_name, markersize=4)

        # Shade the difference
        axes[plant_idx].fill_between(
            x, gt_data, pred_data, alpha=0.15, color='red'
        )

        axes[plant_idx].set_xlabel('Day')
        axes[plant_idx].set_ylabel('Primary Root Length (px)')
        axes[plant_idx].set_title(f'Plant {plant_idx + 1}')
        axes[plant_idx].set_ylim(0, 1300)
        axes[plant_idx].set_xlim(1, 15)
        axes[plant_idx].set_xticks([1, 3, 6, 9, 12, 15])
        axes[plant_idx].grid(True, alpha=0.3)
        axes[plant_idx].legend(fontsize=7)

    plt.suptitle(f'{model_name} — Predicted vs Ground Truth Root Growth', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


# ── Pipeline ─────────────────────────────────────────────────────────────────

def pipeline(models_dir, image_dir, temporal=True, time_steps=15, kernel_size=5,
                 patch_size=256, save_results=True, view_masks=False, grayscale=False):
    """
    Evaluate all models in a directory: compute segmentation metrics,
    root growth plots (pred vs GT), and temporal consistency scores.

    Args:
        temporal: If True, use LSTM inference (sequences of time_steps).
                  If False, use static inference (frame-by-frame).
        grayscale: If True, feed single-channel images (for ViT models).
    """

    model_files = sorted(
        [f for f in os.listdir(models_dir) if f.endswith(('.h5', '.keras'))]
    )

    if not model_files:
        print(f"No models found in {models_dir}")
        return

    mode = "LSTM (temporal)" if temporal else "Static (frame-by-frame)"
    print(f"Found {len(model_files)} models — Mode: {mode}\n" + "=" * 60)

    all_results = []

    for idx, model_file in enumerate(model_files, 1):
        model_path = os.path.join(models_dir, model_file)
        model_name = os.path.splitext(model_file)[0]
        print(f"\n[{idx}/{len(model_files)}] {model_name}")

        try:
            model = load_model(model_path, custom_objects={"f1": f1})

            if temporal:
                metrics, pred_roots, gt_roots = inference_with_gt_comparison(
                    model, image_dir,
                    time_steps=time_steps,
                    kernel_size=kernel_size,
                    patch_size=patch_size,
                    view_masks=view_masks,
                    grayscale=grayscale
                )
            else:
                metrics, pred_roots, gt_roots = inference_static_with_gt_comparison(
                    model, image_dir,
                    kernel_size=kernel_size,
                    patch_size=patch_size,
                    view_masks=view_masks,
                    grayscale=grayscale
                )

            # Temporal consistency
            tc = temporal_consistency_score(pred_roots, gt_roots)

            # Plot
            plot_gt_vs_pred(
                pred_roots, gt_roots, model_name=model_name,
                save_path=f"{model_name}_growth_comparison.png"
            )

            # Combine into one row
            row = {
                'model_name': model_name,
                **metrics,
                'tcs': tc['overall_tcs'],
            }
            for p in range(5):
                plant_tc = tc[f'plant_{p+1}']
                row[f'plant{p+1}_tcs'] = plant_tc['tcs']
                # Store penalty distribution for this plant
                penalties = plant_tc['penalties']
                for severity in range(5):
                    count = penalties.count(severity)
                    row[f'plant{p+1}_severity_{severity}'] = count

            all_results.append(row)

            print(f"  F1={metrics['f1']:.4f}  IoU={metrics['iou']:.4f}  "
                  f"TCS={tc['overall_tcs']:.3f}")

            del model

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({'model_name': model_name, 'error': str(e)})

    df = pd.DataFrame(all_results)

    if save_results:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"temporal_consistency_results_{ts}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(df[['model_name', 'f1', 'iou', 'precision', 'recall', 'tcs']].to_string(index=False))

    return df
