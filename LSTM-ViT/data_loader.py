import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import albumentations as A

# Define augmentation pipeline
augment = A.Compose([
    A.HorizontalFlip(p=0.7),
    A.VerticalFlip(p=0.7),
    A.Rotate(limit=45, p=0.7),
], additional_targets={'mask': 'mask'})

def build_patch_index(image_dir):
    """
    Organizes files into a dict:
    {
      (experiment, plant, day): [patch_filenames...],
      ...
    }
    """
    index = {}
    for fname in sorted(os.listdir(image_dir)):
        # Parse the filename e.g. '28_05_03_12.png'
        parts = fname.split('_')
        experiment = parts[0]
        plant = parts[1]
        day = parts[2]
        patch = parts[3].split('.')[0]  # remove extension
        
        key = (experiment, plant, day)
        if key not in index:
            index[key] = []
        index[key].append(fname)
    
    # Sort patch filenames by patch number
    for k in index:
        index[k].sort(key=lambda x: int(x.split('_')[3].split('.')[0]))
    return index

def build_sequence_index(image_dir, sequence_length=15):
    """
    Organizes files into sequences for each plant across days:
    {
      (experiment, plant): {
          patch_0: [day1_file, day2_file, ...],
          patch_1: [day1_file, day2_file, ...],
          ...
      }
    }
    """
    patch_index = build_patch_index(image_dir)
    sequence_index = {}
    
    # Group by experiment and plant
    plant_days = {}
    for (experiment, plant, day), patch_files in patch_index.items():
        plant_key = (experiment, plant)
        if plant_key not in plant_days:
            plant_days[plant_key] = {}
        plant_days[plant_key][day] = patch_files
    
    # Create sequences for each plant
    for plant_key, days_data in plant_days.items():
        sequence_index[plant_key] = {}
        
        # Get all days for this plant, sorted
        sorted_days = sorted(days_data.keys())
        
        # For each patch position, create a sequence across days
        if sorted_days:
            # Get the number of patches from the first day
            first_day_patches = days_data[sorted_days[0]]
            num_patches = len(first_day_patches)
            
            for patch_idx in range(num_patches):
                patch_sequence = []
                for day in sorted_days:
                    if day in days_data and patch_idx < len(days_data[day]):
                        patch_sequence.append(days_data[day][patch_idx])
                
                # Only include sequences that have at least sequence_length patches
                if len(patch_sequence) >= sequence_length:
                    sequence_index[plant_key][f'patch_{patch_idx}'] = patch_sequence
    
    return sequence_index

def check_sequence_content(sequence_files, mask_dir, min_positive_frames=1):
    """
    Check if a sequence has enough frames with positive mask pixels.
    
    Args:
        sequence_files: List of image filenames in the sequence
        mask_dir: Directory containing mask files
        min_positive_frames: Minimum number of frames that need positive pixels
    
    Returns:
        tuple: (is_valid, positive_frame_count, total_positive_pixels)
    """
    positive_frames = 0
    total_positive_pixels = 0
    
    for patch_file in sequence_files:
        mask_file = patch_file.replace('.png', '.tif')
        mask_path = os.path.join(mask_dir, mask_file)
        
        try:
            # Quick check if mask has any positive pixels
            mask = img_to_array(load_img(mask_path, color_mode='grayscale'))
            positive_pixels = np.sum(mask > 0)
            
            if positive_pixels > 0:
                positive_frames += 1
                total_positive_pixels += positive_pixels
        except Exception as e:
            print(f"Warning: Could not load mask {mask_path}: {e}")
            continue
    
    is_valid = positive_frames >= min_positive_frames
    return is_valid, positive_frames, total_positive_pixels

def data_generator_sequences(image_dir, mask_dir, batch_size, sequence_length=15, 
                           patch_size=256, img_channels=3, is_training=True,
                           reverse_time=True, min_positive_frames=1):
    """
    Generator that yields sequences of patches for each plant.
    
    Args:
        image_dir: Directory containing image files
        mask_dir: Directory containing mask files
        batch_size: Number of sequences per batch
        sequence_length: Length of each sequence (default 15)
        patch_size: Size of each patch (default 256)
        img_channels: Number of image channels (default 3)
        is_training: Whether to apply augmentation and shuffling
        reverse_time: Whether to reverse the temporal order (plant shrinking)
        min_positive_frames: Minimum frames with positive pixels to include sequence
    
    Returns:
        X: (batch_size, sequence_length, height, width, channels)
        y: (batch_size, sequence_length, height, width, 1)
    """
    sequence_index = build_sequence_index(image_dir, sequence_length)
    
    # Create list of all valid sequences
    all_sequences = []
    skipped_empty = 0
    skipped_short = 0
    
    print("Building sequence list and filtering empty sequences...")
    
    for plant_key, patches in sequence_index.items():
        for patch_name, patch_sequence in patches.items():
            # Create sliding windows of sequence_length
            for i in range(len(patch_sequence) - sequence_length + 1):
                sequence_slice = patch_sequence[i:i + sequence_length]
                
                # Check if sequence has meaningful content
                is_valid, positive_frames, total_pixels = check_sequence_content(
                    sequence_slice, mask_dir, min_positive_frames
                )
                
                if is_valid:
                    all_sequences.append(sequence_slice)
                else:
                    skipped_empty += 1
    
    print(f"Total valid sequences: {len(all_sequences)}")
    print(f"Skipped empty sequences: {skipped_empty}")
    print(f"Reverse time order: {reverse_time}")
    
    if len(all_sequences) == 0:
        raise ValueError("No valid sequences found! Check your data paths and min_positive_frames setting.")
    
    while True:
        if is_training:
            np.random.shuffle(all_sequences)
        
        for i in range(0, len(all_sequences), batch_size):
            batch_sequences = all_sequences[i:i + batch_size]
            
            if len(batch_sequences) == 0:
                continue
            
            X_batch = []
            y_batch = []
            
            for sequence in batch_sequences:
                X_sequence = []
                y_sequence = []
                
                # Load each patch in the sequence
                for patch_file in sequence:
                    try:
                        # Load image
                        img_patch = img_to_array(load_img(os.path.join(image_dir, patch_file)))
                        
                        # Load mask
                        mask_file = patch_file.replace('.png', '.tif')
                        mask_patch = img_to_array(load_img(
                            os.path.join(mask_dir, mask_file),
                            color_mode='grayscale'
                        ))
                        
                        if is_training:
                            # Apply augmentation (same augmentation for both image and mask)
                            augmented = augment(
                                image=img_patch.astype('uint8'),
                                mask=mask_patch.astype('uint8')
                            )
                            img_patch = augmented['image'] / 255.0
                            mask_patch = augmented['mask']
                        else:
                            # No augmentation for validation
                            img_patch = img_patch / 255.0
                            mask_patch = mask_patch
                        
                        # Ensure mask has channel dimension
                        if len(mask_patch.shape) == 2:
                            mask_patch = np.expand_dims(mask_patch, axis=-1)
                        
                        X_sequence.append(img_patch)
                        y_sequence.append(mask_patch)
                        
                    except Exception as e:
                        print(f"Error loading {patch_file}: {e}")
                        # Add blank frames if loading fails
                        X_sequence.append(np.zeros((patch_size, patch_size, img_channels)))
                        y_sequence.append(np.zeros((patch_size, patch_size, 1)))
                
                # Reverse temporal order if requested (BEFORE adding to batch)
                if reverse_time:
                    X_sequence = X_sequence[::-1]
                    y_sequence = y_sequence[::-1]
                
                X_batch.append(np.array(X_sequence))
                y_batch.append(np.array(y_sequence))
            
            # Convert to numpy arrays
            X_batch = np.array(X_batch, dtype=np.float32)
            y_batch = np.array(y_batch, dtype=np.float32)
            
            yield X_batch, y_batch

def data_generator_for_training(image_dir, mask_dir, batch_size, sequence_length=15, 
                              patch_size=256, img_channels=3, reverse_time=True,
                              min_positive_frames=3):
    """
    Training generator - infinite loop with augmentation and shuffling
    """
    return data_generator_sequences(
        image_dir, mask_dir, batch_size, sequence_length, 
        patch_size, img_channels, is_training=True,
        reverse_time=reverse_time, min_positive_frames=min_positive_frames
    )

def data_generator_for_validation(image_dir, mask_dir, batch_size, sequence_length=15,
                                patch_size=256, img_channels=3, reverse_time=True,
                                min_positive_frames=1):
    """
    Validation generator - no augmentation, may use lower min_positive_frames
    """
    return data_generator_sequences(
        image_dir, mask_dir, batch_size, sequence_length, 
        patch_size, img_channels, is_training=False,
        reverse_time=reverse_time, min_positive_frames=min_positive_frames
    )

def create_generators(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, 
                     batch_size, sequence_length=15, reverse_time=True,
                     train_min_positive_frames=1, val_min_positive_frames=1):
    """
    Create properly configured training and validation generators for sequences
    """
    # Training generator with augmentation
    train_gen = data_generator_for_training(
        train_image_dir, train_mask_dir, batch_size, sequence_length,
        reverse_time=reverse_time, min_positive_frames=train_min_positive_frames
    )
    
    # Validation generator without augmentation
    val_gen = data_generator_for_validation(
        val_image_dir, val_mask_dir, batch_size, sequence_length,
        reverse_time=reverse_time, min_positive_frames=val_min_positive_frames
    )
    
    print(f"\nGenerator Configuration:")
    print(f"Reverse time: {reverse_time}")
    print(f"Min positive frames (train/val): {train_min_positive_frames}/{val_min_positive_frames}")
    
    return train_gen, val_gen

# Simple wrapper for backward compatibility
def data_generator(image_dir, mask_dir, batch_size=16, sequence_length=15,
                  reverse_time=True, min_positive_frames=1):
    """
    Simple data generator for single dataset (e.g., just training or just validation)
    """
    return data_generator_for_training(
        image_dir, mask_dir, batch_size, sequence_length,
        reverse_time=reverse_time, min_positive_frames=min_positive_frames
    )