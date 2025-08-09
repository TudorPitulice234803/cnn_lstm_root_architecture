import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import albumentations as A

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
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

def data_generator_for_new_vit(image_dir, mask_dir, batch_size, patch_size=256, img_channels=3):
    """
    Updated data generator for the new ViT model that expects 2D images
    Each batch contains individual 256x256 patches (not sequences of patches)
    """
    index = build_patch_index(image_dir)
    
    # Flatten the index to get individual patches instead of grouped patches
    all_patch_files = []
    for key, patch_files in index.items():
        for pf in patch_files:
            all_patch_files.append(pf)
    
    while True:
        np.random.shuffle(all_patch_files)
        
        for i in range(0, len(all_patch_files), batch_size):
            batch_files = all_patch_files[i:i+batch_size]
            X_batch = []
            y_batch = []
            
            for pf in batch_files:
                # Load image patch - keep as 2D image (256, 256, 3)
                img_patch = img_to_array(load_img(os.path.join(image_dir, pf)))
                
                # Load mask patch - keep as 2D mask (256, 256, 1)
                mask_patch = img_to_array(load_img(
                    os.path.join(mask_dir, pf.replace('.png', '.tif')),
                    color_mode='grayscale'
                ))
                
                # Apply augmentation
                augmented = augment(
                    image=img_patch.astype('uint8'),
                    mask=mask_patch.astype('uint8')
                )
                
                # Normalize image to [0, 1]
                img_patch = augmented['image'] / 255.0
                
                # Keep shapes as (256, 256, 3) and (256, 256, 1)
                X_batch.append(img_patch)
                y_batch.append(mask_patch)
            
            yield np.array(X_batch), np.array(y_batch)

# Alternative: If you want to keep your original grouped approach
def data_generator(image_dir, mask_dir, batch_size, patch_size=256, img_channels=1):
    """
    Alternative generator that processes all patches from the same plant/day together
    But still outputs individual patches for the model
    """
    index = build_patch_index(image_dir)
    keys = list(index.keys())
    
    while True:
        np.random.shuffle(keys)
        
        # Collect patches from multiple groups to fill batches
        X_batch = []
        y_batch = []
        
        for key in keys:
            patch_files = index[key]
            
            for pf in patch_files:
                # Load image and mask as 2D
                img_patch = img_to_array(load_img(
                    os.path.join(image_dir, pf),
                    color_mode='grayscale'
                ))
                mask_patch = img_to_array(load_img(
                    os.path.join(mask_dir, pf.replace('.png', '.tif')),
                    color_mode='grayscale'
                ))
                
                # Apply augmentation
                augmented = augment(
                    image=img_patch.astype('uint8'),
                    mask=mask_patch.astype('uint8')
                )
                
                # Normalize
                img_patch = augmented['image'] / 255.0
                mask_patch = augmented['mask']
                
                X_batch.append(img_patch)
                y_batch.append(mask_patch)
                
                # Yield batch when it's full
                if len(X_batch) == batch_size:
                    yield np.array(X_batch), np.array(y_batch)
                    X_batch = []
                    y_batch = []