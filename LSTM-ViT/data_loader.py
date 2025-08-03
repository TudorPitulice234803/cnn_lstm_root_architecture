import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import albumentations as A

# ---------------------
# DATA AUGMENTATION
# ---------------------
augment = A.Compose([
    A.HorizontalFlip(p=0.7),
    A.VerticalFlip(p=0.7),
    A.Rotate(limit=45, p=0.7),
], additional_targets={'mask': 'mask'})


# ---------------------
# SEQUENCE INDEX BUILDER
# ---------------------
def build_sequence_index(image_dir):
    index = {}
    for fname in sorted(os.listdir(image_dir)):
        parts = fname.split('_')
        experiment, plant, day = parts[0], parts[1], parts[2]
        key = (experiment, plant)
        
        if key not in index:
            index[key] = {}
        if day not in index[key]:
            index[key][day] = []
        
        index[key][day].append(fname)
    
    for k in index:
        for day in index[k]:
            index[k][day].sort(key=lambda x: int(x.split('_')[3].split('.')[0]))
    
    return index


# ---------------------
# DATA GENERATOR
# ---------------------
def data_generator(image_dir, mask_dir, batch_size=64, patch_size=256):
    index = build_sequence_index(image_dir)
    plant_keys = list(index.keys())

    while True:
        np.random.shuffle(plant_keys)

        for i in range(0, len(plant_keys), batch_size):
            batch_keys = plant_keys[i:i + batch_size]
            X_batch, y_batch = [], []

            for key in batch_keys:
                sequence_images = []
                sequence_masks = []

                days_sorted = sorted(index[key].keys())
                if len(days_sorted) < 15:
                    continue  # skip if not full sequence

                for day in days_sorted[:15]:  # only take first 15 days
                    day_patches = index[key][day]

                    for pf in day_patches:
                        img_patch = img_to_array(load_img(os.path.join(image_dir, pf)))
                        mask_patch = img_to_array(load_img(
                            os.path.join(mask_dir, pf.replace('.png', '.tif')),
                            color_mode='grayscale'
                        ))

                        augmented = augment(
                            image=img_patch.astype('uint8'),
                            mask=mask_patch.astype('uint8')
                        )

                        img_patch = augmented['image'] / 255.0
                        mask_patch = augmented['mask']

                        sequence_images.append(img_patch)
                        sequence_masks.append(mask_patch)

                if len(sequence_images) != 15 * 49:
                    continue  # skip if incomplete sequence

                # reshape to (15, 49, H, W, C)
                sequence_images = np.array(sequence_images).reshape(15, 49, patch_size, patch_size, 3)
                sequence_masks = np.array(sequence_masks).reshape(15, 49, patch_size, patch_size, 1)

                # transpose to (49, 15, H, W, C)
                sequence_images = sequence_images.transpose(1, 0, 2, 3, 4)
                sequence_masks = sequence_masks.transpose(1, 0, 2, 3, 4)

                X_batch.extend(sequence_images)
                y_batch.extend(sequence_masks)

            yield np.array(X_batch), np.array(y_batch)