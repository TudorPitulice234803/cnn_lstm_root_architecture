import numpy as np
import cv2
import os
from tensorflow.keras.utils import Sequence
from collections import defaultdict

class DataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, time_steps, img_size=(256,256), shuffle=True,
                 normalize_images=True, normalize_masks=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.img_size = img_size
        self.shuffle = shuffle
        self.normalize_images = normalize_images
        self.normalize_masks = normalize_masks

        # Store all images metadata grouped by (plant, patch) with day mapping
        self.grouped = self.group_images_by_plant_patch()
        # Build sequences with padding applied
        self.samples = self.build_sequences_with_padding()
        self.on_epoch_end()

    def parse_filename(self, filename):
        name = os.path.splitext(filename)[0]
        parts = name.split('_')
        return {
            'experiment': parts[0],
            'plant': parts[1],
            'day': int(parts[2]),
            'patch': parts[3],
            'filename': filename
        }

    def group_images_by_plant_patch(self):
        grouped = defaultdict(dict)  # { (plant, patch): {day: metadata} }

        for fname in os.listdir(self.image_dir):
            if fname.startswith('.'):
                continue
            meta = self.parse_filename(fname)
            key = (meta['plant'], meta['patch'])
            grouped[key][meta['day']] = meta

        return grouped

    def build_sequences_with_padding(self):
        sequences = []

        for key, day_dict in self.grouped.items():
            plant, patch = key

            # We consider days from 1 to max day (e.g. 15)
            max_day = max(day_dict.keys())
            # or fixed max day 15 for consistency
            max_day = max(max_day, 15)

            # We'll create sequences starting from day 1 to day (max_day - time_steps + 1)
            for start_day in range(1, max_day - self.time_steps + 2):
                seq = []
                for d in range(start_day, start_day + self.time_steps):
                    if d in day_dict:
                        seq.append(day_dict[d])
                    else:
                        seq.append(None)  # missing day = None (will be padded)
                sequences.append(seq)
        return sequences

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, index):
        batch_seqs = self.samples[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = []
        y_batch = []
    
        for seq in batch_seqs:
            X_seq = []
            for meta in seq:
                if meta is None:
                    # Missing day: add black image
                    X_seq.append(np.zeros((*self.img_size, 3), dtype=np.float32))
                else:
                    X_seq.append(self.load_image(os.path.join(self.image_dir, meta['filename'])))
    
            # For mask, pad with black if last day missing
            last_meta = seq[-1]
            if last_meta is None:
                y_mask = np.zeros((*self.img_size, 1), dtype=np.float32)
            else:
                mask_filename = self.change_extension(last_meta['filename'], '.tif')
                y_mask = self.load_mask(os.path.join(self.mask_dir, mask_filename))
    
            X_batch.append(X_seq)
            y_batch.append(y_mask)
    
        return np.array(X_batch), np.array(y_batch)

    def change_extension(self, filename, new_ext):
        base = os.path.splitext(filename)[0]
        return base + new_ext
    
    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.normalize_images:
            img = img / 255.0
        return img.astype(np.float32)
    
    def load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.normalize_masks:
            mask = mask / 255.0
        return np.expand_dims(mask, axis=-1).astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)