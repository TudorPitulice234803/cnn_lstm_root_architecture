import numpy as np
import cv2
import os
from tensorflow.keras.utils import Sequence
from collections import defaultdict

class DataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, time_steps, img_size=(256,256), shuffle=True,
                 normalize_images=True, normalize_masks=True, infinite=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.img_size = img_size
        self.shuffle = shuffle
        self.normalize_images = normalize_images
        self.normalize_masks = normalize_masks
        self.infinite = infinite

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
    
            available_days = sorted(day_dict.keys())
            max_day = max(available_days)
    
            for start_day in range(1, max_day - self.time_steps + 2):
                seq = []
                for d in range(start_day, start_day + self.time_steps):
                    if d in day_dict:
                        seq.append(day_dict[d])
                    else:
                        seq.append(None)
                sequences.append(seq)
    
        return sequences

    def __len__(self):
        # If infinite, we pretend it's "infinite" so Keras won't call len().
        return len(self.samples) // self.batch_size if not self.infinite else 1000000  # large placeholder

    def __getitem__(self, index):
        if self.infinite:
            index = index % (len(self.samples) // self.batch_size)
    
        batch_seqs = self.samples[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = []
        y_batch = []
    
        for i, seq in enumerate(batch_seqs):
    
            X_seq = []
            y_seq = []
    
            for meta in seq:
                if meta is None:
                    X_seq.append(np.zeros((*self.img_size, 3), dtype=np.float32))
                    y_seq.append(np.zeros((*self.img_size, 1), dtype=np.float32))
                else:
                    X_seq.append(self.load_image(os.path.join(self.image_dir, meta['filename'])))
                    mask_filename = self.change_extension(meta['filename'], '.tif')
                    mask_path = os.path.join(self.mask_dir, mask_filename)
    
                    if os.path.exists(mask_path):
                        y_mask = self.load_mask(mask_path)
                    else:
                        y_mask = np.zeros((*self.img_size, 1), dtype=np.float32)
    
                    y_seq.append(y_mask)
    
            X_batch.append(X_seq)
            y_batch.append(y_seq)
    
        return np.array(X_batch), np.array(y_batch)


    def change_extension(self, filename, new_ext):
        base = os.path.splitext(filename)[0]
        return base + new_ext
    
    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, self.img_size)
        if self.normalize_images:
            img = img / 255.0
        return img.astype(np.float32)
    
    def load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size)
        if self.normalize_masks:
            mask = mask / 255.0
        return np.expand_dims(mask, axis=-1).astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)
