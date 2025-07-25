# Necessary imports
import cv2 
import numpy as np
import keras.backend as K


# Convert the image to grayscale using cv2.cvtColor
def convert_to_gray(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_gray


# Apply a Gaussian blur using cv2.GaussianBlur
def blur_image(image, kernel_size):
    image_blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return image_blur


# Find connected components using cv2.connectedComponents
def connected_components(img):
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    return n_labels, labels, stats, centroids


# Getting the largest component index
def get_largest_component(stats):
    # The area of each component is stored in the last column of the stats
    areas = stats[:, -1]
    # Find the index of the component with the maximum area
    largest_component_index = np.argmax(areas)
    return largest_component_index


# This function creates bounding boxes around the components
def box(stats, component_index, image):
    # Get the coordinates of the bounding box
    x = stats[component_index][0]
    y = stats[component_index][1]
    w = stats[component_index][2]
    h = stats[component_index][3]

    # Drawing the rectangle on the image
    image_box = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 10)

    return image_box


def threshold_image(image):
    # Apply a threshold to the image
    _, image_thresh = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return image_thresh


def crop_image(image, stats, component_index):
    x = stats[component_index][0]  # x-coordinate of the top-left corner
    y = stats[component_index][1]  # y-coordinate of the top-left corner
    w = stats[component_index][2]  # width of the bounding box
    h = stats[component_index][3]  # height of the bounding box

    # Determine the size of the square
    size = max(w, h)

    # Calculate the center of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    # Calculate the new top-left corner to make the bounding box a square
    new_x = max(center_x - size // 2, 0)
    new_y = max(center_y - size // 2, 0)

    # Ensure the new bounding box does not go out of the image boundaries
    new_x = min(new_x, image.shape[1] - size)
    new_y = min(new_y, image.shape[0] - size)

    # Crop the image
    cropped_image = image[new_y:new_y + size, new_x:new_x + size]

    return cropped_image

def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        # Threshold predictions
        y_pred = K.cast(K.greater(y_pred, 0.5), K.dtype(y_true))
        
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives + K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        # Threshold predictions
        y_pred = K.cast(K.greater(y_pred, 0.5), K.dtype(y_true))
        
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))