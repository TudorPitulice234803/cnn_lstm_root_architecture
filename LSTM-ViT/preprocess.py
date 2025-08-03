import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from patchify import patchify
import random
import shutil
import sys
import helpers

def crop(img, kernel_size):
    # This function crops an image to the largest connected component (task 2).
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = helpers.blur_image(img, kernel_size)
    img_thresh = helpers.threshold_image(img_blur)
    num_labels, labels, stats, centroids = helpers.connected_components(img_thresh)
    largest_component = helpers.get_largest_component(stats)
    img_cropped = helpers.crop_image(img, stats, largest_component)
    return img_cropped, stats, centroids

def padder(image, patch_size):
    """
    Adds padding to an image to make its dimensions divisible by a specified patch size.

    This function calculates the amount of padding needed for both the height and width of an image so that its dimensions become divisible by the given patch size. The padding is applied evenly to both sides of each dimension (top and bottom for height, left and right for width). If the padding amount is odd, one extra pixel is added to the bottom or right side. The padding color is set to black (0, 0, 0).

    Parameters:
    - image (numpy.ndarray): The input image as a NumPy array. Expected shape is (height, width, channels).
    - patch_size (int): The patch size to which the image dimensions should be divisible. It's applied to both height and width.

    Returns:
    - numpy.ndarray: The padded image as a NumPy array with the same number of channels as the input. Its dimensions are adjusted to be divisible by the specified patch size.

    Example:
    - padded_image = padder(cv2.imread('example.jpg'), 128)

    """
    h = image.shape[0]
    w = image.shape[1]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w

    top_padding = int(height_padding/2)
    bottom_padding = height_padding - top_padding

    left_padding = int(width_padding/2)
    right_padding = width_padding - left_padding

    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=0)

    return padded_image

def crop_to_coordinates(mask, stats, centroids):
    """
    Crop the mask to a square around the bounding box coordinates of the largest connected component.
    Parameters:
    mask (numpy.ndarray): The binary mask image.
    stats (numpy.ndarray): The statistics of connected components, where each row contains 
                           [x, y, width, height, area] for a component.
    centroids (numpy.ndarray): The centroids of connected components.
    Returns:
    numpy.ndarray: The cropped mask based on a square around the largest connected component.
    """
    # Get the bounding box coordinates of the largest connected component
    largest_component = helpers.get_largest_component(stats)
    x, y, w, h = stats[largest_component][:4]

    # Calculate the size of the square based on the larger dimension
    square_size = max(w, h)

    # Calculate the center of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    # Calculate the top-left corner of the square
    top_left_x = center_x - square_size // 2
    top_left_y = center_y - square_size // 2

    # Crop the mask to the square area, ensuring we stay within the bounds of the mask
    cropped_mask = mask[max(0, top_left_y):min(mask.shape[0], top_left_y + square_size),
                        max(0, top_left_x):min(mask.shape[1], top_left_x + square_size)]

    return cropped_mask
