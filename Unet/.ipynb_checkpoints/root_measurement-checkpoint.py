import sys
import shutil
import cv2
import numpy as np
import pandas as pd
from patchify import patchify, unpatchify
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from skan import Skeleton, summarize
from skan.csr import skeleton_to_csgraph
import networkx as nx

def segment_image(prediction, extension_threshold=5, max_extension=100):
    """
    Segments an image into 5 parts and extends segments to capture full root systems.
    Works with both 2D and 3D arrays.
    
    Args:
        prediction: numpy array of shape (height, width) or (height, width, channels)
        extension_threshold: minimum number of pixels to trigger extension
        max_extension: maximum number of pixels to extend in either direction
        
    Returns:
        list: List of numpy arrays, each representing a segment of the original image
    """
    # Handle both 2D and 3D arrays
    if len(prediction.shape) == 2:
        h, w = prediction.shape
        is_2d = True
    else:
        h, w, c = prediction.shape
        is_2d = False
        
    base_segment_width = w // 5
    
    # Initialize boundaries
    boundaries = []
    extended_segments = []
    
    # First pass: get initial segments and detect extensions needed
    for i in range(5):
        start = i * base_segment_width
        end = (i + 1) * base_segment_width if i < 4 else w
        
        # Check if roots extend beyond the base segment
        segment = prediction[:, start:end] if is_2d else prediction[:, start:end, :]
        extended_left = 0
        extended_right = 0
        
        if i > 0:  # Check left extension (except for first segment)
            for ext in range(1, max_extension + 1):
                if start - ext < 0:
                    break
                left_column = prediction[:, start - ext] if is_2d else prediction[:, start - ext, :].max(axis=1)
                if np.count_nonzero(left_column) > extension_threshold:
                    extended_left = ext
                else:
                    break
                    
        if i < 4:  # Check right extension (except for last segment)
            for ext in range(1, max_extension + 1):
                if end + ext >= w:
                    break
                right_column = prediction[:, end + ext] if is_2d else prediction[:, end + ext, :].max(axis=1)
                if np.count_nonzero(right_column) > extension_threshold:
                    extended_right = ext
                else:
                    break
        
        # Store the extended boundaries
        actual_start = max(0, start - extended_left)
        actual_end = min(w, end + extended_right)
        boundaries.append((actual_start, actual_end))
        
        # Ensure the segment is a numpy array
        segment = prediction[:, actual_start:actual_end] if is_2d else prediction[:, actual_start:actual_end, :]
        extended_segments.append(np.array(segment))
    
    # # Display the results
    # fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    # for ax, segment in zip(axes, extended_segments):
    #     ax.imshow(segment, cmap='gray' if is_2d else None)
    #     ax.axis('off')
    # plt.tight_layout()
    # plt.show()
    
    # Return the list of numpy arrays
    return extended_segments

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
                pass  # No path between these nodes
    return longest_path

def measuring_skeleton(individual_plant):
    # For some reason the image is rotated clockwise 90 degrees
    plant_contiguous = np.ascontiguousarray(individual_plant)

    root = skeletonize(plant_contiguous)
    root_skeleton = Skeleton(root)
    summary = summarize(root_skeleton)

    # Create a graph from the summary
    G = nx.from_pandas_edgelist(summary, source='node-id-src', target='node-id-dst', edge_attr='branch-distance')
    
    # Get the connected components
    components = list(nx.connected_components(G))

    # Changed logic for largest component.
    longest_path_component = None
    longest_path_length = 0
    for component in components:
        subgraph = G.subgraph(component)
        path = find_longest_path(subgraph)
        if len(path) > longest_path_length:
            longest_path_length = len(path)
            longest_path_component = subgraph

    # Now you have the largest component based on path length
    largest_subgraph = longest_path_component
    
    # Use the actual image coordinates
    pos = {}
    for _, row in summary.iterrows():
        if row['node-id-src'] not in pos:
            pos[row['node-id-src']] = (row['image-coord-src-0'], row['image-coord-src-1'])
        if row['node-id-dst'] not in pos:
            pos[row['node-id-dst']] = (row['image-coord-dst-0'], row['image-coord-dst-1'])
    
    # Find start and end nodes
    start_node = min(largest_subgraph, key=lambda node: pos[node][0])
    end_node = max(largest_subgraph, key=lambda node: pos[node][0])
    
    # Calculate path using Dijkstra's algorithm
    path_length = nx.dijkstra_path_length(largest_subgraph, start_node, end_node, weight='branch-distance')

    # Get the path
    path = nx.dijkstra_path(largest_subgraph, start_node, end_node, weight='branch-distance')
    
    # # Visualization
    # plt.figure(figsize=(10, 10))
    # plt.imshow(root, cmap='gray')
    
    # # Plot all nodes
    # for node in largest_subgraph.nodes():
    #     plt.plot(pos[node][0], pos[node][1], 'b.', markersize=5)

    # path_coords = []
    # for i in range(len(path)-1):
    #     node1, node2 = path[i], path[i+1]
    #     x1, y1 = pos[node1]
    #     x2, y2 = pos[node2]
    #     plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2, alpha=0.5)
    #     path_coords.extend([(x1, y1), (x2, y2)])
    
    # # Plot start and end nodes
    # plt.plot(pos[start_node][0], pos[start_node][1], 'g*', markersize=15, label='Start Node')
    # plt.plot(pos[end_node][0], pos[end_node][1], 'r*', markersize=15, label='End Node')
    
    return path_length
    