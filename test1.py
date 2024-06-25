import cv2
import torch
import numpy as np
from scipy.spatial import distance

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 small model for faster inference

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (640, 640))  # Adjust size based on model input requirements
    return image, resized_image

def detect_passengers(image, confidence_threshold=0.5):
    results = model(image)
    bboxes = results.xyxy[0].cpu().numpy()
    return bboxes[bboxes[:, 4] >= confidence_threshold]  # Filter by confidence score

def calculate_centroids(bboxes):
    centroids = []
    for bbox in bboxes:
        x1, y1, x2, y2, conf, cls = bbox
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        centroids.append((centroid_x, centroid_y))
    return np.array(centroids)

def calculate_distances(centroids):
    return distance.cdist(centroids, centroids, 'euclidean')

def find_nearest_neighbors(distances):
    np.fill_diagonal(distances, np.inf)  # Fill diagonal with infinity to ignore self-distance
    nearest_neighbors = np.argmin(distances, axis=1)
    return nearest_neighbors

def draw_results(image, bboxes, centroids, nearest_neighbors):
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, conf, cls = bbox
        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Draw centroid
        cv2.circle(image, (int(centroids[i][0]), int(centroids[i][1])), 5, (0, 0, 255), -1)
        nearest = nearest_neighbors[i]
        # Draw line to nearest neighbor
        cv2.line(image, (int(centroids[i][0]), int(centroids[i][1])), 
                 (int(centroids[nearest][0]), int(centroids[nearest][1])), 
                 (255, 0, 0), 2)
        # Draw text indicating nearest neighbor
        cv2.putText(image, f'Nearest: {nearest}', (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image

def main(image_path):
    original_image, resized_image = preprocess_image(image_path)
    bboxes = detect_passengers(resized_image)
    centroids = calculate_centroids(bboxes)
    distances = calculate_distances(centroids)
    nearest_neighbors = find_nearest_neighbors(distances)
    result_image = draw_results(original_image, bboxes, centroids, nearest_neighbors)
    cv2.imshow('Nearest Neighbors', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return nearest_neighbors

# Example Usage
image_path = 'vr16.jpg'  # Replace with the path to your image
nearest_neighbors = main(image_path)
print("Nearest neighbors for each passenger:", nearest_neighbors)
