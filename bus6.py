import cv2
import numpy as np
# Load the bus image
image = cv2.imread("vr16.jpg")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAYSACLE Image", gray_image)
# Apply a Gaussian blur to the image to smooth it out
blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
cv2.imshow("BLUR Image", blur_image)
# Threshold the image to create a binary image
threshold_image = cv2.threshold(blur_image, 127, 255, cv2.THRESH_BINARY_INV)[1]

cv2.imshow("Thresholded Image", threshold_image)
# Find the contours in the binary image
contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the bounding boxes of the contours
bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

# Find the nearest passenger
nearest_passenger = None
min_distance = float("inf")
for bounding_box in bounding_boxes:
    x, y, w, h = bounding_box
    center_x = x + w / 2
    center_y = y + h / 2
    distance = np.sqrt((0 - center_x)**2 + (0 - center_y)**2)
    if distance < min_distance:
        min_distance = distance
        nearest_passenger = bounding_box

# Print the nearest passenger
if nearest_passenger is not None:
    x, y, w, h = nearest_passenger
    print(f"The nearest passenger is at ({x}, {y}), with dimensions ({w}, {h}).")
else:
    print("There are no passengers in the image.")


# Mark the nearest passenger
if nearest_passenger is not None:
    x, y, w, h = nearest_passenger
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Mark the other passengers
for bounding_box in bounding_boxes:
    if bounding_box != nearest_passenger:
        x, y, w, h = bounding_box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)

