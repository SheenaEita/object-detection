### Illustration ###
# This code is used to detect the object in the image and calculate the centroid and angle of the object.

### Import packages ###
import cv2
import numpy as np
import matplotlib.pyplot as plt

### Set parameters ###
# Otsu method threshold
OTSU_LOW_THRESH = 20
OTSU_HIGH_THRESH = 255
OTSU_SENSITIVITY = 40

# Contour area ratio
min_ratio = 0.005
max_ratio = 0.05

# Load background image and object image
bg_image = cv2.imread(r"yourpath\backgroundfigure.png")
object_image = cv2.imread(r"yourpath\objectfigure.png")

###  Custom functions ###
# Compute image difference using Otsu method
def compute_image_difference(bg, fg, min_thresh, max_thresh, sensitivity):
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)  # make background image gray
    fg_gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)  # make object image gray
    diff_gray = cv2.absdiff(bg_gray, fg_gray)  # compute gray image difference
    diff_gray_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)  # Gaussian blur the difference image
    ret, otsu_thresh = cv2.threshold(diff_gray_blur, min_thresh, max_thresh, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu method threshold
    if ret < sensitivity:
        return np.zeros_like(fg_gray)  # If the threshold is less than the sensitivity, return a zero image of the same size as the foreground image
    else:
        return otsu_thresh  # Otherwise return the Otsu binarization result

# Get valid contours based on area ratio, min_area and max_area are set by the pixel of the image
def get_valid_contours(contours, min_area, max_area):
    valid_contours = []
    for contour in contours:
        if min_area < cv2.contourArea(contour) < max_area:  # If the area of the contour is between the minimum area and the maximum area
            valid_contours.append(contour)  # Add the contour to the list of valid contours
    return valid_contours

# Find contours in the image
def find_contours(bg, fg, min_thresh, max_thresh, sensitivity, min_area, max_area):
    diff = compute_image_difference(bg, fg, min_thresh, max_thresh, sensitivity)  # compute image difference
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    return get_valid_contours(contours, min_area, max_area)  # return valid contours

# Get centroid and angle of the contour
def get_centroid_and_angle(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return (0, 0), 0
    cx = int(M['m10']/M['m00'])  # calculate centroid x coordinate
    cy = int(M['m01']/M['m00'])  # calculate centroid y coordinate
    rect = cv2.minAreaRect(contour)  # get the minimum enclosing rectangle of the contour
    angle = rect[2]  # get the angle of the rectangle
    return (cx, cy), angle # return centroid coordinate and angle

# Get the size of the image (in pixels) to calculate the upper and lower limits of the area of the object detection
def get_image_size(image):
    height, width, _ = image.shape
    return width, height

# Draw rectangles and centroids on the image
def draw_rectangles_and_centroids(image, contours, centroids_angles):
    for contour, (centroid, angle) in zip(contours, centroids_angles):
        # Get the minimum enclosing rectangle of the contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # Convert to integer

        # Draw the rectangle and centroid on the image
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        cv2.circle(image, centroid, 5, (0, 0, 255), -1)

        # Put the angle text
        text = f"Angle: {angle:.1f} deg"
        text_offset_x = centroid[0] + 10
        text_offset_y = centroid[1]
        cv2.putText(image, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Put the centroid text
        centroid_text = f"Centroid: ({centroid[0]}, {centroid[1]})"
        centroid_text_offset_x = centroid[0] + 10
        centroid_text_offset_y = centroid[1] + 20
        cv2.putText(image, centroid_text, (centroid_text_offset_x, centroid_text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return image # return the image with rectangles and centroids


### Main ###
# Get the size of the background image and object image
bg_width, bg_height = get_image_size(bg_image)
object_width, object_height = get_image_size(object_image)

# 輪廓檢測的常數，太大可能是偵測到圖片整體的平移，太小可能是偵測到圖片的雜訊
# Constants for contour detection, too large may be the overall translation of the image, too small may be the noise of the image
MIN_AREA = int(bg_width * bg_height * min_ratio)  # the minimum area of the contour
MAX_AREA = int(bg_width * bg_height * max_ratio)  # the maximum area of the contour

# find contours in the object image
valid_contours = find_contours(bg_image, object_image, OTSU_LOW_THRESH, OTSU_HIGH_THRESH, OTSU_SENSITIVITY, MIN_AREA, MAX_AREA)

# get the centroid and angle of the contour
centroids_angles = [get_centroid_and_angle(contour) for contour in valid_contours]

# draw rectangles and centroids on the image
annotated_image = draw_rectangles_and_centroids(object_image.copy(), valid_contours, centroids_angles) 

# show the image
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig('annotated_image.png')  # save the image
plt.show()