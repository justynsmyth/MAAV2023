import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import pdb

image = cv2.imread('test.png', cv2.IMREAD_UNCHANGED)


img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
th = 100 # 100-128
ret,thresh_img = cv2.threshold(img_grey, th, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = np.zeros(image.shape)
cv2.drawContours(img_contours, contours, -1, (255,255,255), 3)
cv2.imwrite('contoured.png', img_contours)




# mask = cv2.inRange(hsv_image, lower_red, upper_red)

# Shape Detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)
#contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#pdb.set_trace()
for i, contour in enumerate(contours):
    if i == 0:
        continue

    epsilon = 0.01*cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 3: 
        is_curved = False
        for cont in approx:
            if (cv2.isContourConvex(cont)):
                is_curved = True
                break
        if (is_curved):
            print("Quarter-Circle")
        else:
            print("Triangle")
    elif len(approx) == 2: 
        print("Semicircle")
    elif len(approx) == 4: 
        print("Rectangle")
    elif len(approx) == 5: 
        print("Pentagon")
    elif len(approx) == 6: 
        print("Hexagon")
    elif len(approx) == 10:
        print("Star")
    elif len(approx) == 12:
        print("Cross")
    else: 
        print("Circle")

# # Character (Letter or Symbol) Detection via Tesseract for OCR
# central_region = image #[height:y+height, x:x+width]  
# character = pytesseract.image_to_string(central_region)

# # Display or use the results as needed
# cv2.imshow('Color Mask', mask)

# print('Detected Character:', character)

# Color Detection
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

color_ranges = {
    "Red": [(0, 100, 100), (10, 255, 255)],
    "Green": [(110, 100, 100), (130, 255, 255)],
    "Blue": [(90, 100, 100), (120, 255, 255)],
    "Purple": [(260, 100, 100), (280, 255, 255)],
    "White": [(0, 0, 200), (180, 30, 255)],
    "Black": [(0, 0, 0), (180, 255, 30)],
    "Orange": [(10, 100, 100), (25, 255, 255)],
    "Brown": [(20, 100, 70), (40, 255, 153)]
}

# Dict to hold masks for each potential color
color_masks = {}

# Adds mask to color_masks for each color in color_ranges
for color_name, (lower, upper) in color_ranges.items():
    lower_bound = np.array(lower, dtype=np.uint8)
    upper_bound = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    color_masks[color_name] = mask