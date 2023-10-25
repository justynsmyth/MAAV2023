import sys
import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np


def charRecognize(filename, correctChar):
    '''
    Detects for character.

    Returns char detected otherwise -1
    '''
    im = cv2.imread(filename)

    hImg, wImg = im.shape[:2]

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # Set the size of the cropped region
    crop_size = 115

    # Calculate the crop coordinates
    start_x = wImg // 2 - crop_size // 2
    start_y = hImg // 2 - crop_size // 2
    end_x = start_x + crop_size
    end_y = start_y + crop_size

    # Crop the image to the center
    cropped_img = thresh[start_y:end_y, start_x:end_x]
    cropped_img = cv2.resize(cropped_img, (500, 500))
    text = pytesseract.image_to_string(cropped_img, config=(
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 10"))

    # takes image and looks for all contourGroups
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    samples = np.empty((0, 100))
    responses = []
    # keys = [i for i in range(48, 58)] + [i for i in range(65, 91)]

    labels = [str(i) for i in range(10)] + [chr(i)
                                            for i in range(65, 91)]  # 0-9, A-Z
    label_to_int = {label: i for i, label in enumerate(labels)}

    ''' 
    heuristic threshold to filter out small noise contours (streaks in image)
    upper threshold filters out shape contour
    '''
    area_threshold_lower = 50
    area_threshold_upper = 1000

    for cnt in contours:
        if area_threshold_lower < cv2.contourArea(cnt) < area_threshold_upper:
            # draws smallest bounding rectangle that can be drawn around the contour
            [x, y, w, h] = cv2.boundingRect(cnt)

            # heuristic that box must be at least 28 pixels
            if h > 28:
                cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2)

                # crop to region of interest then resize image to 10 x 10
                roi = thresh[y:y+h, x:x+w]
                roismall = cv2.resize(roi, (10, 10))
                cv2.imshow('norm', im)

                responses.append(label_to_int[correctChar])
                sample = roismall.reshape((1, 100))
                samples = np.append(samples, sample, 0)

    responses = np.array(responses, dtype=np.int32)
    responses = responses.reshape((responses.size, 1))
    print("training complete")

    with open('generalsamples.data', 'ab') as f:
        np.savetxt(f, samples)

     # %d informs numpy to treat data as integer
    with open('generalresponses.data', 'ab') as f:
        np.savetxt(f, responses, fmt='%d')

    # np.savetxt('generalsamples.data', samples)
    # np.savetxt('generalresponses.data', responses, fmt='%d')

    if text:
        return text
    else:
        return -1
