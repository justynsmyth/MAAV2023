import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np


def trainModel(filename, correctChar):
    '''
    Detects for character.
    '''
    im = cv2.imread(filename)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # takes image and looks for all contourGroups
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    samples = np.empty((0, 100))
    responses = []

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


def testModel(filename):
    samples = np.loadtxt('generalsamples.data', np.float32)
    responses = np.loadtxt('generalresponses.data', np.float32)
    responses = responses.reshape((responses.size, 1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    im = cv2.imread(filename)

    out = np.zeros(im.shape, np.uint8)

    # 1 use HSV to distinguish
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # calculating absolute difference between the HSV components
    # calculates how each pixel's Hue is different from the average Hue of the whole img
    # The same is done with Saturation and Value.

    diff_hue = cv2.absdiff(hsv[:, :, 0], hsv[:, :, 0].mean())
    diff_sat = cv2.absdiff(hsv[:, :, 1], hsv[:, :, 1].mean())
    diff_val = cv2.absdiff(hsv[:, :, 2], hsv[:, :, 2].mean())

    # creating a mask by thresholding differences
    # converting an image to binary form (only black and white)

    thresh_hue = cv2.threshold(
        diff_hue, diff_hue.mean(), 255, cv2.THRESH_BINARY)[1]
    thresh_sat = cv2.threshold(
        diff_sat, diff_sat.mean(), 255, cv2.THRESH_BINARY)[1]
    thresh_val = cv2.threshold(
        diff_val, diff_val.mean(), 255, cv2.THRESH_BINARY)[1]

    # taking bitwise OR of all masks, this will isolate areas
    # which significantly varies from the average color
    mask = cv2.bitwise_or(thresh_hue, thresh_sat)
    mask = cv2.bitwise_or(mask, thresh_val)

    # finally, taking bitwise_and of the mask with the original image
    result = cv2.bitwise_and(im, im, mask=mask)

    # 2 use GrayScale
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # 3 removes small noise in the image that could be detected as edges but don't really represent the shape.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4 apply threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    area_threshold_lower = 50
    area_threshold_upper = 1000

    corresponding_character = ""

    for cnt in contours:
        if area_threshold_lower < cv2.contourArea(cnt) < area_threshold_upper:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 28:
                cv2.rectangle(im, (x, y),
                              (x+w, y+h), (0, 255, 0), 2)
                roi = thresh[y:y+h, x:x+w]
                roismall = cv2.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(
                    roismall, k=1)
                string = str(int((results[0][0])))

                labels = [str(i) for i in range(10)] + [chr(i)
                                                        for i in range(65, 91)]  # 0-9, A-Z
                label_to_int = {label: i for i, label in enumerate(labels)}
                int_to_label = {i: label for label,
                                i in label_to_int.items()}  # Reverse mapping

                if int(string) in int_to_label:
                    corresponding_character = int_to_label[int(string)]
                else:
                    print(
                        f"No corresponding character found for the integer {string}.")
                    exit(1)

                cv2.putText(out, corresponding_character,
                            (x, y+h), 0, 1, (0, 255, 0))

    im = cv2.resize(im, (400, 400))
    hsv = cv2.resize(hsv, (400, 400))
    result = cv2.resize(result, (400, 400))
    gray = cv2.resize(gray, (400, 400))
    thresh = cv2.resize(thresh, (400, 400))
    out = cv2.resize(out, (400, 400))

    # Convert the 'thresh' and 'out' images to 3 channels if necessary
    if len(gray.shape) < 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if len(result.shape) < 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    if len(result.shape) < 3:
        hsv = cv2.cvtColor(hsv, cv2.COLOR_GRAY2BGR)
    if len(thresh.shape) < 3:
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    if len(out.shape) < 3:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    # combined_images = np.concatenate(
    #     (im, hsv, result, gray, thresh, out), axis=1)
    # cv2.imshow('Combined Images', combined_images)
    # key = cv2.waitKey(0)

    # if key == 27:  # (escape to quit)
    #     sys.exit()

    # cv2.destroyAllWindows()

    return corresponding_character
