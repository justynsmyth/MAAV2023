import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np


def charRecognize(filename, correctChar):
    '''
    Detects for character.

    Returns char detected otherwise -1
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
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

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

    # cv2.imshow('im', im)
    # cv2.imshow('out', out)
    # key = cv2.waitKey(0)

    # if key == 27:  # (escape to quit)
    #     sys.exit()

    return corresponding_character
