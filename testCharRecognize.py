import tempfile
import os
import shutil
import argparse
import numpy as np
import cv2
from Generator_v2 import imageGenerate
from definitions import shapes, color_options, symbols
from CharRecognize_v2 import charRecognize

'''
ArgParse config
'''
parser = argparse.ArgumentParser(
    description="A test script to test character recognition.")
parser.add_argument("-a", "--all", help="Run script on all combinations.",
                    action="store_true")  # changes args.all to true
parser.add_argument(
    "-t", "--train", help="generate responses and samples data", action="store_true")

args = parser.parse_args()

if not any(vars(args).values()):
    parser.print_help()
    exit(1)

'''
class that initializes and deletes a temporary file after usage
'''


class TempDirectory(object):
    def __init__(self, dir_path):
        self.dir_path = tempfile.mkdtemp(dir=dir_path)

    def __del__(self):
        shutil.rmtree(self.dir_path)


if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.abspath(__file__))

    temp_dir = TempDirectory(dir_path)

    print(temp_dir.dir_path)
    print('--------------------------')

    if args.all:
        ctr = 0
        correct = 0
        for shape in shapes:
            for color in color_options:
                for symbol in symbols:
                    ctr += 1
                    filename = os.path.join(
                        temp_dir.dir_path, f'image{ctr}.png')
                    imageGenerate(
                        filename, 500, 500, color, shape, symbol)
                    value = charRecognize(filename, symbol)
                    if str(value).strip() == str(symbol).strip():
                        correct += 1

        percentage_correct = (correct / ctr) * 100
        print(f"Values Tested: {ctr}")
        print(f"Values Correct: {correct}")
        print(f"Percentage: {percentage_correct}%")

    elif args.test:
        samples = np.loadtxt('generalsamples.data', np.float32)
        responses = np.loadtxt('generalresponses.data', np.float32)
        responses = responses.reshape((responses.size, 1))

        model = cv2.ml.KNearest_create()
        model.train(samples, cv2.ml.ROW_SAMPLE, responses)

        im = cv2.imread('test.png')

        out = np.zeros(im.shape, np.uint8)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        area_threshold_lower = 50
        area_threshold_upper = 1000

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

        cv2.imshow('im', im)
        cv2.imshow('out', out)
        cv2.waitKey(0)
