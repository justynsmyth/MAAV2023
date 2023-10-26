import cv2
import pytesseract
import matplotlib.pyplot as plt


def parseImg(img):
    '''
    Utilize Tesseract-OCR that predicts characters A-Z, 0-9 within shapes


    '''

    hImg, wImg = img.shape[:2]
    print(f"Image dimensions: Height={hImg}, Width={wImg}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # crop image

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
    if text:
        print("Detected text:")
        print(text)
    else:
        print("No text detected.")

    boxes = pytesseract.image_to_boxes(
        cropped_img, config=("-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 10"))

    if boxes:
        for b in boxes.splitlines():
            b = b.split(' ')
            print(b)
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(cropped_img, (x, 500 - y),
                          (w, hImg - h), (0, 0, 0), 1)
            cv2.putText(cropped_img, b[0], (x, 500 - y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            # cv2.imshow("done", cropped_img)
            #         # cv2.waitKey(0)
            #         # cv2.destroyAllWindows()


# Load image
# test 1 no change
image = cv2.imread('test.png')
parseImg(image)


# Display or use the results as needed
# cv2.imshow('Processed IMmage', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
