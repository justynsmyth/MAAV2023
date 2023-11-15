import easyocr
import cv2

# this needs to run only once to load the model into memory
reader = easyocr.Reader(['en'], gpu=True)


# for (bbox, text, prob) in results:
#     (tl,  tr, br, bl) = bbox
#     tl = (int(tl[0]), int(tl[1]))
#     tr  = (int(tr[0]),  int(tr[1]))
#     br = (int(br[0]), int(br[1]))
#     bl  = (int(bl[0]),  int(bl[1]))

#     text  = "".join([c if ord(c) < 128 else "" for c in text]).strip()

def easyOCR(filename):
    im = cv2.imread(filename)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray, detail=0)

    # Check if there are any results
    if results:
        # Return the first character of the first result
        return results[0][0]
    else:
        # Return None or a default value if no text is detected
        return ''

