import easyocr

# this needs to run only once to load the model into memory
reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext('chinese.jpg', detail=0)
