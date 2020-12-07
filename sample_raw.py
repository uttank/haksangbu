import io
import os

# Imports the Google Cloud client library
from google.cloud import vision

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.abspath('./data2.jpg')
result_file_name = os.path.abspath('./result.json')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

# Performs text detection on the image file
response = client.text_detection(image=image)
texts = response.text_annotations
print('Texts:')

with io.open(result_file_name,'w',encoding='utf-8') as f :
    for text in texts:
        f.write(text)

    if response.error.message:
        raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))