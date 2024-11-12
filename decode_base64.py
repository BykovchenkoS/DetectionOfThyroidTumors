import base64

with open('base64.txt', 'rb') as bs64_file:
    bs64_data = bs64_file.read()

decoded_img_data = base64.b64decode((bs64_data))

with open('output.jpeg', 'wb') as img_file:
    img_file.write(decoded_img_data)