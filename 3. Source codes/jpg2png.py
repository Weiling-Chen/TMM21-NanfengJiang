import os
import sys

from PIL import Image

input_folder = "trainB/"
output_folder = "gt1/"
# training_data=[]
a = []
for root, dirs, files in os.walk(input_folder):
    for filename in (x for x in files if x.endswith('.jpg')):
        filepath = os.path.join(root, filename)

        object_class = filename.split('.')[0]
        a.append(object_class)
    print(a)

for i in a:
    old_path = input_folder  + str(i) + '.jpg'
    new_path = output_folder  + str(i) + '.png'
    img = Image.open(old_path)
    img.save(new_path)
