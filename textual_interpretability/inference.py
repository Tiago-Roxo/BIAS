import os

from transformers import pipeline

models_dir      = "models"
image_captioner = pipeline(task="image-to-text", model=models_dir, device=0)

images_dir  = "imgs"
images_list = os.listdir(images_dir)

for image in images_list:
    full_image_path = os.path.join(images_dir, image)
    image_captions  = image_captioner(full_image_path)

    print("Image: {}     Captions: {}".format(full_image_path, image_captions))

