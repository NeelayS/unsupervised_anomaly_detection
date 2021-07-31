import pydicom
import cv2 as cv
import os
import shutil

img_dir_1 = "./stage_2_train_images"
img_dir_2 = "./stage_2_test_images"

os.makedirs("images", exist_ok=True)

n_images = 0
for img_dir in (img_dir_1, img_dir_2):

    for img in os.listdir(img_dir):

        ds = pydicom.dcmread(os.path.join(img_dir, img))
        pixel_array = ds.pixel_array

        img = img.replace(".dcm", ".jpg")
        cv.imwrite((os.path.join("./images", img)), pixel_array)

        n_images += 1

print(f"Successfully extracted {n_images} images")
shutil.rmtree(img_dir_1)
shutil.rmtree(img_dir_2)
