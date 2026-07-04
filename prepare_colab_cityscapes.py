import urllib.request
import zipfile
import os
import shutil

url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/cityscapes8.zip"
dest_zip = "data/cityscapes8.zip"
extract_dir = "data/cityscapes8_extracted"

os.makedirs("data", exist_ok=True)

print("Downloading Cityscapes sample...")
urllib.request.urlretrieve(url, dest_zip)

print("Extracting...")
with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

src_img_val = "data/cityscapes8_extracted/cityscapes8/images/val"
src_msk_val = "data/cityscapes8_extracted/cityscapes8/masks/val"

dst_img_root = "data/cityscapes/leftImg8bit/train"
dst_msk_root = "data/cityscapes/gtFine/train"

cities = ["ulm", "bremen", "aachen"]

for city in cities:
    os.makedirs(os.path.join(dst_img_root, city), exist_ok=True)
    os.makedirs(os.path.join(dst_msk_root, city), exist_ok=True)

# Copy validation files to aachen and rename them
for filename in os.listdir(src_img_val):
    if filename.endswith(".png"):
        basename = os.path.splitext(filename)[0]
        dst_img_name = f"{basename}_leftImg8bit.png"
        shutil.copy(
            os.path.join(src_img_val, filename),
            os.path.join(dst_img_root, "aachen", dst_img_name)
        )

for filename in os.listdir(src_msk_val):
    if filename.endswith(".png"):
        basename = os.path.splitext(filename)[0]
        dst_msk_name = f"{basename}_gtFine_labelIds.png"
        shutil.copy(
            os.path.join(src_msk_val, filename),
            os.path.join(dst_msk_root, "aachen", dst_msk_name)
        )

# Clean up temp zip
if os.path.exists(dest_zip):
    os.remove(dest_zip)
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)

print("Cityscapes sample structured successfully!")
