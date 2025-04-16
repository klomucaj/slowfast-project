import os
import shutil

src_folder = "datasets/annotations/annotationsGroup"
dst_folder = "datasets/annotations/srt"
os.makedirs(dst_folder, exist_ok=True)

for file in os.listdir(src_folder):
    if file.endswith(".srt"):
        shutil.copy(os.path.join(src_folder, file), os.path.join(dst_folder, file))

print(" Moved all .srt files to annotations/srt/")
