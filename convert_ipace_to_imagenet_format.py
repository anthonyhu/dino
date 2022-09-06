import os
import shutil
from glob import glob

ROOT = '/mnt/local/datasets/ipace'


jpeg_root = os.path.join(ROOT, 'image-cache/jpeg-full_resolution')

for path in sorted(glob(os.path.join(jpeg_root, '*', '*'))):
    print(path)
    for image_path in glob(os.path.join(path, 'cameras/front-forward', '*.jpeg')):
        shutil.move(image_path, path)

    # Remove empty folder 'cameras/front-forward'
    shutil.rmtree(os.path.join(path, 'cameras'))

    #  Move image folder to root
    shutil.move(path, ROOT)

#  Remove empty image-cache folder
shutil.rmtree(os.path.join(ROOT, 'image-cache'))
shutil.rmtree(os.path.join(ROOT, 'teacher-cache'))
