from glob import glob
from tqdm import tqdm
from skimage import io
import cv2
import os
import numpy as np


def crop_resize_image(img, size=(224, 224)):
    """
    :param img: image in gray scale, shape: (h, w)
    :return: image in RGB, shape: (h_new, w_new, 3)
    """
    # crop 1
    h, w = img.shape
    if w > h:
        s = (w - h) // 2
        e = s + h
        img_cropped = img[:, s:e]
    elif w < h:
        s = (h - w) // 2
        e = s + w
        img_cropped = img[s:e, :]
    else:
        img_cropped = img

    # crop 2
    h = img_cropped.shape[0]
    s = int(h * 0.005)
    e = h - s
    img_cropped = img_cropped[s:e, s:e]

    # resize
    img_resized = cv2.resize(img_cropped, size)

    # make 3-channel image
    h, w = img_resized.shape
    img3 = np.repeat(img_resized, 3).reshape((h, w, 3))

    return img3


def make_classification_data(size=(224, 224), test_rate=0.1):
    src_root = '.\\source_images'
    train_root = '.\\images\\train\\5classification'
    test_root = '.\\images\\test\\5classification'

    class_imgs = []

    # NSF, non-NSF loop
    scate_dirs = os.listdir(src_root)
    for sdir in scate_dirs:

        # case loop
        case_dirs = os.listdir(os.path.join(src_root, sdir))
        ncase_dirs = len(case_dirs)
        train_max = int((1.0 - test_rate) * ncase_dirs)

        for k, cdir in enumerate(tqdm(case_dirs)):
            is_training = k < train_max

            # list files
            ext = 'png'
            img_path = os.path.join(src_root, sdir, cdir, '*.{}'.format(ext))
            img_files = sorted(glob(img_path))

            if len(img_files) == 0:
                # try again with different extension
                ext = 'jpg'
                img_path = os.path.join(src_root, sdir, cdir, '*.{}'.format(ext))
                img_files = sorted(glob(img_path))

            # read, crop, resize images
            for label, img_file in enumerate(img_files):
                assert img_file.lower().endswith('{}.{}'.format(label, ext.lower())), 'error in {}'.format(cdir)

                img = io.imread(img_file, as_gray=True)
                img = crop_resize_image(img, size=size)

                name = '{}_{}.jpg'.format(sdir, cdir.replace(" ", ""))
                img_data = (img, label, name, is_training)
                class_imgs.append(img_data)

    # save images
    for img_data in tqdm(class_imgs):
        img, label, name, is_training = img_data

        dir_path = os.path.join(train_root if is_training else test_root, 'class{}'.format(label))
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        file_path = os.path.join(dir_path, name)
        io.imsave(file_path, img)


def test():

    # read
    sample_path = os.path.join('.\\source_images', 'non-NSF', 'Case8')
    img_files = sorted(glob(os.path.join(sample_path, '*.{}'.format('png'))))
    if len(img_files) == 0:
        img_files = sorted(glob(os.path.join(sample_path, '*.{}'.format('jpg'))))

    img = io.imread(img_files[1], as_gray=True)
    print(img.shape)

    img = crop_resize_image(img)
    print(img.shape)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # test()

    make_classification_data()

