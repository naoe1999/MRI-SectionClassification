from glob import glob
from tqdm import tqdm
from skimage import io, color, transform
from matplotlib import pyplot as plt
import os
import numpy as np


class Dataset:

    def __init__(self, imgs, labels=None):
        # force to make 1d-array
        self._imgs = np.asarray(imgs)

        self._labels = None
        if labels is not None:
            assert len(labels) == len(imgs)
            self._labels = np.asarray(labels)

        # for indexing
        self._idxs = np.arange(self.num_samples)

        # set initial cursor just before the first epoch
        self._epoch = 0
        self._cursor = self.num_samples

    @property
    def num_samples(self):
        return len(self._imgs)

    @property
    def cur_epoch(self):
        return self._epoch

    @property
    def is_empty(self):
        return len(self._imgs) == 0

    def get_image(self, i):
        return self._imgs[i]

    def get_label(self, i):
        assert self._labels is not None
        return self._labels[i]

    def _next_epoch(self, shuffle=False):
        if shuffle:
            np.random.shuffle(self._idxs)
        self._cursor = 0
        self._epoch += 1

    def _next_idxs(self, batch_size, shuffle=True):
        if self._cursor >= self.num_samples:
            self._next_epoch(shuffle)
        idxs = self._idxs[self._cursor: self._cursor + batch_size].copy()
        self._cursor += batch_size

        # recursive call
        if len(idxs) < batch_size:
            idxs_next = self._next_idxs(batch_size - len(idxs), shuffle)
            idxs = np.concatenate((idxs, idxs_next), axis=0)
        return idxs

    def next_batch(self, batch_size, shuffle=True):
        """
        Get next batch dataset
        :param batch_size: int
        :param shuffle: boolean
        :return: imgs: np.ndarray, shape: (batch_size, height, width, 3 or 1)
                  gt_box_sets: np.ndarray, shape: (batch_size, nclasses)
        """
        assert self.num_samples > 0
        idxs = self._next_idxs(batch_size, shuffle)
        imgs = self._imgs[idxs]
        labels = self._labels[idxs] if self._labels is not None else None

        return imgs, labels


def read_data(path, limit=0, img_ext='jpg', size=None, grayscale=False):
    classes = sorted(os.listdir(path))
    nclasses = len(classes)

    imgs = []
    labels = []

    for i, cls in enumerate(classes):
        img_file_paths = sorted(glob(os.path.join(path, cls, '*.{}'.format(img_ext))))

        if (type(limit) is int) and (limit != 0) and (limit < len(img_file_paths)):
            img_file_paths = img_file_paths[:limit]

        for img_file in tqdm(img_file_paths):
            img = io.imread(img_file).astype(np.float) / 255.0

            # channel to 3
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[2] > 3:
                img = img[:, :, 0:3]

            # resize
            if size is not None:
                img = transform.resize(img, size)

            # grayscale
            if grayscale:
                img = color.rgb2gray(img)
                h, w = img.shape
                # img = np.expand_dims(img, axis=-1)
                img = np.repeat(img, 3).reshape((h, w, 3))

            imgs.append(img)

            # label = np.zeros((nclasses))
            # label[i] = 1
            label = i
            labels.append(label)

    return imgs, labels, nclasses, classes


#################################################################################
# for test
def test():
    imgs, labels, ncls, cls = read_data('images\\train', limit=100, size=(224, 224), grayscale=True)
    print('num of classes:', ncls)
    print('num of images:', len(imgs))
    print('num of labels:', len(labels))
    print('classes:', cls)

    ds = Dataset(imgs, labels)

    print('num_of_samples', ds.num_samples)

    imgs, labels = ds.next_batch(16)
    print(imgs.shape)
    print(labels.shape)

    print(labels[0])
    io.imshow(imgs[0])
    plt.show()


if __name__ == '__main__':
    test()

