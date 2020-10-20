from __future__ import print_function, absolute_import

import os
import torch
import shutil
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, extract_archive, verify_str_arg



ARCHIVE_DICT = {
    'data': {
        'url': 'http://cs231n.stanford.edu/tiny-imagenet-200.zip',
        'md5': '90528d7ca1a48142e341f4ef8d21d0de',
    },
}


class TinyImageNet(ImageFolder):
    """
    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    def __init__(self, root, split='train', download=False, **kwargs):
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))
        self.base_folder = 'tiny-imagenet-200'

        if download:
            self.download()
        wnid_to_classes = self._load_meta_file()[0]

        super(TinyImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

    def download(self):
        if not (check_integrity(self.meta_file) and os.path.isdir(self.split_folder)):
            archive_dict = ARCHIVE_DICT['data']
            download_and_extract_archive(archive_dict['url'], self.root,
                                         extract_root=self.root,
                                         md5=archive_dict['md5'])
            base_folder = _splitexts(os.path.basename(archive_dict['url']))[0]
            meta = parse_devkit(os.path.join(self.root, base_folder))
            self._save_meta_file(*meta)

            val_id_to_wnids = self._load_meta_file()[1]
            prepare_val_folder(os.path.join(self.root, base_folder, 'val'), val_id_to_wnids)
        else:
            print('Files already downloaded and verified')

    @property
    def meta_file(self):
        return os.path.join(self.root, self.base_folder, 'meta.bin')

    def _load_meta_file(self):
        if check_integrity(self.meta_file):
            return torch.load(self.meta_file)
        else:
            raise RuntimeError("Meta file not found or corrupted.",
                               "You can use download=True to create it.")

    def _save_meta_file(self, wnid_to_class, val_wnids):
        torch.save((wnid_to_class, val_wnids), self.meta_file)

    @property
    def split_folder(self):
        return os.path.join(self.root, self.base_folder, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


def parse_devkit(root):
    wnid_to_classes = parse_meta(root)
    val_id_to_wnids = parse_val_groundtruth(os.path.join(root, 'val'))
    return wnid_to_classes, val_id_to_wnids


def parse_meta(root, filename='words.txt'):
    with open(os.path.join(root, filename), 'r') as txtfh:
        meta_lines = txtfh.readlines()
    wnid_to_classes = {line.split('\t')[0]: tuple(line.split('\t')[1].split(', ')) for line in meta_lines}
    return wnid_to_classes


def parse_val_groundtruth(root, filename='val_annotations.txt'):
    with open(os.path.join(root, filename), 'r') as txtfh:
        val_lines = txtfh.readlines()
    val_id_to_wnids = {line.split('\t')[0]: line.split('\t')[1] for line in val_lines}
    return val_id_to_wnids


def prepare_train_folder(folder):
    for archive in [os.path.join(folder, archive) for archive in os.listdir(folder)]:
        extract_archive(archive, os.path.splitext(archive)[0], remove_finished=True)


def prepare_val_folder(root, val_id_to_wnids):
    for wnid in set(val_id_to_wnids.values()):
        os.mkdir(os.path.join(root, wnid))

    for img_file, wnid in val_id_to_wnids.items():
        shutil.move(os.path.join(root, 'images', img_file), os.path.join(root, wnid, img_file))
    os.rmdir(os.path.join(root, 'images'))


def _splitexts(root):
    exts = []
    ext = '.'
    while ext:
        root, ext = os.path.splitext(root)
        exts.append(ext)
    return root, ''.join(reversed(exts))
