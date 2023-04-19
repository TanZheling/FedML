import os
import os.path

import torch.utils.data as data
from PIL import Image
import copy
import random
import gzip
import logging
import os
import pickle
import urllib

import numpy as np
import torch.utils.data as data
from io import StringIO
from torchvision.io import read_image
from PIL import Image


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [0,1]
    # classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []

    data_local_num_dict = dict()
    net_dataidx_map = dict()
    sum_temp = 0
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        target_num = 0
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    target_num += 1

        net_dataidx_map[class_to_idx[target]] = (sum_temp, sum_temp + target_num)
        data_local_num_dict[class_to_idx[target]] = target_num
        sum_temp += target_num

    assert len(images) == sum_temp
    return images, data_local_num_dict, net_dataidx_map


    


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage  # pylint: disable=E0401

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        pass


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

class CRCK(data.Dataset):
    """USPS Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, use the training split.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    # url = "https://github.com/mingyuliutw/CoGAN/raw/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, ann_file, cls_ind, train=True, slidesample=False, patchsample=False, transform=None, noise_num=0):
        """Init CRCK dataset."""

        self.train = train
        self.ann_file = ann_file
        self.transform = transform
        self.dataset_size = None
        self.slidesample = slidesample
        self.patchsample = patchsample
        self.noise_num = noise_num
        self.cls_ind = cls_ind
        # self.ann_list = self.list_from_file(self.ann_file,root,cls_ind)
        self.data_infos = self.list_from_file(self.ann_file,root,cls_ind,slidesample,patchsample)


    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index not in self.noise_idx:
            # if self.train:
            #     print(index,"nonoise")
            if self.slidesample: # slide-sample
                c = self.class_counter
                s = self.slide_counter[c]
                p = self.patch_counter[c][s]
                cls_idx = self.class_to_slide[c][s]
                img_idx = self.slide_to_id[cls_idx][p]
                result = copy.deepcopy(self.data_infos[img_idx])
                self.patch_counter[c][s] = (p + 1) % len(self.slide_to_id[cls_idx])
                self.slide_counter[c] = (s + 1) % len(self.class_to_slide[c])
                if self.patch_counter[c][s] == 0:
                    random.shuffle(self.slide_to_id[cls_idx])
                if self.slide_counter[c] == 0:
                    slide_idx = list(range(len(self.class_to_slide[c])))
                    random.shuffle(slide_idx)
                    self.patch_counter[c] = [self.patch_counter[c][s] for s in slide_idx]
                    self.class_to_slide[c] = [self.class_to_slide[c][s] for s in slide_idx]
                self.class_counter = (c + 1) % 2
            elif self.patchsample:
                c = self.class_counter
                p = self.patch_counter[c]
                img_idx = self.class_to_id[c][p]
                result = copy.deepcopy(self.data_infos[img_idx])
                self.patch_counter[c] = (p + 1) % len(self.class_to_id[c])
                if self.patch_counter[c] == 0:
                    random.shuffle(self.class_to_id[c])
                self.class_counter = (c + 1) % 2

            else:
                result = copy.deepcopy(self.data_infos[index])
            img_path = result['img_path']
            label = result['label']
            bagname = result['bagname']
            img = Image.open(img_path).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            label = np.int64(label).item()
        else:
            # if self.train:
            #     print(index,"noise")
            result = copy.deepcopy(self.data_infos[index])
            img = (np.random.rand(224, 224, 3)*255).astype(np.uint8)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)
            label = result['label']
            # bagname = result['bagname']
            label = np.int64(label).item()
            bagname = result['bagname']

        if self.train:
            return img, label
        else:
            return img, label, bagname

    def __len__(self):
        """Return size of dataset."""
        return len(self.data_infos)

    def list_from_file(self,ann,pre,cls_ind,slidesample=False,patchsample=False):
        """Load a text file and parse the content as a list of strings."""
        # print(ann)
        item_list = []
        data_infos = []
        self.class_to_slide = [set() for _ in range(2)]
        self.slide_to_id=dict()
        count_img = 0
        # add_noise = self.noise_num
        self.noise_idx = []
        with open(ann,'r') as f:
            for line in f:
                if self.noise_num !=0:
                    if count_img % 100 == 0:
                        for i in range(self.noise_num):
                            info = {'img_path': ''}
                            info['label'] = 0
                            info['bagname'] = 'fake'
                            data_infos.append(info)
                            info = {'img_path': ''}
                            info['label'] = 1
                            info['bagname'] = 'fake'
                            data_infos.append(info)
                        
                # if count_img % batch_size == 0:
                #     img = (np.random.rand(224, 224, 3)*255).astype(np.uint8)
                #     img = Image.fromarray(img, mode='RGB')
                tmp = pre+'/'+line.rstrip('\n\r')
                img_path = tmp.split(' ')[0]
                label = tmp.split(' ')[1].split(',')[cls_ind]
                info = {'img_path': img_path}
                # info['img_info'] = {'filename': filename}
                info['label'] = label
                info['bagname'] = img_path[-27:-15]
                data_infos.append(info)
                self.class_to_slide[int(label)].add(info['bagname'])
                count_img+=1

        for idx, info in enumerate(data_infos):
            bagname = info['bagname']
            if bagname=='fake':
                self.noise_idx.append(idx)
            self.slide_to_id[bagname] = self.slide_to_id.get(bagname,[]) + [idx]
        self.class_to_slide = [list(x) for x in self.class_to_slide]
        self.class_counter = 0
        self.slide_counter = [0 for _ in range(2)]
        if slidesample:
            self.patch_counter = [[0 for _ in self.class_to_slide[c]] for c in range(2)]
        elif patchsample:
            self.patch_counter = [0 for _ in range(2)]
            self.class_to_id = [sum([self.slide_to_id[i] for i in self.class_to_slide[c]], [])
                                    for c in range(2)]
            for c in range(2):
                random.shuffle(self.class_to_id[c])
        # print(self.noise_idx)
        return data_infos

# class CRCK_old(data.Dataset):
#     def __init__(
#         self,
#         data_dir,
#         dataidxs=None,
#         train=True,
#         transform=None,
#         target_transform=None,
#         download=False,
#     ):
#         """
#         Generating this class too many times will be time-consuming.
#         So it will be better calling this once and put it into ImageNet_truncated.
#         """
#         self.dataidxs = dataidxs
#         self.train = train
#         self.transform = transform
#         self.target_transform = target_transform
#         self.download = download
#         self.loader = default_loader
#         # if self.train:
#         #     self.data_dir = os.path.join(data_dir, "train")
#         # else:
#         #     self.data_dir = os.path.join(data_dir, "val")
#         self.data_dir = data_dir
#         (
#             self.all_data,
#             self.data_local_num_dict,
#             self.net_dataidx_map,
#         ) = self.__getdatasets__()
#         if dataidxs == None:
#             self.local_data = self.all_data
#         elif type(dataidxs) == int:
#             (begin, end) = self.net_dataidx_map[dataidxs]
#             self.local_data = self.all_data[begin:end]
#         else:
#             self.local_data = []
#             for idxs in dataidxs:
#                 (begin, end) = self.net_dataidx_map[idxs]
#                 self.local_data += self.all_data[begin:end]

#     def get_local_data(self):
#         return self.local_data

#     def get_net_dataidx_map(self):
#         return self.net_dataidx_map

#     def get_data_local_num_dict(self):
#         return self.data_local_num_dict

#     def __getdatasets__(self):
#         # all_data = datasets.ImageFolder(data_dir, self.transform, self.target_transform)

#         classes, class_to_idx = find_classes(self.data_dir)
#         IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]
#         all_data, data_local_num_dict, net_dataidx_map = make_dataset(
#             self.data_dir, class_to_idx, IMG_EXTENSIONS
#         )
#         if len(all_data) == 0:
#             raise (
#                 RuntimeError(
#                     "Found 0 files in subfolders of: " + self.data_dir + "\n"
#                     "Supported extensions are: " + ",".join(IMG_EXTENSIONS)
#                 )
#             )
#         return all_data, data_local_num_dict, net_dataidx_map

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         # img, target = self.data[index], self.target[index]

#         path, target = self.local_data[index]
#         img = self.loader(path)
#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target

#     def __len__(self):
#         return len(self.local_data)

class CRCK_truncated(data.Dataset):
    def __init__(
        self,
        CRCK_dataset: CRCK,
        dataidxs,
        net_dataidx_map,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):

        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.net_dataidx_map = net_dataidx_map
        self.loader = default_loader
        self.all_data = imagenet_dataset.get_local_data()
        if dataidxs == None:
            self.local_data = self.all_data
        elif type(dataidxs) == int:
            (begin, end) = self.net_dataidx_map[dataidxs]
            self.local_data = self.all_data[begin:end]
        else:
            self.local_data = []
            for idxs in dataidxs:
                (begin, end) = self.net_dataidx_map[idxs]
                self.local_data += self.all_data[begin:end]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], self.target[index]

        path, target = self.local_data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.local_data)

