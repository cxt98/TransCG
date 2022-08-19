#!/usr/bin/env python3
import os
import cv2
import os.path
import numpy as np
from PIL import Image
import scipy.io as scio
import glob
import random
import yaml
import torchvision.transforms as TF
from torch.utils.data import Dataset
import torch
from utils.data_preparation import process_data

class ClearposeDataset(Dataset):
    """
    Template for loading clearpose data, define specific dataset as inheritance
    """
    def __init__(self, dataset_name, args, ratio=1.0, root="./TransNet/data/clearpose", transforms=None):
        assert dataset_name in ['train', 'val', 'test']
        self.dataset_name = dataset_name
        self.args = args
        self.transforms = transforms
        self.resolution = [640, 480]
        self.cam_intrinsic = np.array([[601.3, 0.    , 334.7],
                                       [0.   , 601.3 , 248.0],
                                       [0.   , 0.    , 1.0]])
        self.root = root
        if self.dataset_name == 'train':
            self.add_noise = True
            data_list = {
                # "set2": [1, 3, 4, 5, 6],
                "set4": [1, 2, 3, 4],
                "set5": [1, 2, 3, 4],
                "set6": [1, 2, 3, 4],
                "set7": [1, 2, 3, 4],
            }           
        elif self.dataset_name == 'val':
            data_list = {
                "set3": [4, 8, 11],
                "set4": [5],
                "set5": [5],
                "set6": [5],
                "set7": [5],
            }            
            self.add_noise = False
        elif self.dataset_name == 'test':
            data_list = {
                "set3": [1, 3],
                "set4": [6],
                "set5": [6],
                "set6": [6],
                "set7": [6],
                "set8": [1, 2] # 3, 4, 5, 6 also available
                # "set9": [7, 8, 9, 10, 11, 12] also available
            }
            self.add_noise = False
        self.trancolor = TF.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.target_cate = {
            "bottle": 1,
            "bowl": 2,
            "container": 3,
            "tableware": 4,
            "water_cup": 5,
            "wine_cup": 6            
        }
        self.target_cate_inv = {item:key for (key, item) in self.target_cate.items()}
        ### cate that train by maskrcnn but don't want to use for pose estimation
        if self.args:
            self.ignore_cats = self.args.ignore_cats
        # self.ignore_cats_id = [self.target_cate[cate] for cate in self.ignore_cats]
        self.clearpose_obj_dict = {
            "beaker_1": 1,
            "dropper_1": 2,
            "dropper_2": 3,
            "flask_1": 4,
            "funnel_1": 5,
            "graduated_cylinder_1": 6,
            "graduated_cylinder_2": 7,
            "pan_1": 8,
            "pan_2": 9,
            "pan_3": 10,
            "reagent_bottle_1": 11,
            "reagent_bottle_2": 12,
            "stick_1": 13,
            "syringe_1": 14,
            "bottle_1": 15,
            "bottle_2": 16,
            "bottle_3": 17,
            "bottle_4": 18,
            "bottle_5": 19,
            "bowl_1": 20,
            "bowl_2": 21,
            "bowl_3": 22,
            "bowl_4": 23,
            "bowl_5": 24,
            "bowl_6": 25,
            "container_1": 26,
            "container_2": 27,
            "container_3": 28,
            "container_4": 29,
            "container_5": 30,
            "fork_1": 31,
            "knife_1": 32,
            "knife_2": 33,
            "mug_1": 34,
            "mug_2": 35,
            "pitcher_1": 36,
            "plate_1": 37,
            "plate_2": 38,
            "spoon_1": 39,
            "spoon_2": 40,
            "water_cup_1": 41,
            "water_cup_2": 42,
            "water_cup_3": 43,
            "water_cup_4": 44,
            "water_cup_5": 45,
            "water_cup_6": 46,
            "water_cup_7": 47,
            "water_cup_8": 48,
            "water_cup_9": 49,
            "water_cup_10": 50,
            "water_cup_11": 51,
            "water_cup_12": 52,
            "water_cup_13": 53,
            "water_cup_14": 54,
            "wine_cup_1": 55,
            "wine_cup_2": 56,
            "wine_cup_3": 57,
            "wine_cup_4": 58,
            "wine_cup_5": 59,
            "wine_cup_6": 60,
            "wine_cup_7": 61,
            "wine_cup_8": 62,
            "wine_cup_9": 63
            }
        self.clearpose_obj_dict_inv = {v:k for (k, v) in self.clearpose_obj_dict.items()}
        ## self.obj_config store 'category', 'center', 'keypoint', 'type'
        self.obj_config = self.parsemodel(self.root)
        self.all_lst = self.dataset_list(self.root, data_list, ratio)
        self.meta_data = self.loadmeta(self.root, data_list)
        self.obj_cate = {}
        for obj in self.obj_config:
            if self.obj_config[obj]['category'] in self.target_cate:
                self.obj_cate[obj] = self.target_cate[self.obj_config[obj]['category']]
        self.obj_cate = {k: v for k, v in sorted(self.obj_cate.items(), key=lambda item: item[1])}
        print("{}_dataset_size: ".format(self.dataset_name), len(self.all_lst))
        
    @staticmethod
    def dataset_list(root_path, dataset_list, ratio, shuffle=True):
        # datalst = [('set7', 'scene6', '000040')] # specify a frame if just want its result
        # return datalst
        datalst = []
        for set_idx in dataset_list:
            for scene_idx in dataset_list[set_idx]:
                file_lst = glob.glob(os.path.join(root_path, set_idx, f"scene{scene_idx}", "*-color.png"))
                datalst += [(set_idx, f"scene{scene_idx}", os.path.basename(f).split("-color.png")[0]) for f in file_lst]
        if shuffle:
            random.shuffle(datalst)
        return datalst[:int(len(datalst)*ratio)]

    @staticmethod
    def loadmeta(root_path, dataset_list):
        metalist = {}
        for set_idx in dataset_list:
            metalist[set_idx] = {}
            for scene_idx in dataset_list[set_idx]:
                metalist[set_idx][f"scene{scene_idx}"] = scio.loadmat(os.path.join(root_path, set_idx, f"scene{scene_idx}", "metadata.mat"))

        return metalist

    def parsemodel(self, root_path):
        model = {}
        models_pkg = os.listdir(os.path.join(root_path, "model"))
        for model_pkg in models_pkg:
            if model_pkg in self.clearpose_obj_dict:
                f = open(os.path.join(root_path, "model", model_pkg, f"{model_pkg}_description.txt"))
                model[self.clearpose_obj_dict[model_pkg]] = yaml.safe_load(f)
                model[self.clearpose_obj_dict[model_pkg]]['name'] = model_pkg
                model[self.clearpose_obj_dict[model_pkg]]['path'] = os.path.join(root_path, "model", model_pkg, f"{model_pkg}.obj")
        return model
    
    def setcate(self, cate):
        assert cate in self.target_cate
        self.ignore_cats = list(self.target_cate.keys()).copy()
        self.ignore_cats.remove(cate)

    def get_item(self, item_name):
        raise NotImplementedError

    def __len__(self):
        return len(self.all_lst)

    def __getitem__(self, idx):
        item_name = self.all_lst[idx]
        data = self.get_item(item_name)
        return data

    def verify(self, idx):
        pass

class SegmentationDataset(ClearposeDataset):
    def __init__(self, dataset_name, args, ratio=1, root="./TransNet/data/clearpose", transforms=None):
        super().__init__(dataset_name, args, ratio, root)
        self.transforms = transforms

        
    def __getitem__(self, idx):
        item_name = self.all_lst[idx]
        scene_idx, set_idx, data_idx = item_name

        color_path = os.path.join(self.root, scene_idx, set_idx, data_idx+'-color.png')
        mask_path = os.path.join(self.root, scene_idx, set_idx, data_idx+'-label.png')

        color = Image.open(color_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        boxes = []
        obj_ids_ = []
        category_ids = []
        for i in range(len(obj_ids)):
            if self.obj_config[obj_ids[i]]['category']!='none' and self.obj_config[obj_ids[i]]['category'] not in self.ignore_cats:
                category_ids.append(self.target_cate[self.obj_config[obj_ids[i]]['category']])
                obj_ids_.append(obj_ids[i])
        obj_ids = np.array(obj_ids_)
        num_objs = len(obj_ids)
        
        masks = mask == obj_ids[:, None, None]

        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(category_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        min_area = self.args.pixelthreshold
        valid_area = area>min_area
        boxes = boxes[valid_area]
        area = area[valid_area]
        labels = labels[valid_area]
        masks = masks[valid_area]
        iscrowd = iscrowd[valid_area]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            color, target = self.transforms(color, target)

        return color, target


class DepthCompletionDataset(ClearposeDataset):
    def __init__(self, dataset_name, args, ratio=1, root="./Data", **kwargs):
        super().__init__(dataset_name, args, ratio, root)
        self.use_aug = kwargs.get('use_augmentation', True)
        self.rgb_aug_prob = kwargs.get('rgb_augmentation_probability', 0.8)
        self.image_size = kwargs.get('image_size', (640, 480))
        self.depth_min = kwargs.get('depth_min', 0.3)
        self.depth_max = kwargs.get('depth_max', 3.0)
        self.depth_norm = kwargs.get('depth_norm', 1.0)
        self.with_original = kwargs.get('with_original', False)
        
    def __getitem__(self, idx):
        item_name = self.all_lst[idx]
        scene_idx, set_idx, data_idx = item_name
        data_path = os.path.join(self.root, scene_idx, set_idx, data_idx)
        rgb = np.array(Image.open(data_path + '-color.png'), dtype=np.float32)
        depth = cv2.imread(data_path + '-depth.png', -1)
        depth_gt_mask = np.array(Image.open(data_path + '-label.png'), dtype=np.uint8)
        depth_gt = cv2.imread(data_path + '-depth_true.png', -1)
        return process_data(data_path, rgb, depth, depth_gt, depth_gt_mask, self.cam_intrinsic, scene_type = 'cluttered', camera_type = 1, split = self.dataset_name, image_size = self.image_size, depth_min = self.depth_min, depth_max = self.depth_max, depth_norm = self.depth_norm, use_aug = self.use_aug, rgb_aug_prob = self.rgb_aug_prob, with_original = self.with_original)


def testSegmentationDataset():
    import argparse
    parser = argparse.ArgumentParser(description="Arg parser for SegmentationDataset")
    parser.add_argument(
        "-pixelthreshold", type=int, default = 200, help="minimal bbx pixel area for selecting as training sample"
    )
    parser.add_argument(
        "-root", type=str, default="./data/clearpose", help="path to root dataset directory"
    )

    #### For the dataset 
    args = parser.parse_args()
    ds = SegmentationDataset('val', args, ratio=0.1, root=args.root)
    while True:
        index = np.random.randint(0, len(ds))
        ds.__getitem__(index)




if __name__ == "__main__":
    testSegmentationDataset()
