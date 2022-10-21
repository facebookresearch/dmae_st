
import os
import pickle as pkl
import re
from functools import partial
from multiprocessing.sharedctypes import Value
from pprint import pprint
from typing import Any, Callable, Dict, Optional, Tuple, cast

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import Compose

from data.transforms.uniform_sampling import UniformSampling

MAX_FRAMES = 300
MAX_SKELETONS = 3
MAX_JOINTS = 25
NUM_COORDS = 3

def nturgbd_loader(file: str):
    sample = np.zeros((MAX_FRAMES, MAX_SKELETONS, MAX_JOINTS, NUM_COORDS))
    with open(file, 'r') as f:
        num_frames = int(next(f))
        for frame in range(num_frames):
            num_skeletons = int(next(f))
            for skeleton in range(num_skeletons):
                next(f) # ignore the skeletonID, clippedEdges, ... etc
                
                num_joints = int(next(f))
                assert num_joints == 25 # this is assumed if not, I need to know
                for joint in range(num_joints):
                    sample[frame, skeleton, joint, :] = [float(x) for x in next(f).split(" ")[:3]]
    return torch.tensor(sample.astype(np.float32))
    
def nturgbd_validation(file: str, exclude: str):
    return file.split(".")[-1] == "skeleton" and file.split("/")[-1].split(".")[0] not in exclude

class NTURGBDSkeletons(VisionDataset):
    """ NTURGBDSkeletons Dataset
    
    Inherits from torchvision.datasets.folder.DatasetFolder.
    It overrides `find_classes`, `make_dataset` and sets `extensions`, `transform`, `target_transform` and `is_valid_file`.
    
    
    Args:
        root (string): Root directory path.
        loader (callable, optional): A function to load a NTURGBD text file given its path.
        transform (callable, optional): A function/transform that  takes in an action skeleton sequence
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an NTURGBD text file
            and check if the file is a valid file.
    """

    def __init__(
        self,
        root: str,
        eval_type: Optional[str] = None,
        split: Optional[str] = None,
        loader: Callable[[str], Any] = nturgbd_loader,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):  
        super(NTURGBDSkeletons, self).__init__(root,
                                               transform=transform,
                                               target_transform=target_transform)
        
        example_root = os.path.join(self.root, "examples")
        
        if not is_valid_file:
            with open(os.path.join(root, "exclude.txt"), "r") as exclude_f:
                exclude_list = set(exclude_f.read().splitlines()) 
            is_valid_file = partial(nturgbd_validation, exclude=exclude_list)
            
        assert (eval_type is not None) == (split is not None) # Both or neither are specified
        if eval_type:
            if eval_type == "xsub":
                split_list = open(os.path.join(root, "xsub.txt"), 'r').read().splitlines()
            elif eval_type == "xview":
                split_list = open(os.path.join(root, "xview.txt"), 'r').read().splitlines()
            else:
                raise ValueError(f"NTURGBDSkeletons invalid eval_type: {eval_type}.")
            if split == "train":
                split_list = split_list[0].split(",")
            elif split == "eval":
                split_list = split_list[1].split(",")
            else:
                raise ValueError(f"NTURGBDSkeletons invalid split: {split}.")
        else:
            split_list = None
        
        classes, class_to_idx = self.find_classes(example_root, split_list)
        samples = self.make_dataset(example_root, class_to_idx, is_valid_file, split_list)
        
        self.loader = loader
        
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        
        self.dim = self.loader(self.samples[0][0]).shape[-1]

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        is_valid_file: Callable[[str], bool],
        split_list: Optional[str] = None,
    ):
        directory = os.path.expanduser(directory)
        
        if not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")
        
        is_valid_file = cast(Callable[[str], bool], is_valid_file) 
        
        if split_list:
            regex = re.compile("(.*)({})(.*)".format("|".join(split_list)))
        
        instances = []
        available_classes = set()
        for fname in os.listdir(directory):
            # if not valid, or if not in split -> continue
            if not is_valid_file(fname) or (split_list and not regex.match(fname)):
                continue
            path = os.path.join(directory, fname)
            cls = fname.split(".")[0][-4:]
            instances.append((path, class_to_idx[cls]))
            if cls not in available_classes:
                available_classes.add(cls)
                
        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            raise FileNotFoundError(msg)
        return instances
    
    def find_classes(self, directory: str, split_list: Optional[None] = None):
        if split_list:
            regex = re.compile("(.*)({})(.*)".format("|".join(split_list)))
            file_list = filter(lambda x: not regex.match(x), os.listdir(directory))
        else:
            file_list = os.listdir(directory)
        classes = sorted(list(set(
            map(lambda x: x.split(".")[0][-4:],file_list)
        )))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"x": sample, "y": torch.tensor(target)}

    def __len__(self) -> int:
        return len(self.samples)

if __name__ == "__main__":
    if not os.path.exists("data/nturgb+d_skeletons/stats.pkl"):
        excludes = set(open("data/nturgb+d_skeletons/exclude.txt").read().splitlines())
        path = "data/nturgb+d_skeletons/examples"
        frames_count, skeletons_count, joints_count = [], [], []
        skip_count = 0
        for i, fname in enumerate(os.listdir(path)):
            if fname.split(".")[0] in excludes:
                skip_count += 1
                continue
            fpath = os.path.join(path, fname)
            if i % 1000 == 0:
                print(i, skip_count, fpath)
            with open(fpath, "r") as f:
                frames = int(next(f))
                frames_count.append(frames)
                for frame in range(frames):
                    skeletons = int(next(f))
                    skeletons_count.append(skeletons)
                    for skeleton in range(skeletons):
                        next(f)
                        joints = int(next(f))
                        joints_count.append(joints)
                        for joint in range(joints):
                            next(f)
                        
        print(f"MaxFrames: {max(frames_count)} | MaxSkeletons: {max(skeletons_count)} | MaxJoints: {max(joints_count)}")
        pkl.dump({"frames": frames_count, "skeletons": skeletons_count, "joints": joints_count}, open("data/nturgb+d_skeletons/stats.pkl", 'wb'))
        
    d = pkl.load(open("data/nturgb+d_skeletons/stats.pkl", "rb"))
    print("Frames {} +/- {}".format(np.mean(d['frames']), np.std(d['frames'])))
    print("Skeletons {} +/- {}".format(np.mean(d['skeletons']), np.std(d['skeletons'])))
    print("Joints {} +/- {}".format(np.mean(d['joints']), np.std(d['joints'])))
                
    
    dataset = NTURGBDSkeletons("data/nturgb+d_skeletons")
    print(dataset)
    
    train_loader = DataLoader(dataset,
                              batch_size=16,
                              num_workers=4,
                              shuffle=True)
    dsiter = iter(train_loader)
    for i in range(10):
        print(next(dsiter)["x"].shape)
    
    transforms = Compose([UniformSampling(32)])
    dataset = NTURGBDSkeletons(root="data/nturgb+d_skeletons",
                               transform=transforms)
    train_loader = DataLoader(dataset,
                              batch_size=16,
                              num_workers=4,
                              shuffle=True)
    dsiter = iter(train_loader)
    for i in range(10):
        print(next(dsiter)["x"].shape)
        
    dataset = NTURGBDSkeletons(root="data/nturgb+d_skeletons",
                               eval_type="xsub",
                               split="train")
    print(dataset)
    dataset = NTURGBDSkeletons(root="data/nturgb+d_skeletons",
                               eval_type="xsub",
                               split="eval")
    print(dataset)
    dataset = NTURGBDSkeletons(root="data/nturgb+d_skeletons",
                               eval_type="xview",
                               split="train")
    print(dataset)
    dataset = NTURGBDSkeletons(root="data/nturgb+d_skeletons",
                               eval_type="xview",
                               split="eval")
    print(dataset)

