#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import pickle as pkl
import random
from pprint import pprint
from typing import Any, Dict, Tuple

import dmae.util.decoder.decoder_v2 as decode
import torch
import torchvision.transforms.functional as tvF
from dmae.util.decoder.sampling import spatial_sampling_v2
from mae.util.decoder import utils
from mae.util.decoder.random_erasing import RandomErasing
from mae.util.decoder.transform import create_random_augment
from mae.util.decoder.video_container import get_video_container
from torch.utils.data import Dataset


class KineticsV2(Dataset):
    """
    Kinetics video loader.
    Construct the kinetis video loader then sample clips from the videos.
    For training and validation a single clip is randomly sampled from every video with random cropping, scaling and flipping.
    For testing, multiple clips are sampled uniformly with uniform cropping to over the entire video.
    In uniform cropping, we take the left, center and right crop if width > height else take top, center, and bottom crop.
    
    identically to the original frames. It can also be normalized into a probability distribution.
    """
    
    def __init__(
        self,
        # required arguments
        mode: str,
        path_to_data_dir: str,
        # video sampling / decoding
        sampling_rate: int,
        num_frames: int,
        crop_size: int,
        repeat_aug: int=1,
        target_fps: int=30,
        # test time sampling
        test_num_ensemble_views: int=None,
        test_num_spatial_crops: int=None,
        # augmentation / processing settings
        random_horizontal_flip: bool=None,
        pre_crop_size: Tuple[int]=None,
        jitter_scale_relative: Tuple[int]=None,
        jitter_aspect_relative: Tuple[int]=None,
        rand_aug: bool=None,
        aa_type: str=None,
        rand_erase_prob: float=0.0,
        rand_erase_mode: str=None,
        rand_erase_count: int=None,
        # normalization
        norm_mean: Tuple[int]=(0.45, 0.45, 0.45),
        norm_std: Tuple[int]=(0.225, 0.225, 0.225),
        # other parameters
        enable_multi_thread_decode: bool=False,
        num_retries: int=12,
        backend: str="torchvision",
        log_missing_files: bool=False,
        log_bad_decode: bool=True,
    ):
        """
        Kinetics dataloading.
        Loads the kinetics dataset and applies transformations to sequences of images.

        Args:
            mode (str): data split / loading mode ["pretrain", "finetune", "val", "test"]
            path_to_data_dir (str): Path to kinetics data directory
            sampling_rate (int): number of frames between sampled frames
            num_frames (int): total number of frames sampled
            crop_size (int): output frame H and W
            repeat_aug (int, optional): times to sample spatio-temporally sample from each vid. Defaults to 1.
            target_fps (int, optional): video target fps. Defaults to 30.
            test_num_ensemble_views (int, optional): number of temporal views for multi-view testing. Defaults to None.
            test_num_spatial_crops (int, optional): number of spatial crops for multi-view testing. Defaults to None.
            random_horizontal_flip (bool, optional): enable random horizontal flipping. Defaults to None.
            pre_crop_size (Tuple[int], optional): shortest dimension scaling of video upon loading. Defaults to None.
            jitter_scale_relative (Tuple[int], optional): scale for random_resize_crop. Defaults to None.
            jitter_aspect_relative (Tuple[int], optional): aspect ratio for RRC. Defaults to None.
            rand_aug (bool, optional): enable auto_augment random augmentation. Defaults to None.
            aa_type (str, optional): auto_augment configuration string. Defaults to None.
            rand_erase_prob (float, optional): random erase augmentation. Defaults to 0.0.
            rand_erase_mode (str, optional): random_erase mode. Defaults to None.
            rand_erase_count (int, optional): random_erase count. Defaults to None.
            norm_mean (Tuple[int], optional): frame normalization mean. Defaults to (0.45, 0.45, 0.45).
            norm_std (Tuple[int], optional): frame normalization std. Defaults to (0.225, 0.225, 0.225).
            enable_multi_thread_decode (bool, optional): multi-thread video decoding. Defaults to False.
            num_retries (int, optional): number of attempts to lead AN example. Defaults to 12.
            backend (str, optional): loading backend ("torchvision", "pyav"). Defaults to "torchvision".
            log_missing_files (bool, optional): log error when example video not found. Defaults to False.
            log_bad_decode (bool, optional): log error when example video decode failed. Defaults to True.
        """
        assert mode in [
            "pretrain",
            "finetune",
            "val",
            "test"
        ], (f"Split {mode} not supported for KineticsBbox. "
            f"Please select from (pretrain, finetune, val, test).")
        
        assert backend in [
            "torchvision", 
            "pyav"
        ], (f"Backend {backend} not support for KineticsBbox. "
            "Please select from (torchvision, pyav).")
        
        self.mode = mode
        self.path_to_data_dir = path_to_data_dir
        
        # video sampling / decoding configs
        self.sampling_rate = sampling_rate
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.repeat_aug = repeat_aug
        self.target_fps = target_fps
        
        # additional sampling configs for multi-view @ test-time
        self.test_num_ensemble_views = test_num_ensemble_views
        self.test_num_spatial_crops = test_num_spatial_crops
        
        # augmentation parameters
        self.random_horizontal_flip = random_horizontal_flip
        
        assert ((jitter_scale_relative is not None) == (jitter_aspect_relative is not None), 
                f"jitter_scale_relative is {jitter_scale_relative} while jitter_aspect_relative is {jitter_aspect_relative}")
        
        self.pre_crop_size = pre_crop_size
        self.jitter_scale_relative = jitter_scale_relative
        self.jitter_aspect_relative = jitter_aspect_relative
        
        self.rand_aug = rand_aug
        self.aa_type = aa_type
        self.rand_erase_prob = rand_erase_prob
        self.rand_erase_mode = rand_erase_mode
        self.rand_erase_count = rand_erase_count
        
        # normalization
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        
        # other loading with defaults
        self.enable_multi_thread_decode = enable_multi_thread_decode
        self.num_retries = num_retries
        self.backend = backend
        self.log_missing_files = log_missing_files
        self.log_bad_decode = log_bad_decode
        
        print("="*40)
        print(f"{self.__class__.__name__} attributes.")
        pprint(locals())
        print("="*40)
        
        if backend == "pyav":
            print("WARNING: pyav backend is typically significantly slower than torchvision.")
        
        # set num_clips
        if mode in ["pretrain", "finetune", "val"]:
            self.num_clips = 1
            if test_num_spatial_crops or test_num_ensemble_views:
                print(f"WARNING: spatial_crops {test_num_spatial_crops} or ensemble views{test_num_ensemble_views}"
                      f"set for {mode} mode.")
        elif mode in ["test"]:
            self.num_clips = test_num_ensemble_views * test_num_spatial_crops
        print(f"{self.mode} mode, num_clips={self.num_clips}.")
        
        # check random augmentation settings
        if mode in ["pretrain", "val", "test"]:
            if rand_aug:
                print(f"WARNING: rand_aug={rand_aug} in {mode} mode. Setting to False.")
            if rand_erase_prob and rand_erase_prob > 0.0:
                print(f"WARNING: rand_erase={rand_erase_prob} in {mode} mode. Setting to 0.0.")
            self.rand_aug = False
            self.rand_erase_prob = 0.0
        elif mode in ["finetune"]:
            self.rand_aug = rand_aug
        print(f"{self.mode} mode, rand_aug={self.rand_aug}.")
        
        if rand_erase_prob > 0.0:
            assert rand_erase_mode is not None, f"rand_erase_mode({rand_erase_mode}) while rand_erase_prob({rand_erase_prob})"
            assert rand_erase_count is not None, f"rand_erase_mode({rand_erase_count}) while rand_erase_prob({rand_erase_count})"
        
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0
        
        print(f"Constructing KineticsV2 in {mode} mode...")
        self._construct_loader()
        
    def _construct_loader(self):
        """
        Construct the video loader.
        """
        csv_file_name = {
            "pretrain": "train",
            "finetune": "train",
            "val": "val",
            "test": "val",
        }
        path_to_file = os.path.join(
            self.path_to_data_dir,
            "lists",
            "{}.csv".format(csv_file_name[self.mode]),
        )
        
        assert os.path.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._video_meta = {}
        self.cls_name_to_id_map = {}
        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()[1:]):
                try:
                    line_split = path_label.split(",")
                    assert len(line_split) == 6
                except Exception as e:
                    print(f"CSV format error in file: {path_to_file}")
                    raise e
                # _, label, path = path_label.split()
                cls_name, youtube_id, start, stop, _, _ = line_split
                cls_name = cls_name.strip('"')
                
                if cls_name not in self.cls_name_to_id_map:
                    self.cls_name_to_id_map[cls_name] = len(self.cls_name_to_id_map)
                label = self.cls_name_to_id_map[cls_name]
                
                path = f"{youtube_id}_{int(start):06d}_{int(stop):06d}.mp4"
                full_path = os.path.join(self.path_to_data_dir,
                                         csv_file_name[self.mode]+"_288px",
                                         path)
                
                for idx in range(self.num_clips):
                    self._path_to_videos.append(full_path)
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self.num_clips + idx] = {}
                    
        assert (len(self._path_to_videos) > 0), "Failed to load Kinetics split {} from {}".format(
            self.mode, path_to_file
        )
        print(
            "Constructed kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )
    
    def __getitem__(self, index):
        # Kinetics testing entails multi-view (temporal and spatial) sampling.
        if self.mode in ["pretrain", "finetune", "val"]:
            # Setting to -1 for random sampling for pretrain/finetune/val
            temporal_sample_index = -1
            spatial_sample_index = -1
        elif self.mode in ["test"]:
            # Setting temporal to integer count up to K clips
            temporal_sample_index = self._spatial_temporal_idx[index] // self.test_num_spatial_crops
            # Setting spatial to 0, 1, 2 for different spatial locations.
            spatial_sample_index = self._spatial_temporal_idx % self.test_num_spatial_crops
            
            # no spatial resizing should be done at test time
            assert (not self.pre_crop_size) and (not self.jitter_scale_relative) and (not self.jitter_aspect_relative)
            
        for i_try in range(self.num_retries):
            # Skip if backend=torchvision and examples cause unrecoverable errors.
            if (self.backend == "torchvision" and 
                "/".join(self._path_to_videos[index].split("/")[-1:]) in set([
                    # These are for CVDF
                    # "hA0MPNIdQLY_000031_000041.mp4",
                    # "93Rr9Izzk-Q_000009_000019.mp4",
                    # "EHS9nzV7Lq4_000006_000016.mp4",
                    # "J547SN2w5W8_000012_000022.mp4",
                    # "ejzZqi13zfI_000002_000012.mp4",
                    # "LhZ1jlQXT3w_000006_000016.mp4",
                    # below is for 288px
                    "93Rr9Izzk-Q_000009_000019.mp4", # train set
                    "J547SN2w5W8_000012_000022.mp4", # train set
                    "hA0MPNIdQLY_000031_000041.mp4", # train set
                    "EHS9nzV7Lq4_000006_000016.mp4", # train set
                    "ejzZqi13zfI_000002_000012.mp4", # validation set
            ])):
                print(
                    "SKIPPED example {} with backend {}.".format(
                        self._path_to_videos[index],
                        self.backend
                    )
                )
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue
            
            # Open video container.
            video_container = None
            try:
                video_container = get_video_container(
                    handle=self._path_to_videos[index],
                    multi_thread_decode=self.enable_multi_thread_decode,
                    backend=self.backend
                )
            except Exception as e:
                if self.log_missing_files:
                    print(
                        "LOAD failed for example {} with error {}".format(
                            self._path_to_videos[index], e
                        )
                    )
                if i_try >= 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue
            
            # Decode video from container.
            try:
                frames, fps = decode.decode(
                    video_container=video_container,
                    video_min_dimension=self.pre_crop_size, # must be set to reduce torchvision errors
                    backend=self.backend
                )
            except Exception as e:
                if self.log_bad_decode:
                    print(
                        "DECODE failed for example {} with error {}".format(
                            self._path_to_videos[index], e
                        )
                    )
                if i_try >= 2:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue
            
            frames_list = []
            label_list = []
            
            label = self._labels[index]
            # sampled self.repeat_aug number of examples
            for _ in range(self.repeat_aug):
                clip_size = self.sampling_rate * self.num_frames * fps / self.target_fps
                start_idx, end_idx = decode.get_video_start_end_idx(
                    video_size=frames.shape[0],
                    clip_size=clip_size,
                    clip_idx=temporal_sample_index,
                    num_clips=self.num_clips
                )
                
                sampled_frames = decode.temporal_sampling(
                    frames=frames,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    num_samples=self.num_frames
                )
                
                # set to both to pre-crop size
                sampled_frames = sampled_frames.permute(0, 3, 1, 2) # T, C, H, W
                # sampled_frames = tvF.resize(sampled_frames, size=self.pre_crop_size)
                
                T, C, H, W = sampled_frames.shape # reset values after sampling and resizing
                
                if self.rand_aug:
                    aug_transform = create_random_augment(
                        input_size=(sampled_frames.size(1), sampled_frames.size(2)),
                        auto_augment=self.aa_type,
                        interpolation="bicubic",
                    )
                    
                    frame_list = [tvF.to_pil_image(sampled_frames[i]) for i in range(T)]
                    augmented_frame_list = aug_transform(frame_list)
                    sampled_frames = torch.stack([tvF.to_tensor(augmented_frame_list[i])
                                                for i in range(T)])
                
                sampled_frames = utils.tensor_normalize(
                    sampled_frames.permute(0, 2, 3, 1), # T, H, W, C,
                    self.norm_mean,
                    self.norm_std,
                ).permute(3, 0, 1, 2) # C, T, H, W
                
                sampled_frames = spatial_sampling_v2(
                    frames=sampled_frames,
                    spatial_idx=spatial_sample_index,
                    crop_size=self.crop_size,
                    jitter_scale=self.jitter_scale_relative,
                    jitter_aspect=self.jitter_aspect_relative,
                    random_horizontal_flip=self.random_horizontal_flip,
                )
                
                if self.rand_erase_prob > 0.0:
                    erase_transform = RandomErasing(
                        probability=self.rand_erase_prob,
                        mode=self.rand_erase_mode,
                        max_count=self.rand_erase_count,
                        num_splits=self.rand_erase_count
                    )
                    
                    sampled_frames = sampled_frames.permute(1, 0, 2, 3) # T, C, H, W
                    sampled_frames = erase_transform(sampled_frames)
                    sampled_frames = sampled_frames.permute(1, 0, 2, 3) # C, T, H, W
                
                frames_list.append(sampled_frames)
                label_list.append(label)
            frames = torch.stack(frames_list, dim=0)
            
            if self.mode in ["test"]:
                return frames, torch.tensor(label_list), index
            else:
                return frames, torch.tensor(label_list)
        else:
            raise RuntimeError(
                f"Failed to fetch video after {self.num_retries} retries."
            )
            
    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
            
    
                
                
                
            

            
            
            
            
        