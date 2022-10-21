#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import pickle as pkl
import random
from pprint import pprint
from time import time
from typing import Any, Dict, Tuple

import dmae.util.decoder.decoder_v2 as decode
import numpy as np
import torch
import torchvision.transforms.functional as tvF
from dmae.util.decoder import transform
from dmae.util.decoder.bboxes import bbox_to_binary_map
from dmae.util.decoder.sampling import spatial_sampling_v2
from mae.util.decoder import utils
from mae.util.decoder.random_erasing import RandomErasing
from mae.util.decoder.transform import create_random_augment
from mae.util.decoder.video_container import get_video_container
from torch.utils.data import Dataset


class KineticsBbox(Dataset):
    """
    Kinetics video loader with preprocessed bounding boxes.
    Construct the kinetis video loader then sample clips from the videos.
    For training and validation a single clip is randomly sampled from every video with random cropping, scaling and flipping.
    For testing, multiple clips are sampled uniformly with uniform cropping to over the entire video.
    In uniform cropping, we take the left, center and right crop if width > height else take top, center, and bottom crop.
    
    Bounding boxes are loaded and converted as a set of vectors describing a box.
    They are converted into a binary map of the same resolution as the video frame which can be augmented and sampled
    identically to the original frames. It can also be normalized into a probability distribution.
    """
    
    def __init__(
        self,
        # required arguments
        mode: str,
        path_to_data_dir: str,
        path_to_bbox_dir: str,
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
        # rand_aug: bool=None,
        # aa_type: str=None,
        # rand_erase_prob: float=0.0,
        # rand_erase_mode: str=None,
        # rand_erase_count: int=None,
        # normalization
        norm_mean: Tuple[float]=(0.45, 0.45, 0.45),
        norm_std: Tuple[float]=(0.225, 0.225, 0.225),
        # other parameters
        enable_multi_thread_decode: bool=False,
        num_retries: int=12,
        backend: str="torchvision",
        log_missing_files: bool=False,
        log_bad_decode: bool=True,
        bbox_base_size: int=540,
    ):
        """
        Kinetics dataloading.
        Loads the kinetics dataset and applies transformations to sequences of images.
        
        Also loads bbox data extracted for the kinetics dataset. Bboxes are then converted to
        a binary map of in or out of box that are augmented to correspond to the sampled video examples.

        NOTE: Modes other than "pre-train" and random augmentations are disabled.

        Args:
            mode (str): data split / loading mode ["pretrain", "finetune", "val", "test"]
            path_to_data_dir (str): Path to kinetics data directory
            path_to_bbox_dir (str): Path to kinetics bbox data directory
            sampling_rate (int): number of frames between sampled frames
            num_frames (int): total number of frames sampled
            crop_size (int): output frame H and W
            repeat_aug (int, optional): times to sample spatio-temporally sample from each vid. Defaults to 1.
            target_fps (int, optional): video target fps. Defaults to 30.
            random_horizontal_flip (bool, optional): enable random horizontal flipping. Defaults to None.
            pre_crop_size (Tuple[int], optional): shortest dimension scaling of video upon loading. Defaults to None.
            jitter_scale_relative (Tuple[int], optional): scale for random_resize_crop. Defaults to None.
            jitter_aspect_relative (Tuple[int], optional): aspect ratio for RRC. Defaults to None.
            norm_mean (Tuple[int], optional): frame normalization mean. Defaults to (0.45, 0.45, 0.45).
            norm_std (Tuple[int], optional): frame normalization std. Defaults to (0.225, 0.225, 0.225).
            enable_multi_thread_decode (bool, optional): multi-thread video decoding. Defaults to False.
            num_retries (int, optional): number of attempts to lead AN example. Defaults to 12.
            backend (str, optional): loading backend ("torchvision", "pyav"). Defaults to "torchvision".
            log_missing_files (bool, optional): log error when example video not found. Defaults to False.
            log_bad_decode (bool, optional): log error when example video decode failed. Defaults to True.
            bbox_base_size (int, optional): frame short-side dimension when extracting bounding boxes. Defaults to 540.
        """
        assert mode in [
            "pretrain",
            "val",
            "test",
            "testset"
        ], (f"Split {mode} not supported for KineticsBbox. "
            f"Please select from (pretrain).")
        
        assert backend in [
            "torchvision", 
            "pyav"
        ], (f"Backend {backend} not support for KineticsBbox. "
            "Please select from (torchvision, pyav).")
        
        self.mode = mode
        self.path_to_data_dir = path_to_data_dir
        self.path_to_bbox_dir = path_to_bbox_dir
        
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
        
        # self.rand_aug = rand_aug
        # self.aa_type = aa_type
        # self.rand_erase_prob = rand_erase_prob
        # self.rand_erase_mode = rand_erase_mode
        # self.rand_erase_count = rand_erase_count
        
        # normalization
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        
        # other loading with defaults
        self.bbox_base_size = bbox_base_size
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
        if mode in ["pretrain", "val"]:
            self.num_clips = 1
            # if test_num_spatial_crops or test_num_ensemble_views:
            #     print(f"WARNING: spatial_crops {test_num_spatial_crops} or ensemble views{test_num_ensemble_views}"
            #           f"set for {mode} mode.")
        elif mode in ["test", "testset"]:
            self.num_clips = test_num_ensemble_views * test_num_spatial_crops
        print(f"{self.mode} mode, num_clips={self.num_clips}.")
        
        # check random augmentation settings
        # most random operations are not supported as they must replicated for the bboxes
        # if mode in ["pretrain", "val", "test"]:
        #     # if rand_aug:
        #     #     print(f"WARNING: rand_aug={rand_aug} in {mode} mode. Setting to False.")
        #     # if rand_erase_prob and rand_erase_prob > 0.0:
        #     #     print(f"WARNING: rand_erase={rand_erase_prob} in {mode} mode. Setting to 0.0.")
        #     self.rand_aug = False
        #     self.rand_erase_prob = 0.0
        # elif mode in ["finetune"]:
        #     self.rand_aug = rand_aug
        # print(f"{self.mode} mode, rand_aug={self.rand_aug}.")
        
        # if rand_erase_prob > 0.0:
        #     assert rand_erase_mode is not None, f"rand_erase_mode({rand_erase_mode}) while rand_erase_prob({rand_erase_prob})"
        #     assert rand_erase_count is not None, f"rand_erase_mode({rand_erase_count}) while rand_erase_prob({rand_erase_count})"
        
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0
        
        print(f"Constructing KineticsBbox in {mode} mode...")
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
            "testset": "test"
        }
        path_to_file = os.path.join(
            self.path_to_data_dir,
            "lists",
            "{}.csv".format(csv_file_name[self.mode]),
        )
        if not os.path.isfile(path_to_file):
            path_to_file = os.path.join(
                self.path_to_data_dir,
                "annotations",
                "{}.csv".format(csv_file_name[self.mode])
            )
        
        assert os.path.exists(path_to_file), "{} dir not found".format(path_to_file)
        assert os.path.isdir(self.path_to_bbox_dir)
        
        folder = os.path.join(self.path_to_data_dir, csv_file_name[self.mode] + "_288px")
        if not os.path.isdir(folder):
            folder = os.path.join(self.path_to_data_dir, csv_file_name[self.mode])

        self._path_to_videos = []
        self._path_to_bboxes = []
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
                
                #MODIFY THESE PATHS
                full_path = os.path.join(folder,
                                         path)
                bbox_path = os.path.join(self.path_to_bbox_dir,
                                         csv_file_name[self.mode],
                                         cls_name,
                                         path.replace(".mp4", ".pkl"))
                
                for idx in range(self.num_clips):
                    self._path_to_videos.append(full_path)
                    self._path_to_bboxes.append(bbox_path)
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
        elif self.mode in ["test", "testset"]:
            # Setting temporal to integer count up to K clips
            temporal_sample_index = self._spatial_temporal_idx[index] // self.test_num_spatial_crops
            # Setting spatial to 0, 1, 2 for different spatial locations.
            spatial_sample_index = self._spatial_temporal_idx[index] % self.test_num_spatial_crops
            
            # no spatial resizing should be done at test time
            assert (not self.random_horizontal_flip) and (not self.jitter_scale_relative) and (not self.jitter_aspect_relative)
            
        for i_try in range(self.num_retries):
            # Skip if backend=torchvision and examples cause unrecoverable errors.
            if (self.backend == "torchvision" and 
                "/".join(self._path_to_videos[index].split("/")[-1:]) in set([
                    # These are for CVDF
                    "ejzZqi13zfI_000002_000012.mp4",
                    "LhZ1jlQXT3w_000006_000016.mp4",
                    # below is for 288px
                    "93Rr9Izzk-Q_000009_000019.mp4",
                    "J547SN2w5W8_000012_000022.mp4",
                    "hA0MPNIdQLY_000031_000041.mp4",
                    "EHS9nzV7Lq4_000006_000016.mp4",
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
            
            T, H, W, C = frames.shape
            # Load bbox data.
            # Compute scaled frame size, the smallest video dimension is set to self.bbox_base_size
            try:
                bbox_size = tvF.resize(torch.zeros(1, H, W),
                                        size=self.bbox_base_size).shape[-2:]
                
                with open(self._path_to_bboxes[index], 'rb') as bbox_f:
                    bbox_list = pkl.load(bbox_f)
                    assert len(bbox_list) == frames.shape[0], (f"Bbox frame count({len(bbox_list)}) doesn't match"
                                                                f" video frame count({frames.shape[0]}).")
                    ragged_bbox_list_np = np.array(bbox_list, dtype=object)
            except Exception as e:
                if self.log_bad_decode:
                    print(
                        "BBOX LOAD failed for example {} with error {}".format(
                            self._path_to_bboxes[index], e
                        )
                    )
                if i_try >= 2:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue
            
            frames_list = []
            label_list = []
            bbox_map_list = []
            
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
                ).permute(0, 3, 1, 2) # T, C, H, W
                
                # sample the frames from the bounding box binary map
                sampled_ragged_bbox_list_np = decode.temporal_sampling_numpy(
                    frames=ragged_bbox_list_np,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    num_samples=self.num_frames
                )
                
                sampled_bbox_map = torch.zeros(self.num_frames, *bbox_size, 1) # T, H, W, 1
                for idx, bboxes in enumerate(sampled_ragged_bbox_list_np):
                    bboxes = torch.tensor(bboxes[:, :4].astype(np.int64)) # drop probability value
                    sampled_bbox_map[idx] = bbox_to_binary_map(bboxes, bbox_size)
                
                # permute for resize
                sampled_bbox_map = sampled_bbox_map.permute(0, 3, 1, 2)
                
                # set to both to pre-crop size
                # sampled_frames = tvF.resize(sampled_frames, size=self.pre_crop_size)
                sampled_bbox_map = tvF.resize(sampled_bbox_map, size=self.pre_crop_size)
                
                T, C, H, W = sampled_frames.shape # reset values after sampling and resizing
                
                # if self.rand_aug:
                #     aug_transform = create_random_augment(
                #         input_size=(sampled_frames.size(1), sampled_frames.size(2)),
                #         auto_augment=self.aa_type,
                #         interpolation="bicubic",
                #     )
                    
                #     frame_list = [tvF.to_pil_image(sampled_frames[i]) for i in range(T)]
                #     augmented_frame_list = aug_transform(frame_list)
                #     sampled_frames = torch.stack([tvF.to_tensor(augmented_frame_list[i])
                #                                 for i in range(T)])
                
                sampled_frames = utils.tensor_normalize(
                    sampled_frames.permute(0, 2, 3, 1), # T, H, W, C,
                    self.norm_mean,
                    self.norm_std,
                ).permute(3, 0, 1, 2) # C, T, H, W
                sampled_bbox_map = sampled_bbox_map.permute(1, 0, 2, 3) # 1, T, H, W
                
                # Transformations must be the same for frames and bboxes
                # Only scaled resize and crop is supported
                # spatial_sampling_v2
                # sampled_frames = spatial_sampling_v2(
                #     frames=sampled_frames,
                #     spatial_idx=spatial_sample_index,
                #     crop_size=self.crop_size,
                #     jitter_scale=self.jitter_scale_relative,
                #     jitter_aspect=self.jitter_aspect_relative,
                #     random_horizontal_flip=self.random_horizontal_clip,
                # )
                assert spatial_sample_index in [-1, 0, 1, 2]
                if spatial_sample_index == -1: # if index = -1, use random sampling a la RRC and horizontal flip
                    if self.jitter_aspect_relative and self.jitter_scale_relative:
                        sampled_frames, params = transform.random_resized_crop(
                            images=sampled_frames,
                            target_height=self.crop_size,
                            target_width=self.crop_size,
                            scale=self.jitter_scale_relative,
                            ratio=self.jitter_aspect_relative,
                            return_params=True
                        )
                        sampled_bbox_map = transform.resized_crop(
                            images=sampled_bbox_map,
                            target_height=self.crop_size,
                            target_width=self.crop_size,
                            params=params
                        )
                    if self.random_horizontal_flip:
                        sampled_frames, flipped = transform.random_horizontal_flip(
                            prob=0.5,
                            images=sampled_frames,
                            return_params=True
                        )
                        if flipped:
                            sampled_bbox_map = transform.horizontal_flip(
                                images=sampled_bbox_map
                            )
                else:
                    # Deterministic "jitter" + spatial sampling when spatial_sample_index in [0, 1, 2]
                    sampled_frames, params = transform.random_short_side_scale_jitter(
                        images=sampled_frames,
                        min_size=self.crop_size,
                        max_size=self.crop_size,
                        return_params=True
                    )
                    sampled_bbox_map = transform.short_side_scale_jitter(
                        images=sampled_bbox_map,
                        params=params
                    )
                    
                    sampled_frames = transform.uniform_crop(
                        images=sampled_frames,
                        size=self.crop_size,
                        spatial_idx=spatial_sample_index
                    )
                    sampled_bbox_map = transform.uniform_crop(
                        images=sampled_bbox_map,
                        size=self.crop_size,
                        spatial_idx=spatial_sample_index
                    )
                
                # if self.rand_erase_prob > 0.0:
                #     erase_transform = RandomErasing(
                #         probability=self.rand_erase_prob,
                #         mode=self.rand_erase_mode,
                #         max_count=self.rand_erase_count,
                #         num_splits=self.rand_erase_count
                #     )
                    
                #     sampled_frames = sampled_frames.permute(1, 0, 2, 3) # T, C, H, W
                #     sampled_frames = erase_transform(sampled_frames)
                #     sampled_frames = sampled_frames.permute(1, 0, 2, 3) # C, T, H, W
                
                frames_list.append(sampled_frames)
                label_list.append(label)
                bbox_map_list.append(sampled_bbox_map)
            frames = torch.stack(frames_list, dim=0)
            bbox_map = torch.stack(bbox_map_list, dim=0)
            
            if self.mode in ["test", "testset"]:
                return frames, bbox_map, torch.tensor(label_list), index
            elif self.mode in ["val"]:
                return frames, bbox_map, torch.tensor(label_list)
            elif self.mode in ["pretrain"]:
                return frames, bbox_map, torch.tensor(label_list)
            else:
                raise ValueError(f"Mode error. {self.mode}")
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
