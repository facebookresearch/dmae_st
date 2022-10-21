#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os

import torch
import torch.utils.data
from torchvision import transforms

from mae.util.decoder import decoder as decoder
from mae.util.decoder import video_container as container


class KineticsRaw(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(
        self,
        mode,
        path_to_data_dir,
        enable_multi_thread_decode=False,
        min_resize_res=0,
    ):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
            "all"
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode

        self._video_meta = {}
        self._path_to_data_dir = path_to_data_dir
        self._enable_multi_thread_decode = enable_multi_thread_decode
        self._min_resize_res = min_resize_res

        # print(self)
        print(locals())

        print("Constructing Kinetics {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        csv_file_name = {
            "train": ["train"],
            "val": ["val"],
            "test": ["test"],
            "all": ["train", "val", "test"],
        }
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self.cls_name_to_id_map = {}
        for mode_path in csv_file_name[self.mode]:
            path_to_file = os.path.join(
                self._path_to_data_dir,
                "annotations",
                "{}.csv".format(mode_path),
            )
            assert os.path.exists(path_to_file), "{} dir not found".format(path_to_file)

            with open(path_to_file, "r") as f:
                start_size = len(self._video_meta)
                for clip_idx, path_label in enumerate(f.read().splitlines()[1:]):
                    line_split = path_label.split(",")
                    assert len(line_split) == 6
                    # _, label, path = path_label.split()
                    
                    cls_name, youtube_id, start, stop, _, _ = line_split
                    cls_name = cls_name.strip('"')
                    
                    if cls_name not in self.cls_name_to_id_map:
                        self.cls_name_to_id_map[cls_name] = len(self.cls_name_to_id_map)
                        
                    label = self.cls_name_to_id_map[cls_name]
                    path = f"{youtube_id}_{int(start):06d}_{int(stop):06d}.mp4"
                    full_path = os.path.join(self._path_to_data_dir,
                                            "videos", mode_path, cls_name,
                                            path)
                    
                    self._path_to_videos.append(full_path)
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(0)
                    self._video_meta[start_size + clip_idx] = {}
            assert (
                len(self._path_to_videos) > 0
            ), "Failed to load Kinetics split {} from {}".format(
                self.mode, path_to_file
            )
            assert len(self._path_to_videos) == len(self._video_meta)
            print(
                "Constructing kinetics dataset (size: {}) from split {}".format(
                    len(self._path_to_videos), path_to_file
                )
            )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        
        rigid_decode_all_video = True #DECODE ALL VIDEO 
        
        # We are skipping the 4 examples that malloc torchvision loading.
        # if "/".join(self._path_to_videos[index].split("/")[-2:]) in set([
        #     "tango dancing/hA0MPNIdQLY_000031_000041.mp4",
        #     "golf putting/93Rr9Izzk-Q_000009_000019.mp4",
        #     "extinguishing fire/EHS9nzV7Lq4_000006_000016.mp4",
        #     "eating spaghetti/J547SN2w5W8_000012_000022.mp4",
        #     "golf putting/ejzZqi13zfI_000002_000012.mp4",
        #     "kicking field goal/0R5ZqkVbpLo_000000_000010.mp4",
        #     "kicking soccer ball/db9S59stuUY_000014_000024.mp4",
        #     "jogging/wk-HnjDJ4FA_000002_000012.mp4",
        #     "juggling balls/LQNTqoBOhDA_000509_000519.mp4",
        # ]):
        #     print("Hit MALLOC example: {}".format(
        #         self._path_to_videos[index]
        #     ))
        #     return None
        
        try:
            video_container = container.get_video_container(
                self._path_to_videos[index],
                self._enable_multi_thread_decode,
                backend="pyav"
            )
        except Exception as e:
            print("Failed to LOAD video from {} with error {}".format(
                self._path_to_videos[index], e
            ))
            return None
        
        try:
            # None parameters are irrelevant as we decode all frames
            frames, fps, decode_all_video = decoder.decode(
                container=video_container,
                sampling_rate=None,
                num_frames=None,
                clip_idx=None,
                num_clips=None,
                video_meta=self._video_meta[index],
                target_fps=None,
                max_spatial_scale=self._min_resize_res,
                use_offset=None,
                rigid_decode_all_video=rigid_decode_all_video,
                backend="pyav"
            )
            # frames, fps, decode_all_video = decoder.torchvision_decode(
            #     video_handle=video_container,
            #     sampling_rate=None,
            #     num_frames=None,
            #     clip_idx=None,
            #     video_meta=self._video_meta[index],
            #     num_clips=None,
            #     target_fps=None,
            #     max_spatial_scale=self._min_resize_res, # do not rescale
            #     use_offset=None,
            #     rigid_decode_all_video=rigid_decode_all_video,
            # )
            assert decode_all_video == True
        except Exception as e:
            print("Failed to DECODE video from {} with error {}".format(
                self._path_to_videos[index], e
            ))
            return None

        if (frames is None) or (not torch.isfinite(frames).all()) or (frames.size(0) == 0):
            print(
                "Decoded video idx {} from {}; empty or non-finite.".format(
                    index, self._path_to_videos[index]
                )
            )
            return None
        return {"video": frames, 
                "file": "/".join(self._path_to_videos[index].split("/")[-3:]), 
                "abs_path": self._path_to_videos[index],
                "fps": fps}

    def _frame_to_list_img(self, frames):
        img_list = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

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

def drop_collate_fn(batch):
    """
    Removes examples in the batch that are None. This is to be used in conjunction with
    a dataset that returns None at load-time if there is an issue with the example.
    
    NOTE: This is not an ideal solution, we will get variable batch sizes and also empty
    batches for small batch sizes. This will be used for inference to process data -- 
    uneven and None batches cna be handled.

    Args:
        batch (List): List of examples from dataset.__getitem__()

    Returns:
        _type_: collated batch of examples
    """
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)