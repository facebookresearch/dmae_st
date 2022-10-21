# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import math
import random
from typing import Any, Dict

import numpy as np
import torch
from torchvision import io
from torchvision.transforms import functional as tvF


def temporal_sampling(
    frames,
    start_idx,
    end_idx,
    num_samples
):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    new_frames = torch.index_select(frames, 0, index)

    return new_frames

def temporal_sampling_numpy(
    frames,
    start_idx,
    end_idx,
    num_samples
):
    index = np.linspace(start_idx, end_idx, num_samples)
    index = np.clip(index, 0, frames.shape[0] - 1).astype(np.int64)
    new_frames = np.take(frames, indices=index, axis=0)

    return new_frames

def get_video_start_end_idx(
    video_size: int,
    clip_size: int,
    clip_idx: int,
    num_clips: int=None,
):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip. Int ranging from [-1, num_clips-1]
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index. Inclusive.
    """
    delta = max(video_size - clip_size, 0)
    
    if clip_idx == -1:
        assert num_clips in {None, 1}, f"num_clips({num_clips}) set when clip_idx({clip_idx}) == -1."
        start_idx = round(random.uniform(0, delta))
    else:
        assert clip_idx < num_clips, f"clip_idx({clip_idx} must be less than num_clips({num_clips})."
        if num_clips == 1:
            # Take the center clip
            start_idx = math.floor(delta / 2)
        else:
            if num_clips == 2:
                print(f"WARNING: num_clips({num_clips}) is 2, this samples from the start and end of the full video.")
            # Take the clip_idx-th segment of num_clips equal divisions of delta
            start_idx =  round(delta * clip_idx / (num_clips - 1))
    end_idx = start_idx + clip_size - 1 # subtract by 1 to ensure this is inclusive
    return start_idx, end_idx
        

def torchvision_decode(
    video_handle,
    video_min_dimension=0,
):
    """
    Simplified torchvision_decode.
    Decodes the full video from the stream.
    Args:
        video_handle (bytes): raw bytes of the video file.
        video_min_dimension (int): the resolution of the spatial shorter
            edge size during decoding.
    Returns:
        frames (tensor): decoded frames from the video.
        fps (float): the number of frames per second of the video.
    """
    # Convert the bytes to a tensor.
    video_tensor = torch.from_numpy(np.frombuffer(video_handle, dtype=np.uint8))

    # Fetch the meta data from the raw video.
    # Tracking the meta info for selective decoding in the future.
    meta = io._probe_video_from_memory(video_tensor)
    video_meta = {}
    # Using the information from video_meta to perform selective decoding.
    video_meta["video_timebase"] = meta.video_timebase
    video_meta["video_numerator"] = meta.video_timebase.numerator
    video_meta["video_denominator"] = meta.video_timebase.denominator
    video_meta["has_video"] = meta.has_video
    video_meta["video_duration"] = meta.video_duration
    video_meta["video_fps"] = meta.video_fps
    video_meta["audio_timebas"] = meta.audio_timebase
    video_meta["audio_numerator"] = meta.audio_timebase.numerator
    video_meta["audio_denominator"] = meta.audio_timebase.denominator
    video_meta["has_audio"] = meta.has_audio
    video_meta["audio_duration"] = meta.audio_duration
    video_meta["audio_sample_rate"] = meta.audio_sample_rate

    fps = video_meta["video_fps"]
        # failed selective decoding
    video_start_pts, video_end_pts = 0, -1
    v_frames, _ = io._read_video_from_memory(
        video_tensor,
        seek_frame_margin=1.0,
        read_video_stream=True,
        read_audio_stream=False,
        video_height=0,
        video_width=0,
        video_min_dimension=video_min_dimension,
        video_pts_range=(video_start_pts, video_end_pts),
        video_timebase_numerator=video_meta["video_numerator"],
        video_timebase_denominator=video_meta["video_denominator"],
    )

    return v_frames, fps

def pyav_decode(
    container,
):
    """
    Simplified pyav_decode.
    Decodes the full video from the stream.
    
    Args:
        container (container): pyav container.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): If True, the entire video was decoded.
    """
    fps = float(container.streams.video[0].average_rate)

    frames = None
    # If video stream was found, fetch video frames from the video.
    if container.streams.video:
        video_frames, max_pts = pyav_decode_stream(
            container=container,
            start_pts=0,
            end_pts=math.inf,
            stream=container.streams.video[0],
            stream_name={"video": 0},
        )
        container.close()

        frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
        frames = torch.as_tensor(np.stack(frames))
    return frames, fps

def pyav_decode_stream(
    container, 
    start_pts, 
    end_pts, 
    stream, 
    stream_name, 
    buffer_size=0
):
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
        buffer_size (int): number of additional frames to decode beyond end_pts.
    Returns:
        result (list): list of frames decoded.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """
    # Seeking in the stream is imprecise. Thus, seek to an ealier PTS by a
    # margin pts.
    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts

def decode(
    video_container: Any,
    video_min_dimension:int=0, # scale of the shorter size during decoding
    backend: str="torchvision"
):
    """
    Rewrite of video decoding that also handles spatial and temporal sampling.

    Args:
        video_container (Any): video decoder, either PyAV container or bytestream depending on backend.
        num_frames (int): total number of video frames to return.
        video_min_dimension (int, optional): Size of smaller dimension, 0 for no resize. Defaults to 0.
        backend (str, optional): Backend decoder: (torchvision, pyav). Defaults to "torchvision".

    Raises:
        Exception: Decode errors or empty decodings

    Returns:
        torch.Tensor: Video tensor (num_frames, H, W, C)
        float: FPS of the video.
    """
    
    
    if backend == "torchvision":
        frames, fps = torchvision_decode(
            video_container,
            video_min_dimension,
        )
    elif backend == "pyav":
        frames, fps = pyav_decode(
            video_container,
        )
        if video_min_dimension != 0:
            frames = tvF.resize(
                frames.permute(0, 3, 1, 2),
                size=video_min_dimension
            ).permute(0, 2, 3, 1)
    
    if frames is None or frames.size(0) == 0:
        raise Exception(f"Frames {frames} is None or empty.")
    if not torch.isfinite(frames).all():
        raise Exception(f"Frames is not finite.")
    
    return frames, fps
