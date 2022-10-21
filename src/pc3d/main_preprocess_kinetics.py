# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import argparse
import datetime
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict

import cv2
import mae.util.misc as misc
import numpy as np
import torch
import torch.distributed as dist
import torchvision.models.detection as models
from torchvision.io import write_png, write_video
from torchvision.transforms.functional import resize
from torchvision.utils import draw_bounding_boxes, make_grid

from pc3d.util.kinetics_raw import KineticsRaw, drop_collate_fn

# from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def get_args_parser():
    parser = argparse.ArgumentParser("Kinetics Preprocessing", add_help=False)
    
    parser.add_argument("--output_dir", default="experiments/pc3d/preprocess-kinetics", type=str,
                        help="output logging directory")
    parser.add_argument("--device", default="cuda", 
                        help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--model", default="fasterrcnn_resnet50_fpn", type=str,
                        help="model name from torchvision.models.detection")
    
    parser.add_argument("--data_dir", default="/datasets01/kinetics_400/k400", type=str,)
    parser.add_argument("--data_save_dir", required=True, type=str,
                        help="Where to save the pkl files.")
    parser.add_argument("--split", default="all", type=str,
                        help="Kinetics dataset split: train, val, test, all ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Needs to be 1 as we are loading full videos -> variable shape")
   
    parser.add_argument("--distributed", action="store_true",
                        help="Run in distributed mode.")
    parser.add_argument("--fb_env", action="store_true",
                        help="not used")
    parser.add_argument("--dist_url", default="env://", 
                        help="url used to set up distributed training")
    parser.add_argument("--world_size", default=1, type=int,
                        help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N",
                        help="start epoch")
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--pin_mem", action="store_true",
                        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    
    return parser


def main(args):
    if args.distributed:
        misc.init_distributed_mode(args)
        
        
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))
    
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dataset = KineticsRaw("all", 
                          path_to_data_dir=args.data_dir,
                          min_resize_res=540)
    
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.SequentialSampler(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler_train,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=drop_collate_fn
    )
    
    model_fn = models.__dict__[args.model]
    model = model_fn(pretrained=True)
    model.eval()
    model.to(device)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            # find_unused_parameters=True,
        )
        model_without_ddp = model.module
    
    start_time = time.time()
    metric_logger = misc.MetricLogger(delimiter="  ", default_print=True)
    header = "Epoch: [{}]".format(0)
    print_freq = 10
    save_freq = 100
    chunk_size = 64
    
    for data_iter_step, batch in enumerate(
        metric_logger.log_every(dataloader, print_freq, header)
    ):
        if batch is None:
            continue
        
        video = batch['video'].flatten(0, 1).permute(0, 3, 1, 2).type(torch.float32) / 255. # N, C, H, W
        file = batch['file'][0]
        N, C, H, W = video.shape
        bounding_boxes = []
        with torch.no_grad():
            video_chunks = (video[pos:pos + chunk_size] for pos in range(0, video.shape[0], chunk_size))
            for video_chunk in video_chunks:
                video_chunk = video_chunk.to(device)
                outputs = model(video_chunk)
                for output in outputs:
                    filter = torch.logical_and(output["labels"] == 1, output["scores"] >= 0.8)
                    box_tensor = torch.cat((output["boxes"][filter], output["scores"][filter][:, None]), dim=-1)
                    bounding_boxes.append(box_tensor.cpu())
                del video_chunk
                    

            out_file = file[:-4] + ".pkl"
            out_path = os.path.join(args.data_save_dir, out_file)
            dir = os.path.dirname(out_path)
            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)
            with open(out_path, 'wb') as f:
                pickle.dump([x.numpy() for x in bounding_boxes], f)

            if data_iter_step % save_freq == 0:
                try:
                    video_drawn = torch.zeros_like(video, device="cpu")
                    for frame_idx in range(video.shape[0]):
                        img = (video[frame_idx] * 255.).type(torch.uint8).cpu()
                        boxes = torch.round(bounding_boxes[frame_idx])
                        
                        boxed_img = draw_bounding_boxes(img, boxes[:, :4], colors=(0, 255, 0), width=4)
                        video_drawn[frame_idx] = boxed_img

                    f = str(args.output_dir) + "/outputs/vids/" + file
                    dir = os.path.dirname(f)
                    if not os.path.exists(dir):
                        os.makedirs(dir, exist_ok=True)
                    write_video(f, video_drawn.permute(0, 2, 3, 1), int(batch['fps'][0]))
                    
                    f = str(args.output_dir) + "/outputs/imgs/" + file[:-4] + ".png"
                    dir = os.path.dirname(f)
                    if not os.path.exists(dir):
                        os.makedirs(dir, exist_ok=True)
                    resized_imgs = resize(video_drawn, 200)
                    img_grid = make_grid(resized_imgs.type(torch.uint8), nrow=int(N ** 0.6))
                    write_png(img_grid, f)
                except Exception as e:
                    print(f"Outputting image/video {file} encountered error: {e}.")
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())

# if __name__ == "__main__":
#     args = get_args_parser()
#     args = args.parse_args()
#     if args.output_dir:
#         Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#     main(args)
