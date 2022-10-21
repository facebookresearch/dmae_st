# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import numpy as np
import os
import torch
from torch.utils import data as torchdata
from dmae.util.kinetics_bbox import KineticsBbox
from torchvision.transforms.functional import resize

from torchvision.utils import make_grid
from torchvision.io import write_png
from mae.main_test import get_args_parser
from mae.util.kinetics import Kinetics
from mae import models_vit
from iopath.common.file_io import g_pathmgr as pathmgr
import mae.util.misc as misc

from mae.util.pos_embed import interpolate_pos_embed

from torch import Tensor
from torch.utils.data import Sampler
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

def main(args):
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))
    
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    num_spatial_crops, num_ensemble_views = 3, 10
    
    dataset_test = KineticsBbox(
        mode="test",
        path_to_data_dir="/datasets01/kinetics/092121/400",
        path_to_bbox_dir="/data/home/alexnw/alexnw/projects/skel_act_recg/data/kinetics_400_annotations-videos",
        num_retries=12,
        sampling_rate=args.sampling_rate,
        num_frames=args.num_frames,
        crop_size=args.input_size,
        pre_crop_size=256,
        repeat_aug=1,
        test_num_spatial_crops=num_spatial_crops,
        test_num_ensemble_views=num_ensemble_views,
    )
    
    args.batch_size = num_spatial_crops * num_ensemble_views # each batch is a full example
    sampler_test = RandomBlockSampler(dataset_test,
                                      block_size=args.batch_size) # always in the same order
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    model_mae = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        **vars(args),
    )
    model_mae = load_ckpt(model_mae, args.finetune_mae).cuda()
    
    model_dmae = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        **vars(args),
    )
    model_dmae = load_ckpt(model_dmae, args.finetune_dmae).cuda()
    
    model_mae.eval()
    model_dmae.eval()
    
    softmax = torch.nn.Softmax(dim=1).cuda()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("RR", misc.SmoothedValue(window_size=1, fmt="{global_avg:.3f}"))
    metric_logger.add_meter("RR_maskp", misc.SmoothedValue(window_size=1, fmt="{global_avg:.3f}"))
    metric_logger.add_meter("WW", misc.SmoothedValue(window_size=1, fmt="{global_avg:.3f}"))
    metric_logger.add_meter("WW_maskp", misc.SmoothedValue(window_size=1, fmt="{global_avg:.3f}"))
    metric_logger.add_meter("RW", misc.SmoothedValue(window_size=1, fmt="{global_avg:.3f}"))
    metric_logger.add_meter("RW_maskp", misc.SmoothedValue(window_size=1, fmt="{global_avg:.3f}"))
    metric_logger.add_meter("WR", misc.SmoothedValue(window_size=1, fmt="{global_avg:.3f}"))
    metric_logger.add_meter("WR_maskp", misc.SmoothedValue(window_size=1, fmt="{global_avg:.3f}"))
    
    for cur_iter, (images, bbox_map, labels, video_idx) in enumerate(
        metric_logger.log_every(data_loader_test, 2, "")
    ):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        video_idx = video_idx.cuda(non_blocking=True)
        
        if len(images.shape) == 6:
            b, r, c, t, h, w = images.shape
            images = images.view(b * r, c, t, h, w)
            labels = labels.view(b * r)
            if len(bbox_map.shape) == 6:
                bbox_map = bbox_map.reshape(b * r, 1, t, h, w)
            
        with torch.no_grad():
            pred_mae = softmax(model_mae(images))
            pred_dmae = softmax(model_dmae(images))
            
        label_mae = torch.mode(torch.argmax(pred_mae, dim=-1), dim=-1).values
        label_dmae = torch.mode(torch.argmax(pred_mae, dim=-1), dim=-1).values
        true_label = torch.mode(labels).values
        
        maskp = bbox_map.mean()
        
        if label_mae == true_label and label_dmae == true_label: 
            metric_logger.update(RR=1., WW=0., WR=0., RW=0., RR_maskp=maskp)
            f = f"experiments/dmae-st/SAMPLE/incorrect_test/RR/img/{cur_iter}.png"
        elif label_mae != true_label and label_dmae != true_label:
            metric_logger.update(RR=0., WW=1., WR=0., RW=0., WW_maskp=maskp)
            f = f"experiments/dmae-st/SAMPLE/incorrect_test/WW/img/{cur_iter}.png"
        elif label_mae == true_label and label_dmae != true_label:
            metric_logger.update(RR=0., WW=0., WR=1., RW=0., RW_maskp=maskp)
            f = f"experiments/dmae-st/SAMPLE/incorrect_test/RW/{cur_iter}.png"
        else:
            metric_logger.update(RR=0., WW=0., WR=0., RW=1., WR_maskp=maskp)
            f = f"experiments/dmae-st/SAMPLE/incorrect_test/WR/{cur_iter}.png"
        
        bbox = bbox_map.tile(1, 3, 1, 1, 1)
        sample = images.permute(0, 2, 1, 3, 4).detach().cpu() # 2, T, C, H, W
        bbox = bbox.permute(0, 2, 1, 3, 4).detach().cpu()
        third = sample * bbox
        
        combined = torch.cat((sample[:, :, None],
                            bbox[:, :, None],
                            third[:, :, None]),
                            dim=2).flatten(0, 2)
        combined = resize(combined, 200) * 255.
        N = combined.shape[0]
        img_grid = make_grid(combined.type(torch.uint8), nrow=3)
        
        dir = os.path.dirname(f)
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        write_png(img_grid, f)
        
    
class RandomBlockSampler(Sampler[int]):
    data_source: Sized

    def __init__(self, data_source: Sized, block_size: int,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.block_size = block_size
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        n_blocks = n // self.block_size
        block_list = torch.randperm(n_blocks, generator=generator).tolist()
        
        idx_list = []
        for x in block_list:
            idx_list.extend([y for y in range(x, x + self.block_size)])

        yield from idx_list

    def __len__(self) -> int:
        return self.num_samples 
    
def load_ckpt(model, path):
    with pathmgr.open(path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    
    print("Load pre-trained checkpoint from: %s" % path)
    if "model" in checkpoint.keys():
        checkpoint_model = checkpoint["model"]
    else:
        checkpoint_model = checkpoint["model_state"]
    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)
    checkpoint_model = misc.convert_checkpoint(checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    return model

if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--finetune_mae", type=str,
                        default="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/finetune-frac100/checkpoints/checkpoint-00090.pth")
    parser.add_argument("--finetune_dmae", type=str,
                        default="experiments/dmae-st/DMAE_ST-200epochs-TempCos150_050/finetune-frac100/checkpoints/checkpoint-00099.pth")
    
    args = parser.parse_args()
    main(args)
