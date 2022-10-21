

from audioop import reverse
import math
import numpy as np
import os
import torch
from torch.utils import data as torchdata
from dmae.util.kinetics_bbox import KineticsBbox
from torchvision.transforms.functional import resize

from torchvision.utils import make_grid
from torchvision.transforms import Normalize
from torchvision.io import write_png
from mae.main_test import get_args_parser as main_get_args_parser
from mae.util.kinetics import Kinetics
from mae import models_vit
from iopath.common.file_io import g_pathmgr as pathmgr
from torch import distributed as dist
import mae.util.misc as misc

from mae.util.pos_embed import interpolate_pos_embed

from torch import Tensor
from torch.utils.data import Sampler
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

def main(args):
    misc.init_distributed_mode(args)
    
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
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_test = DistributedBlockSampler(block_size=args.batch_size,
                                               dataset=dataset_test,
                                               num_replicas=num_tasks,
                                               rank=global_rank,
                                               shuffle=False)
    else:
        sampler_test = BlockSampler(dataset_test,
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
    
    if args.distributed:
        model_mae = torch.nn.parallel.DistributedDataParallel(
            model_mae, device_ids=[torch.cuda.current_device()]
        )
        model_without_ddp = model_mae.module
        model_dmae = torch.nn.parallel.DistributedDataParallel(
            model_dmae, device_ids=[torch.cuda.current_device()]
        )
        model_without_ddp = model_dmae.module
    
    softmax = torch.nn.Softmax(dim=1).cuda()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("RR", misc.SmoothedValue(window_size=20, fmt="{global_avg:.3f}"))
    metric_logger.add_meter("RR_maskp", misc.SmoothedValue(window_size=20, fmt="{global_avg:.3f}"))
    metric_logger.add_meter("WW", misc.SmoothedValue(window_size=20, fmt="{global_avg:.3f}"))
    metric_logger.add_meter("WW_maskp", misc.SmoothedValue(window_size=20, fmt="{global_avg:.3f}"))
    metric_logger.add_meter("RW", misc.SmoothedValue(window_size=20, fmt="{global_avg:.3f}"))
    metric_logger.add_meter("RW_maskp", misc.SmoothedValue(window_size=20, fmt="{global_avg:.3f}"))
    metric_logger.add_meter("WR", misc.SmoothedValue(window_size=20, fmt="{global_avg:.3f}"))
    metric_logger.add_meter("WR_maskp", misc.SmoothedValue(window_size=20, fmt="{global_avg:.3f}"))
    
    inv_normalize = Normalize(
        mean=[-m/s for m, s in zip(dataset_test.norm_mean, dataset_test.norm_std)],
        std=[1./s for s in dataset_test.norm_std]
    )
    reverse_dict = {v: k for k, v in dataset_test.cls_name_to_id_map.items()}
    
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
        
        label_mae = torch.argmax(pred_mae.sum(dim=0)).detach().cpu()
        label_dmae = torch.argmax(pred_dmae.sum(dim=0)).detach().cpu()
        true_label = torch.mode(labels).values.detach().cpu()
        
        label_mae_val, label_dmae_val, true_label_val = int(label_mae), int(label_dmae), int(true_label)
        
        maskp = bbox_map.nanmean()
        
        if args.distributed:
            cur_iter = (cur_iter * args.world_size) + args.rank
    
        if label_mae == true_label and label_dmae == true_label: 
            metric_logger.update(RR=1., WW=0., WR=0., RW=0., RR_maskp=maskp)
            f = f"{args.output_dir}/output/RR/{cur_iter}-{reverse_dict[true_label_val]}-{reverse_dict[label_mae_val]}_{reverse_dict[label_dmae_val]}.png"
        elif label_mae != true_label and label_dmae != true_label:
            metric_logger.update(RR=0., WW=1., WR=0., RW=0., WW_maskp=maskp)
            f = f"{args.output_dir}/output/WW/{cur_iter}-{reverse_dict[true_label_val]}-{reverse_dict[label_mae_val]}_{reverse_dict[label_dmae_val]}.png"
        elif label_mae == true_label and label_dmae != true_label:
            metric_logger.update(RR=0., WW=0., WR=0., RW=1., RW_maskp=maskp)
            f = f"{args.output_dir}/output/RW/{cur_iter}-{reverse_dict[true_label_val]}-{reverse_dict[label_mae_val]}_{reverse_dict[label_dmae_val]}.png"
        else:
            metric_logger.update(RR=0., WW=0., WR=1., RW=0., WR_maskp=maskp)
            f = f"{args.output_dir}/output/WR/{cur_iter}-{reverse_dict[true_label_val]}-{reverse_dict[label_mae_val]}_{reverse_dict[label_dmae_val]}.png"
        
        bbox = bbox_map.tile(1, 3, 1, 1, 1)
        sample = images.permute(0, 2, 1, 3, 4).detach().cpu() # bs, T, C, H, W
        bbox = bbox.permute(0, 2, 1, 3, 4).detach().cpu()
        
        sample = inv_normalize(sample)
        third = sample * bbox
        
        combined = torch.cat((sample[:, :, None],
                            bbox[:, :, None],
                            third[:, :, None]),
                            dim=2).flatten(1, 2).flatten(0, 1)
        combined = resize(combined, 200) * 255.
        N = combined.shape[0]
        img_grid = make_grid(combined.type(torch.uint8), nrow=3 * args.batch_size)
        
        dir = os.path.dirname(f)
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        write_png(img_grid, f)
        
        dist.barrier()
        torch.cuda.synchronize()
        
    metric_logger.synchronize_between_processes()
    print(metric_logger)
    
class BlockSampler(Sampler[int]):
    data_source: Sized

    def __init__(self, data_source: Sized, block_size: int,
                 num_samples: Optional[int] = None, generator=None,
                 shuffle: bool=False) -> None:
        self.data_source = data_source
        self.block_size = block_size
        self._num_samples = num_samples
        self.generator = generator
        self.shuffle = shuffle
        
        if not shuffle:
            assert generator == None

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
        if not self.shuffle:
            block_list = torch.range(0, n_blocks)
        else:
            block_list = torch.randperm(n_blocks, generator=generator).tolist()
        
        idx_list = []
        for x in block_list:
            idx_list.extend([y for y in range(x * self.block_size, (x + 1) * self.block_size)])

        yield from idx_list

    def __len__(self) -> int:
        return self.num_samples 
    
class DistributedBlockSampler(torchdata.DistributedSampler):
    def __init__(self, dataset: torchdata.Dataset, block_size: int, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False):
        super(DistributedBlockSampler, self).__init__(dataset=dataset,
                                                           num_replicas=num_replicas,
                                                           rank=rank,
                                                           shuffle=shuffle,
                                                           seed=seed,
                                                           drop_last=drop_last)
        
        self.block_size = block_size
        
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas*block_size) / (self.num_replicas * block_size)
            ) * block_size
        else:
            self.num_samples = math.ceil(len(self.dataset) / (self.num_replicas*block_size)) * block_size
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
    
    def __iter__(self) -> Iterator:
        n = len(self.dataset)
        n_blocks = n // self.block_size
        
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            block_list = torch.randperm(n_blocks, generator=g).tolist()
        else:
            block_list = list(range(n_blocks))
            
        if not self.drop_last:
            padding_size = self.total_size//self.block_size - len(block_list)
            if padding_size <= len(block_list):
                block_list += block_list[:padding_size]
            else:
                block_list += (block_list * math.ceil(padding_size / len(block_list)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            block_list = block_list[:(self.total_size//self.block_size)]
        block_list = block_list[self.rank:self.total_size:self.num_replicas]
        indices = []
        for x in block_list:
            indices.extend([y for y in range((x * self.block_size), (x + 1) * self.block_size)])
        assert len(indices) == self.num_samples, f"{len(indices)}, {self.num_samples}"
        return iter(indices)
            
    
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

def get_args_parser():
    parser = main_get_args_parser()
    parser.add_argument("--finetune_mae", type=str,
                        default="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/finetune-frac100/checkpoints/checkpoint-00090.pth")
    parser.add_argument("--finetune_dmae", type=str,
                        default="experiments/dmae-st/DMAE_ST-200epochs-TempCos150_050/finetune-frac100/checkpoints/checkpoint-00099.pth")
    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
