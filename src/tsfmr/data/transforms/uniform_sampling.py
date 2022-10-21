import torch

class UniformSampling(object):
    def __init__(self, num_frames):
        self.num_frames = num_frames
        
    def __call__(self, sample):
        # assume zero padding.to
        sample_len = torch.sum(~(sample == 0.).flatten(1).all(dim=-1))
        
        # sample_len = sample.shape[0]
        segment_len = sample_len / self.num_frames
        segment_starts = torch.arange(0, sample_len, segment_len, dtype=torch.float64)
        
        rand = torch.rand(self.num_frames, dtype=torch.float64)
        segment = rand * segment_len
        
        indices = torch.floor(segment + segment_starts).type(torch.int64)
        sample = sample[indices]
        return sample
        
if __name__ == "__main__":
    transform = UniformSampling(32)
    x = torch.ones((300, 1))
    for _ in range(1000000):
        __ = transform(x)
    print(x)