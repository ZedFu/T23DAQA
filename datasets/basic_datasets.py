import os 
import torch, torchvision
import numpy as np
from einops import rearrange

import decord
decord.bridge.set_bridge('torch')

class T23dqaDataset(torch.utils.data.Dataset):
    def __init__(self,root,mode,ann_file,sample_num,size):
        super(T23dqaDataset).__init__()
        self.root = root
        self.mode = mode
        self.ann_file = ann_file
        self.sample_num = sample_num
        self.size = size
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        if isinstance(self.ann_file, list):
            self.videos_infos = self.ann_file
        else:
            raise ValueError('ann_file must a list, not support other format!')

    def _sample_frames(self):
        t_vals = np.linspace(0., 1., num= self.sample_num)
        mids = 0.5 * ( t_vals[:-1] + t_vals[1:] )
        if self.mode == 'train':
            lowers = np.concatenate([t_vals[:1],mids],axis=-1)
            uppers = np.concatenate([mids,t_vals[-1:]],axis=-1)
            rand_vals = np.random.rand(self.sample_num)
            samples = lowers + (uppers - lowers) * rand_vals
            return samples
        t_vals = np.linspace(0., 0.999, num= self.sample_num)
        return mids

    def __len__(self):
        return len(self.videos_infos)

    def __getitem__(self,index):
        video_info = self.videos_infos[index]
        file_name = video_info['filename']
        prompt = video_info['prompt']
        quality = video_info['quality'] / 100
        authenticity = video_info['authenticity'] / 100
        correspondence = video_info['correspondence'] / 100
        
        vreader = decord.VideoReader(os.path.join(self.root,file_name))
        samples = np.floor(self._sample_frames() * len(vreader)).astype(np.uint8)
        imgs = [vreader[idx] for idx in samples]
        video = torch.stack(imgs,dim=0)
        video = rearrange(video,'t w h c -> c t w h')
        video = torch.nn.functional.interpolate(video,size=(self.size,self.size))
        video = ((video.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)

        data = {
            'video': video,
            'prompt': prompt,
            'authenticity': authenticity,
            'correspondence': correspondence,
            'quality': quality
        }
        return data

import random
def train_test_split(dataset_path, ann_file, ratio=0.8, seed=42):
    random.seed(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split("|")
            filename, prompt, quality, authenticity, correspondence = line_split
            quality = float(quality)
            authenticity = float(authenticity)
            correspondence = float(correspondence)
            # filename = os.path.join(dataset_path, filename)
            video_infos.append(dict(filename=filename,
                                    prompt=prompt,
                                    quality=quality,
                                    authenticity=authenticity,
                                    correspondence=correspondence))
    random.shuffle(video_infos)
    return (
        video_infos[: int(ratio * len(video_infos))],
        video_infos[int(ratio * len(video_infos)) :],
    )
if __name__ == "__main__":
    train_list, test_list = train_test_split(dataset_path='../data/',ann_file='../data/anno.txt')
    dataset = T23dqaDataset(
        root='/home/fk/data/T23DQA/', mode='test',ann_file=train_list,
        sample_num=12,size = 224
    )
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset)
    sample = iter(dataloader).__next__()
    print(sample['video'].shape)
    print(sample['prompt'])
    print(sample['quality'])
    print(sample['authenticity'])
    print(sample['correspondence'])
    # video = decord.VideoReader('/home/fk/data/T23DQA/videos/sjc/prompt_000.mp4')
    # print(type(video))
    # print(len(video))
    # print(type(video[0]))
    # print(video[0].shape)
    # frames_inds = np.array([0,12,43,56,72])
    # sample = [video[idx] for idx in frames_inds]
    # sample = torch.stack(sample,dim=0)
    # print(sample.shape)

