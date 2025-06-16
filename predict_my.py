import os 
from glob import glob
import argparse 
import yaml
import random
import numpy as np 
import torch

from torch.utils.data import DataLoader
from model import T23DCQA
from datasets import T23dqaDataset
from parse import ParserClass

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from scipy.stats import spearmanr, kendalltau, pearsonr
from scipy.optimize import curve_fit
import decord
from einops import rearrange
import pandas as pd
decord.bridge.set_bridge('torch')

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic


def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

class TSystem(L.LightningModule):
    def __init__(self, opc : ParserClass):
        super(TSystem, self).__init__()
        self.opc = opc
        self.model = T23DCQA(opc.model_backbone, opc.vl_model, opc.sd_model, opc.td_model)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opc.op_lr)
        return [optimizer]

    def decode_batch(self,batch):
        video = batch['video']
        prompt = batch['prompt']
        authenticity = batch['authenticity'].float()
        correspondence = batch['correspondence'].float()
        quality = batch['quality'].float()
        return video, prompt, authenticity, correspondence, quality

    def loss(self, y_pred, y):
        p_loss, r_loss = plcc_loss(y_pred, y), rank_loss(y_pred, y)
        loss = p_loss + 0.3 * r_loss
        return loss

    def forward(self,prompt,video):
        return self.model(video,prompt)

    def training_step(self,batch, batch_idx):
        video, prompt, authenticity, correspondence, quality = self.decode_batch(batch)
        a,c,q = self.model(prompt, video)
        a_loss = self.loss(a,authenticity)
        c_loss = self.loss(c,correspondence)
        q_loss = self.loss(q, quality)
        loss = (a_loss + c_loss + q_loss) / 3
        self.log("train/a_loss",a_loss)
        self.log("train/c_loss",c_loss)
        self.log("train/q_loss",q_loss)
        self.log("train/loss",loss)
        return loss

    def on_test_start(self):
        self.a_p, self.a_l = [], []
        self.c_p, self.c_l = [], []
        self.q_p, self.q_l = [], []
        
    def test_step(self,batch, batch_idx):
        video, prompt, authenticity, correspondence, quality = self.decode_batch(batch)
        a,c,q = self.model(prompt, video)
        self.a_p.append(a)
        self.a_l.append(authenticity)
        self.c_p.append(c)
        self.c_l.append(correspondence)
        self.q_p.append(q)
        self.q_l.append(quality)

    def test_metric(self,y_pred, y):
        y_pred, y = y_pred.cpu().numpy(), y.cpu().numpy()
        y_pred = fit_function(y, y_pred)
        # print(y_pred.shape, y.shape)
        s = spearmanr(y_pred, y)[0]
        k = kendalltau(y_pred, y)[0]
        p = pearsonr(y_pred, y)[0]
        return s, k ,p

    def on_test_end(self):
        a_p = torch.cat(self.a_p,dim=0)
        a_l = torch.cat(self.a_l,dim=0)
        c_p = torch.cat(self.c_p,dim=0)
        c_l = torch.cat(self.c_l,dim=0)
        q_p = torch.cat(self.q_p,dim=0)
        q_l = torch.cat(self.q_l,dim=0)
        a_srcc, a_krcc, a_plcc = self.test_metric(a_p, a_l)
        c_srcc, c_krcc, c_plcc = self.test_metric(c_p, c_l)
        q_srcc, q_krcc, q_plcc = self.test_metric(q_p, q_l)
        a_str = 'authenticity: SRCC: {}, krcc: {}, PLCC: {}'.format(a_srcc,a_krcc,a_plcc)
        c_str = 'correspondence: SRCC: {}, krcc: {}, PLCC: {}'.format(c_srcc,c_krcc,c_plcc)
        q_str = 'quality: SRCC: {}, krcc: {}, PLCC: {}'.format(q_srcc,q_krcc,q_plcc)
        with open(os.path.join('./logs/',self.opc.exp_name,'test_result.txt'),'a') as f:
            f.write(a_str + '\n' + c_str + '\n' + q_str)
            f.write('\n')
    def predict_step(self,batch,batch_idx):
        return 0

def train_test_split(ann_file, ratio=0.8, seed=42):
    random.seed(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split("|")
            filename, prompt, quality, authenticity, correspondence = line_split
            quality = float(quality)
            authenticity = float(authenticity)
            correspondence = float(correspondence)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--con', type=str, default='./config/test_my.yml',help='the config file!')
    args = parser.parse_args()
    with open(args.con,"r") as f:
        opt = yaml.safe_load(f)

    opc = ParserClass(opt)
    system = TSystem.load_from_checkpoint('./logs/train_my_model/epoch=19.ckpt',opc = opc)

    input_dir = '../test_video_dir/newtrain_p/'
    df = pd.read_csv(os.path.join(input_dir,'prompt.csv'))
    prompts_list = df['prompt']
    mean = torch.FloatTensor([123.675, 116.28, 103.53])
    std = torch.FloatTensor([58.395, 57.12, 57.375])
    prompt = 'A cute golden dog'
    video_list = sorted(glob(os.path.join(input_dir,'*.mp4')))
    for idx,mp4path in enumerate(video_list):
        vreader = decord.VideoReader(mp4path)
        mids = np.linspace(0., 0.999, num=12)
        samples = np.floor(mids* len(vreader)).astype(np.uint8)
        imgs = [vreader[idx] for idx in samples]
        video = torch.stack(imgs,dim=0)
        video = rearrange(video,'t w h c -> c t w h')
        video = torch.nn.functional.interpolate(video,size=(224,224))
        video = ((video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
        video = video.unsqueeze(0).to('cuda:0')
        print(idx,mp4path)
        name = mp4path.split('/')[-1].split('.')[0]
        print(name)
        prompt = prompts_list[int(name)]
        print(prompt)
        a,c,q = system(video,prompt)
        print('authenticity: {:.3f}'.format(a.detach().cpu().numpy()[0]*100))
        print('correspondence: {:.3f}'.format(c.detach().cpu().numpy()[0]*100))
        print('quality: {:.3f}'.format(q.detach().cpu().numpy()[0]*100))



if __name__ == "__main__":
    main()

