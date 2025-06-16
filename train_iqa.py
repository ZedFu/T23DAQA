import os 
import argparse 
import yaml
import random
import numpy as np 
import torch

from torch.utils.data import DataLoader
from model import BenchMarkIQA
from datasets import T23dqaDataset
from parse import ParserClass

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from scipy.stats import spearmanr, kendalltau, pearsonr
from scipy.optimize import curve_fit

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
        self.model = BenchMarkIQA(opc.model_backbone)

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

    def training_step(self,batch, batch_idx):
        video, prompt, authenticity, correspondence, quality = self.decode_batch(batch)
        num_img = video.shape[2]
        a_l,c_l,q_l = [], [], []
        for i in range(num_img):
            img = video[:,:,i,:,:]
            a,c,q = self.model(img)
            a_l.append(a)
            c_l.append(c)
            q_l.append(q)
        a = torch.mean(torch.stack(a_l,dim=-1),dim=-1)
        c = torch.mean(torch.stack(c_l,dim=-1),dim=-1)
        q = torch.mean(torch.stack(q_l,dim=-1),dim=-1)
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
        a_l,c_l,q_l = [], [], []
        for i in range(video.shape[2]):
            img = video[:,:,i,:,:]
            a,c,q = self.model(img)
            a_l.append(a)
            c_l.append(c)
            q_l.append(q)
        a = torch.mean(torch.stack(a_l,dim=-1),dim=-1)
        c = torch.mean(torch.stack(c_l,dim=-1),dim=-1)
        q = torch.mean(torch.stack(q_l,dim=-1),dim=-1)
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
    parser.add_argument('-c', '--con', type=str, default='./config/test_iqa.yml',help='the config file!')
    args = parser.parse_args()
    with open(args.con,"r") as f:
        opt = yaml.safe_load(f)

    opc = ParserClass(opt)

    for i in range(opc.dataset_num_split):
        checkpoint_callback = ModelCheckpoint(os.path.join('./logs',opc.exp_name),filename='{epoch:02d}',save_last=False,save_top_k=1)
        logger = TensorBoardLogger(save_dir='./logs',name=opc.exp_name)

        system = TSystem(opc)
        trainer = L.Trainer(
            # callbacks=[checkpoint_callback],
            logger=logger,
            max_epochs=opc.train_epochs,
            devices=opc.exp_device
        )
        train_list, test_list = train_test_split(opc.dataset_ann,ratio = 0.8)
        train_dataset = T23dqaDataset(opc.dataset_root,mode='train',ann_file=train_list,sample_num=opc.dataset_sample_num,size=opc.dataset_size)
        test_dataset = T23dqaDataset(opc.dataset_root,mode='test',ann_file=test_list,sample_num=opc.dataset_sample_num,size=opc.dataset_size)
    
        trainer.fit(system,train_dataloaders=DataLoader(train_dataset,batch_size=opc.train_bs,num_workers=8))
        trainer.test(system,dataloaders=DataLoader(test_dataset,batch_size=opc.train_bs,num_workers=8))


if __name__ == "__main__":
    main()

