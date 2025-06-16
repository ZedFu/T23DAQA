import os
import argparse
import yaml

class ParserClass:
    '''
    This class is used for LSP ( code completion )!
    '''
    def __init__(self, opt):
        self.opt = opt

        # exp(logs) config
        self.exp_name = opt['exp']['name']
        self.exp_device = opt['exp']['device']
        
        # dataset config
        self.dataset_num_split = opt['dataset']['num_split']
        self.dataset_root = opt['dataset']['root']
        self.dataset_ann = opt['dataset']['ann']
        self.dataset_sample_num = opt['dataset']['sample_num']
        self.dataset_size = opt['dataset']['size']

        # train config
        self.train_epochs = opt['train']['epochs']
        self.train_bs = opt['train']['batch_size']

        # model config
        self.model_backbone = opt['model']['backbone']
        if 'vl_model' in opt['model'].keys():
            self.vl_model = opt['model']['vl_model']
        if 'sd_model' in opt['model'].keys():
            self.sd_model = opt['model']['sd_model']
        if 'td_model' in opt['model'].keys():
            self.td_model = opt['model']['td_model']

        # optimizer config
        self.op_lr = opt['optimizer']['lr']
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--con', type=str, default='./config/test.yml',help='the config file!')
    args = parser.parse_args()
    with open(args.con,"r") as f:
        opt = yaml.safe_load(f)
    print(opt)
    opc = ParserClass(opt)
    print(opc.exp_name)

