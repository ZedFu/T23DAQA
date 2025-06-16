import os
import torch
from torchvision import models as tmd
import transformers
import timm
import clip
# from .cnniqa import CNNIQAnet
# from .hyperiqa import Hyperiqa
# from .maniqa import MANIQA
# from .stairiqa import resnet50_imdt as stairiqa

class BenchMarkPT(torch.nn.Module):
    def __init__(self,backbone):
        super(BenchMarkPT,self).__init__()
        self.backbone = None
        if backbone == 'mc318':
            self.backbone = tmd.video.mc3_18(weights=tmd.video.MC3_18_Weights)
        elif backbone == 'r3d18':
            self.backbone = tmd.video.r3d_18(weights=tmd.video.R3D_18_Weights)
        elif backbone == 'r2p':
            self.backbone = tmd.video.r2plus1d_18(weights=tmd.video.R2Plus1D_18_Weights)
        elif backbone == 'swin3d-b':
            self.backbone = tmd.video.swin3d_b(weights=tmd.video.Swin3D_B_Weights)
        elif backbone == 'swin3d-s':
            self.backbone = tmd.video.swin3d_s(weights=tmd.video.Swin3D_S_Weights)
        elif backbone == 'swin3d-t':
            self.backbone = tmd.video.swin3d_t(weights=tmd.video.Swin3D_T_Weights)
        else:
            raise ValueError("Backbone not implement!")

        # for parm in self.backbone.parameters():
        #     parm.requires_grad = False
        self.linear = torch.nn.Linear(400,3)

    def forward(self,video):
        feature = self.backbone(video)
        out = self.linear(feature)
        return out[...,0], out[...,1], out[...,2]

class BenchMarkIQA(torch.nn.Module):
    def __init__(self,backbone):
        super(BenchMarkIQA, self).__init__()
        self.backbone = None
        if backbone == 'resnet18':
            self.backbone = timm.create_model('resnet18',pretrained=True,num_classes=3)
        elif backbone == 'resnet34':
            self.backbone = timm.create_model('resnet34',pretrained=True,num_classes=3)
        elif backbone == 'resnet50':
            self.backbone = timm.create_model('resnet50',pretrained=True,num_classes=3)
        elif backbone == 'vgg16':
            self.backbone = timm.create_model('vgg16',pretrained=True,num_classes=3)
        elif backbone == 'vgg19':
            self.backbone = timm.create_model('vgg19',pretrained=True,num_classes=3)
        elif backbone == 'swin-t':
            self.backbone = timm.create_model('swin_tiny_patch4_window7_224',pretrained=True,num_classes=3)
        elif backbone == 'swin-s':
            self.backbone = timm.create_model('swin_small_patch4_window7_224',pretrained=True,num_classes=3)
        elif backbone == 'swin-b':
            self.backbone = timm.create_model('swin_base_patch4_window7_224',pretrained=True,num_classes=3)
        elif backbone == 'swin-l':
            self.backbone = timm.create_model('swin_large_patch4_window7_224',pretrained=True,num_classes=3)
        elif backbone == 'cnniqa':
            self.backbone = CNNIQAnet()
        elif backbone == 'hyperiqa':
            self.backbone = Hyperiqa()
        elif backbone == 'maniqa':
            self.backbone = MANIQA(embed_dim=768,num_outputs=3)
        elif backbone == 'stairiqa':
            self.backbone = stairiqa()
        else:
            raise ValueError("Backbone not implement!")

    def forward(self,img):
        out = self.backbone(img)
        return out[...,0], out[...,1], out[...,2]


class T23DCQA(torch.nn.Module):
    def __init__(self,device,vl_model,sd_model,td_model):
        super(T23DCQA,self).__init__()
        self.device = device
        self.vl_model = vl_model
        self.sd_model = sd_model 
        self.td_model = td_model
        self.input_size = 0
        if self.vl_model != 'none':
            self.Vision_Language_model, _ = clip.load("ViT-B/32",device=self.device) 
            for param in self.Vision_Language_model.parameters():
                param.requires_grad = False
            self.linear1 = torch.nn.Sequential(
                torch.nn.Linear(512*2,512),
                torch.nn.ReLU()
            )
            self.input_size += 512
            self.linear1.to(device)
        if self.sd_model != 'none':
            self.img_encoder_1 = tmd.swin_s(tmd.Swin_S_Weights)
            self.img_encoder_2 = tmd.swin_s(tmd.Swin_S_Weights)
            self.linear2 = torch.nn.Sequential(
                torch.nn.Linear(1000*2,512),
                torch.nn.ReLU()
            )
            self.input_size += 512
            self.img_encoder_1.to(device)
            self.img_encoder_2.to(device)
            self.linear2.to(device)
        if self.td_model != 'none':
            self.video_encoder = tmd.video.swin3d_s(tmd.video.Swin3D_S_Weights)
            self.input_size += 400
            self.video_encoder.to(device)
        self.fusion_module_1 = torch.nn.Sequential(
            torch.nn.Linear(self.input_size,1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,1),
        )
        self.fusion_module_2 = torch.nn.Sequential(
            torch.nn.Linear(self.input_size,1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,1),
        )
        self.fusion_module_3 = torch.nn.Sequential(
            torch.nn.Linear(self.input_size,1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,1),
        )
        self.fusion_module_1.to(device)
        self.fusion_module_2.to(device)
        self.fusion_module_3.to(device)

    def forward(self,prompts, video):
        # image = self.preprocess(video[:,:,0,:,:])
        image = video[:,:,0,:,:]
        image_2 = video[:,:,6,:,:]
        feature_list = []
        if self.vl_model != 'none':
            token_p = clip.tokenize(prompts).to(self.device)
            img_feature = self.Vision_Language_model.encode_image(image)
            text_features = self.Vision_Language_model.encode_text(token_p)
            l1 = self.linear1(torch.cat([img_feature.float(), text_features.float()],dim=-1))
            feature_list.append(l1)
        if self.sd_model != 'none':
            s_f, s_f2 = self.img_encoder_1(image), self.img_encoder_2(image_2)
            l2 = self.linear2(torch.cat([s_f,s_f2],dim=-1))
            feature_list.append(l2)
        if self.td_model != 'none':
            t_f = self.video_encoder(video)
            feature_list.append(t_f)
        out_1 = self.fusion_module_1(torch.cat(feature_list,dim=-1))
        out_2 = self.fusion_module_2(torch.cat(feature_list,dim=-1))
        out_3 = self.fusion_module_3(torch.cat(feature_list,dim=-1))
        return out_1[...,0], out_2[...,0], out_3[...,0]
    
class T23DCQAV(torch.nn.Module):
    def __init__(self,device):
        super(T23DCQAV,self).__init__()
        self.device = device
        self.input_size = 0
        self.Vision_Language_model, _ = clip.load("ViT-B/32",device=self.device) 
        for param in self.Vision_Language_model.parameters():
            param.requires_grad = False
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(512*2,512),
            torch.nn.ReLU()
        )
        self.input_size += 512
        self.linear1.to(device)
        self.img_encoder_1 = tmd.swin_s(tmd.Swin_S_Weights)
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(1000,512),
            torch.nn.ReLU()
        )
        self.input_size += 512
        self.img_encoder_1.to(device)
        self.linear2.to(device)
        self.fusion_module_1 = torch.nn.Sequential(
            torch.nn.Linear(self.input_size,1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,1),
        )
        self.fusion_module_2 = torch.nn.Sequential(
            torch.nn.Linear(self.input_size,1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,1),
        )
        self.fusion_module_3 = torch.nn.Sequential(
            torch.nn.Linear(self.input_size,1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,1),
        )
        self.fusion_module_1.to(device)
        self.fusion_module_2.to(device)
        self.fusion_module_3.to(device)

    def forward(self,prompts, image):
        # image = self.preprocess(video[:,:,0,:,:])
        image = torch.nn.functional.interpolate(image,size=(224,224),mode='bilinear')
        feature_list = []
        token_p = clip.tokenize(prompts).to(self.device)
        img_feature = self.Vision_Language_model.encode_image(image)
        text_features = self.Vision_Language_model.encode_text(token_p)
        l1 = self.linear1(torch.cat([img_feature.float(), text_features.float()],dim=-1))
        feature_list.append(l1)
        s_f = self.img_encoder_1(image)
        l2 = self.linear2(s_f)
        feature_list.append(l2)
        out_1 = self.fusion_module_1(torch.cat(feature_list,dim=-1))
        out_2 = self.fusion_module_2(torch.cat(feature_list,dim=-1))
        out_3 = self.fusion_module_3(torch.cat(feature_list,dim=-1))
        return out_1[...,0], out_2[...,0], out_3[...,0]
    
if __name__ == "__main__":
    ts = torch.rand((2,3,224,224))
    prompt = ['a tennis court that is very wet from lots of rain', 'hugging face hub']
    print(ts.shape)
    model_state_dict = torch.load('/mnt/sda/fk/project/tmp/t23dqa/logs/train_my_modelv/test.ckpt')
    model = T23DCQAV('cuda').to('cuda')
    model.load_state_dict(model_state_dict)
    qua, aut, cor = model(prompt,ts.to('cuda'))
