import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm

from loss import *
from model.biasModel import biasModel
import pickle

import warnings
warnings.filterwarnings("ignore")

device="cuda:3"    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        self.fc_comb = nn.Sequential(
            nn.Linear(channel, channel // 3, bias=False),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        b, t, c = x.size()
        x = x.transpose(0, 2) # B, T<, C -> C, T, B
        y = self.avg_pool(x).view(c)

        y = self.fc(y).view(c, 1, 1)
        x = x * y
        x = x.transpose(2, 0) # Reshape to original

        return x

class bias(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, **kwargs):
        super(bias, self).__init__()        

        self.model = biasModel().to(device)
        self.se = SELayer(channel=128*3).to(device)
        self.lossA = lossA().to(device)
        self.lossV = lossV().to(device)
        self.lossVB = lossB().to(device)
        self.lossFBA = lossFBA().to(device)
        self.lossComb = lossComb().to(device)

        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']        
        for num, (audioFeature, visualFeature, visualFeatureBody, labels) in enumerate(loader, start=1):
            self.zero_grad()
            audioEmbed = self.model.forward_audio_frontend(audioFeature[0].to(device)) # feedForward
            visualEmbed = self.model.forward_visual_frontend(visualFeature[0].to(device))
            visualEmbedBody = self.model.forward_visual_frontend_body(visualFeatureBody[0].to(device))

            # Self-Attention
            audioEmbed = self.model.a_att(src = audioEmbed, tar = audioEmbed)
            visualEmbed = self.model.v_att(src = visualEmbed, tar = visualEmbed)
            visualEmbedBody = self.model.vb_att(src = visualEmbedBody, tar = visualEmbedBody)
            # Feature combination
            comb_feat = torch.cat((audioEmbed, visualEmbed, visualEmbedBody), dim=2).to(device)
            outsComb = self.se(comb_feat)
            outsComb = self.model.comb_att(src = outsComb, tar = outsComb)

            outsComb = self.model.forward_comb_backend(outsComb)
            outsA = self.model.forward_audio_backend(audioEmbed)
            outsV = self.model.forward_visual_backend(visualEmbed)
            outsVB = self.model.forward_visual_backend(visualEmbedBody)

            labels = labels[0].reshape((-1)).to(device) # Loss
            nlossAV, _, _, prec = self.lossComb.forward(outsComb, labels)
            nlossA = self.lossA.forward(outsA, labels)
            nlossV = self.lossV.forward(outsV, labels)
            nlossVB = self.lossVB.forward(outsVB, labels)
            
            nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV + 0.4 * nlossVB
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  

        sys.stdout.write("\n")      
        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        print(evalCsvSave, evalOrig)
        predScores = []
        for audioFeature, visualFeature, visualFeatureBody, labels in tqdm.tqdm(loader):
            with torch.no_grad():               
                audioEmbed = self.model.forward_audio_frontend(audioFeature[0].to(device)) # feedForward
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].to(device))
                visualEmbedBody = self.model.forward_visual_frontend_body(visualFeatureBody[0].to(device))

                # Self-Attention
                audioEmbed = self.model.a_att(src = audioEmbed, tar = audioEmbed)
                visualEmbed = self.model.v_att(src = visualEmbed, tar = visualEmbed)
                visualEmbedBody = self.model.vb_att(src = visualEmbedBody, tar = visualEmbedBody)
                # Feature combination
                comb_feat = torch.cat((audioEmbed, visualEmbed, visualEmbedBody), dim=2).to(device)
                outsComb = self.se(comb_feat)
                outsComb = self.model.comb_att(src = outsComb, tar = outsComb)

                outsComb = self.model.forward_comb_backend(outsComb)

                labels = labels[0].reshape((-1)).to(device) # Loss
                _, predScore, _, _ = self.lossComb.forward(outsComb, labels)
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
                

        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        print(str(subprocess.run(cmd, shell=True, capture_output =True).stdout), evalOrig,evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, capture_output =True).stdout).split(' ')[2][:5])
        return mAP


    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
