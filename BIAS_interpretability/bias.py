import torch
import torch.nn as nn

import sys, time, os, subprocess, pandas, tqdm

from loss import *
from model.biasModel import biasModel
import pickle

import warnings
warnings.filterwarnings("ignore")

device="cuda:0"    

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

    def only_consider_top_chanels(self, vetor_SE_combination, kwargs):  

        def replace_vetor(vetor_se, Z):
 
            mean, std = torch.mean(vetor_se), torch.std(vetor_se)
            threshold = mean + Z*std
            vetor_se = torch.where(vetor_se > threshold, vetor_se, torch.tensor(0.).to(device))
            return vetor_se

        # Alternative to get top 10%, X = u + Zo, u = mean and o = std
        # For percentil:
        # 75%, Z = 0.674
        # 90%, Z = 1.282
        # 95%, Z = 1.645
        Z = 1.282
        change_audio = kwargs["changeAudio"]
        change_face = kwargs["changeFace"]
        change_body = kwargs["changeBody"]

        vetor_SE_combination = vetor_SE_combination.squeeze()
        audio_se, face_se, body_se = torch.split(vetor_SE_combination, 128) 

        if change_audio:
            audio_se = replace_vetor(audio_se, Z)
        if change_face:
            face_se = replace_vetor(face_se, Z)
        if change_body:
            body_se = replace_vetor(body_se, Z)

        vetor_SE_combination_remade = torch.cat((audio_se, face_se, body_se))
        vetor_SE_combination_remade = vetor_SE_combination_remade[:,None,None]

        return vetor_SE_combination_remade

    def forward(self, x, kwargs):
        b, t, c = x.size()
        # Squeeze over 128 dmodel
        x = x.transpose(0, 2) # B, T, C -> C, T, B
        y = self.avg_pool(x).view(c)

        y = self.fc(y).view(c, 1, 1)
        y = self.only_consider_top_chanels(y, kwargs)

        x = x * y
        x = x.transpose(2, 0) # Reshape to original
        return x, y


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

    # To create dictionary of features
    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        print(evalCsvSave, evalOrig)
        dict_vetor = {}
        predScores = []

        for audioFeature, visualFeature, visualFeatureBody, labels, (face_path, bodypath) in tqdm.tqdm(loader):
            with torch.no_grad():               
                audioEmbed = self.model.forward_audio_frontend(audioFeature[0].to(device)) # feedForward
                visualEmbed, face_vetor, _ = self.model.forward_visual_frontend(visualFeature[0].to(device))
                visualEmbedBody, body_vetor, _ = self.model.forward_visual_frontend_body(visualFeatureBody[0].to(device))
                # Self-Attention
                audioEmbed = self.model.a_att(src = audioEmbed, tar = audioEmbed)
                visualEmbed = self.model.v_att(src = visualEmbed, tar = visualEmbed)
                visualEmbedBody = self.model.vb_att(src = visualEmbedBody, tar = visualEmbedBody)
                # Feature combination
                comb_feat = torch.cat((audioEmbed, visualEmbed, visualEmbedBody), dim=2).to(device)
                outsComb, vetor = self.se(comb_feat, kwargs)
                outsComb = self.model.comb_att(src = outsComb, tar = outsComb)

                outsComb = self.model.forward_comb_backend(outsComb)

                labels = labels[0].reshape((-1)).to(device) # Loss
                _, predScore, _, _ = self.lossComb.forward(outsComb, labels)
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)

                # ------- Feature importance  --------------
                vetor = vetor.squeeze()
                a, f, b = torch.split(torch.tensor(vetor), 128, dim=0)
                a, f, b = a.tolist(), f.tolist(), b.tolist()
                
                # vidname_person = face_path[0].split("/")[-1] # last one is the frame timestamp
                vidname_person = face_path[0][0].split("/")[-2] # last one is the timestamp name, so we want the previous

                if vidname_person in dict_vetor:
                    print(vidname_person)
                    assert False

                # ------- Face and Body importance  --------------
                body_vetor = torch.squeeze(body_vetor).tolist()
                face_vetor = torch.squeeze(face_vetor).tolist()

                dict_vetor[vidname_person] = {
                    "audio" : a,
                    "face"  : f,
                    "body"  : b, 
                    "body_backbone": body_vetor,
                    "face_backbone": face_vetor
                }

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

        os.makedirs("vetor_dictionary", exist_ok=True)
        dict_Name = "vetor_dictionary/{}.pkl".format(((kwargs["inputModel"].split("/"))[-1]).split(".model")[0])
        print("Saving info in dictionary:", dict_Name)

        # Save dictionary as a pikle
        with open(dict_Name, 'wb') as handle:
            pickle.dump(dict_vetor, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Dictionary saved!")

        return mAP

    def evaluate_network_se(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        print(evalCsvSave, evalOrig)
        predScores = []
        for audioFeature, visualFeature, visualFeatureBody, labels, (face_path, body_path) in tqdm.tqdm(loader):
            with torch.no_grad():               
                audioEmbed = self.model.forward_audio_frontend(audioFeature[0].to(device)) # feedForward
                visualEmbed, se_face_vetor, se_face_feat = self.model.forward_visual_frontend(visualFeature[0].to(device))
                visualEmbedBody, se_body_vetor, se_body_feat = self.model.forward_visual_frontend_body(visualFeatureBody[0].to(device))
                break
        
        return se_face_feat, se_face_vetor, face_path, se_body_feat, se_body_vetor, body_path

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
