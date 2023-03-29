import torch
import torch.nn as nn

from model.audioEncoder      import audioEncoder
from model.visualEncoder     import visualFrontend, visualTCN, visualConv1D
from model.attentionLayer    import *

class biasModel(nn.Module):
    def __init__(self):
        super(biasModel, self).__init__()
        # Visual Temporal Encoder
        self.visualFrontend  = visualFrontend() # Visual Frontend 
        self.visualFrontendBody  = visualFrontend() # Visual Frontend 

        self.visualTCN       = visualTCN()      # Visual Temporal Network TCN
        self.visualTCNBody   = visualTCN()      # Visual Temporal Network TCN
        self.visualConv1D    = visualConv1D()   # Visual Temporal Network Conv1d
        self.visualConv1DBody = visualConv1D()   # Visual Temporal Network Conv1d

        # Audio Temporal Encoder 
        self.audioEncoder  = audioEncoder(layers = [3, 4, 6, 3],  num_filters = [16, 32, 64, 128])
        
        # Audio-visual Cross Attention
        self.crossA2V = attentionLayer(d_model = 128, nhead = 8)
        self.crossV2A = attentionLayer(d_model = 128, nhead = 8)

        # Audio-visual Self Attention
        self.selfAV = attentionLayer(d_model = 256, nhead = 8)
        self.selfABV = attentionLayer(d_model = 256, nhead = 8)

        # Test of se_complex
        self.selfComb = attentionLayer(d_model = 128, nhead = 8)

        # Self Attention
        self.a_att = attentionLayer(d_model = 128, nhead = 8)
        self.v_att = attentionLayer(d_model = 128, nhead = 8)
        self.vb_att = attentionLayer(d_model = 128, nhead = 8)

        self.comb_att = attentionLayer(d_model = 128*3, nhead = 8)
        self.comb_att_2 = attentionLayer(d_model = 128*2, nhead = 8)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape  
        x = x.view(B*T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)      
        x = x.transpose(1,2)     
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)
        return x

    def forward_visual_frontend_body(self, x):
        B, T, W, H = x.shape  
        x = x.view(B*T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontendBody(x)
        x = x.view(B, T, 512)        
        x = x.transpose(1,2)     
        x = self.visualTCNBody(x)
        x = self.visualConv1DBody(x)
        x = x.transpose(1,2)
        return x

    def forward_audio_frontend(self, x):    
        x = x.unsqueeze(1).transpose(2, 3)        
        x = self.audioEncoder(x)
        return x

    def forward_cross_attention(self, x1, x2):
        x1_c = self.crossA2V(src = x1, tar = x2)
        x2_c = self.crossV2A(src = x2, tar = x1)        
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2): 
        x = torch.cat((x1,x2), 2)    
        x = self.selfAV(src = x, tar = x)      
        x = torch.reshape(x, (-1, 256))
        return x    

    def forward_audio_body_visual_backend(self, x1, x2): 
        x = torch.cat((x1,x2), 2)    
        x = self.selfABV(src = x, tar = x)      
        x = torch.reshape(x, (-1, 256))
        return x 

    def forward_audio_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_visual_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_body_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_comb_backend(self,x):
        x = torch.reshape(x, (-1, 128*3))
        return x


