
import torch
import torch.nn as nn
from .feature_extractor import build_feature_extractor
from .SubLayers import MultiHeadAttention, FeedForward

class Trajnet(nn.Module):
    def __init__(self, args, device):
        super(Trajnet, self).__init__()
        self.device = device
        self.batch_size = args.batch_size
        self.hidden_size_traj = args.hidden_size_traj
        self.hidden_size_sp = args.hidden_size_sp
        self.loc_dim = args.loc_dim
        self.sp_dim = args.sp_dim
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.dropout = args.dropout
        self.n_head = args.n_head
        self.d_k = args.d_k
        self.d_v = args.d_v
        self.d_inner = args.d_inner
        self.d_model_traj = args.d_model_traj
        self.feature_extractor_traj = build_feature_extractor(args, ft_type='traj')
        self.feature_extractor_speed = build_feature_extractor(args, ft_type='speed')
        self.pos_em_enc_traj = self.pos_embed(self.enc_steps, self.hidden_size_traj)
        self.pos_em_enc_sp = self.pos_embed(self.enc_steps, self.hidden_size_sp)
        self.pos_em_dec_traj = self.pos_embed(self.dec_steps, self.hidden_size_traj)
        self.EncoderLayer = EncoderLayer(args)
        self.DecoderLayer_traj = DecoderLayer(args, self.device, block='traj')
        self.regressor_traj = nn.Sequential(nn.Linear(self.hidden_size_traj
                                       ,self.loc_dim))

        
    def pos_embed(self, length, hidden_size, n=10000):
        P = torch.zeros((length, hidden_size), device=self.device)
        for k in range(length):
            for i in torch.arange(int(hidden_size/2)):
                denominator = torch.pow(n, 2*i/hidden_size)
                P[k, 2*i] = torch.sin(k/denominator)
                P[k, 2*i+1] = torch.cos(k/denominator)
        return P

    def forward(self, inputs, targets = 0, start_index = 0, training=True, mask=None, loop=None):
        [traj,speed] = inputs
        traj_input = self.feature_extractor_traj(traj) + self.pos_em_enc_traj
        speed_input = self.feature_extractor_speed(speed) + self.pos_em_enc_sp
        traj_speed = torch.cat((traj_input, speed_input), axis=-1)
        encoded = self.EncoderLayer(traj_speed)
        dec_input = torch.zeros((encoded.shape[0], self.dec_steps, self.hidden_size_traj), device=self.device)
        dec_input += self.pos_em_dec_traj
        decoded_traj = self.DecoderLayer_traj(dec_input, encoded, mask=mask)
        dec_input_traj = self.regressor_traj(decoded_traj)
        return dec_input_traj


class EncoderLayer(nn.Module):

    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.att_enc_in = MultiHeadAttention(args, block='enc')
        self.feedforward_enc = FeedForward(args, block='enc')
        

    def forward(self, traj_input):
        traj_att = self.att_enc_in(traj_input, traj_input, traj_input, self.enc_steps, self.enc_steps)
        traj_enc = self.feedforward_enc(traj_att)    

        return traj_enc

class DecoderLayer(nn.Module):

    def __init__(self, args, device, block):
        super(DecoderLayer, self).__init__()
        self.device = device
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.hidden_size_traj = args.hidden_size_traj

        if block == 'traj':
            self.att_enc_out = MultiHeadAttention(args, block='enc_dec_traj')
            self.feedforward_dec = FeedForward(args, block='dec_traj')
        

    def forward(self, dec_in, enc_out, mask=None):

        enc_dec_att = self.att_enc_out(dec_in, enc_out, enc_out, self.dec_steps, self.enc_steps)
        dec_out_all = self.feedforward_dec(enc_dec_att)
        
        return dec_out_all  

