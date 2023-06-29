import torch.nn as nn
import torch.nn.functional as F
import torch
'''
Thanks to the following link that helped me code the transformers, 
parts of this code was written using their code.
https://github.com/jadore801120/attention-is-all-you-need-pytorch
'''
class MultiHeadAttention(nn.Module):

    def __init__(self, args, block):
        super().__init__()
        self.batch_size = args.batch_size
        self.enc_steps = args.enc_steps
        self.dropout = args.dropout
        self.n_head = args.n_head
        self.d_k = args.d_k
        self.d_v = args.d_v
        self.d_inner = args.d_inner
        self.d_model_sp = args.d_model_sp
        self.d_model_traj = args.d_model_traj
        self.temperature = self.d_k ** 0.5

        if block=='enc':
            self.w_qs = nn.Linear(self.d_model_sp+self.d_model_traj, self.n_head * self.d_k, bias=False)
            self.w_ks = nn.Linear(self.d_model_sp+self.d_model_traj, self.n_head * self.d_k, bias=False)
            self.w_vs = nn.Linear(self.d_model_sp+self.d_model_traj, self.n_head * self.d_v, bias=False)
            self.fc = nn.Linear(self.n_head * self.d_v, self.d_model_sp+self.d_model_traj, bias=False)
            self.layer_norm = nn.LayerNorm(self.d_model_sp+self.d_model_traj, eps=1e-6)

        if block=='enc_dec_traj':
            self.w_qs = nn.Linear(self.d_model_traj, self.n_head * self.d_k, bias=False)
            self.w_ks = nn.Linear(self.d_model_sp+self.d_model_traj, self.n_head * self.d_k, bias=False)
            self.w_vs = nn.Linear(self.d_model_sp+self.d_model_traj, self.n_head * self.d_v, bias=False)
            self.fc = nn.Linear(self.n_head * self.d_v, self.d_model_traj, bias=False)
            self.layer_norm = nn.LayerNorm(self.d_model_traj, eps=1e-6)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        


    def forward(self, q, k, v, q_steps, k_steps, mask=None):
        
        residual = q

        q = self.w_qs(q).view(q.shape[0], q_steps, self.n_head, self.d_k)
        k = self.w_ks(k).view(q.shape[0], k_steps, self.n_head, self.d_k)
        v = self.w_vs(v).view(q.shape[0], k_steps, self.n_head, self.d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn += mask
       
        attn = self.dropout1(F.softmax(attn, dim=-1))
        
        q = torch.matmul(attn, v)
        q = q.transpose(1, 2).contiguous().view(q.shape[0], q_steps, -1)
        q = self.dropout2(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q


class FeedForward(nn.Module):

    def __init__(self, args, block):
        super().__init__()
        self.d_inner = args.d_inner
        self.d_model_sp = args.d_model_sp
        self.d_model_traj = args.d_model_traj
        self.dropout = args.dropout
        if block=='enc':
            self.w_1 = nn.Linear(self.d_model_sp+self.d_model_traj, self.d_inner) # position-wise
            self.w_2 = nn.Linear(self.d_inner, self.d_model_sp+self.d_model_traj) # position-wise
            self.w_3 = nn.Linear(self.d_model_sp+self.d_model_traj, self.d_model_sp+self.d_model_traj)
            self.layer_norm = nn.LayerNorm(self.d_model_sp+self.d_model_traj, eps=1e-6)
        elif block=='dec_traj':
            self.w_1 = nn.Linear(self.d_model_traj, self.d_inner) # position-wise
            self.w_2 = nn.Linear(self.d_inner, self.d_model_traj) # position-wise
            self.w_3 = nn.Linear(self.d_model_traj, self.d_model_traj)
            self.layer_norm = nn.LayerNorm(self.d_model_traj, eps=1e-6)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        x = self.w_3(x)
        return x
