import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiAttention(nn.Module):
    def __init__(self,hidden_size,num_head,causality=False):
        super(MultiAttention, self).__init__()
        self.fc_q = nn.Linear(hidden_size,hidden_size)
        self.fc_k = nn.Linear(hidden_size,hidden_size)
        self.fc_v = nn.Linear(hidden_size,hidden_size)
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.normalize = nn.LayerNorm([hidden_size])
        self.causality = causality
        
    def forward(self,queries,queries_mask,keys,keys_mask):
        '''
        queries: (bs,T,emb_size)
        query_mask: (bs,T,1)
        queries: (bs,T,emb_size)
        query_mask: (bs,T,1)
        '''
        Q_ = F.relu(self.fc_q(queries))
        K_ = F.relu(self.fc_k(keys))
        V_ = F.relu(self.fc_v(keys))
        
        Q = torch.cat(Q_.chunk(self.num_head,2),0)
        K = torch.cat(K_.chunk(self.num_head,2),0)
        V = torch.cat(V_.chunk(self.num_head,2),0)
        
        output = torch.bmm(Q,K.transpose(1,2))
        output = output / ( self.hidden_size**0.5)
        
        queries_mask_ = torch.cat([queries_mask]*self.num_head,0).float()
        keys_mask_ = torch.cat([keys_mask]*self.num_head,0).float()
        output_mask = 1 - queries_mask_.bmm(keys_mask_.transpose(1,2)).float()
        output_ = output.masked_fill(output_mask.byte() , -2**32)
        
        if self.causality:
            bs,s1,s2 = output_mask.size()
            tri = np.triu(np.ones((s1,s2))) - np.eye(s1,s2)
            tri = torch.from_numpy(tri).to(output_mask.device)
            tri = torch.stack([tri]*bs,0).byte()
            output_ = output_.masked_fill(tri , -2**32)
            
        output_ = F.softmax(output_,2)
        
        output_ = torch.cat(output_.bmm(V).chunk(self.num_head,0),2)
        output_ = output_ * queries_mask.float()
        return self.normalize(output_ + queries)

class EncoderBlock(nn.Module):
    def __init__(self,
                 hidden_size=512,
                 num_head=8,
                 fc_size=[2048,512]):
        super(EncoderBlock, self).__init__()
        self.attention = MultiAttention(hidden_size,num_head)
        self.fc = nn.Sequential(nn.Linear(fc_size[1],fc_size[0]),
                                nn.ReLU(),
                                nn.Linear(fc_size[0],fc_size[1]))
        self.normalize = nn.LayerNorm([fc_size[1]])
        
    def forward(self,queries,queries_mask,keys,keys_mask):
        atted = self.attention(queries,queries_mask,queries,queries_mask)
        output = self.fc(atted)
        return self.normalize(output + queries)

class DecoderBlock(nn.Module):
    def __init__(self,
                 hidden_size=512,
                 num_head=8,
                 fc_size=[2048,512]):
        super(DecoderBlock, self).__init__()
        self.sf_attention = MultiAttention(hidden_size,num_head,causality=True)
        self.attention = MultiAttention(hidden_size,num_head)
        self.fc = nn.Sequential(nn.Linear(fc_size[1],fc_size[0]),
                                nn.ReLU(),
                                nn.Linear(fc_size[0],fc_size[1]))
        self.normalize = nn.LayerNorm([fc_size[1]])
        
    def forward(self,queries,queries_mask,keys,keys_mask):
        queries_atted = self.sf_attention(queries,queries_mask,queries,queries_mask)
        atted = self.attention(queries_atted,queries_mask,keys,keys_mask)
        output = self.fc(atted)
        return self.normalize(output + queries)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.ModuleList([EncoderBlock() for i in range(6)])
        
    def forward(self,enc,enc_mask):
        for layer in self.model:
            enc = layer(queries=enc,
                        queries_mask=enc_mask,
                        keys=None,
                        keys_mask=None)
        return enc

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.ModuleList([DecoderBlock() for i in range(6)])
        
    def forward(self,dec,dec_mask,enc,enc_mask):
        for layer in self.model:
            dec = layer(dec,dec_mask,enc,enc_mask)
        return dec

class Input_Embedding(nn.Module):
    def __init__(self,vocab_size,T,emb_size):
        super(Input_Embedding, self).__init__()
        self.word_emb = nn.Embedding(vocab_size,emb_size,padding_idx=2)
        self.position_emb = nn.Embedding(T,emb_size)
        
    def forward(self,inputs,mask,position):
        output = self.word_emb(inputs) + self.position_emb(position)
        output = output * mask.float()
        return output

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 T=20,
                 emb_size=512):
        super(Transformer, self).__init__()
        self.src_emb = Input_Embedding(src_vocab_size,T,emb_size)
        self.trg_emb = Input_Embedding(trg_vocab_size,T,emb_size)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.cls = nn.Linear(emb_size,trg_vocab_size)
        self.T = T
        
    def forward(self,src,src_mask,src_position,trg,trg_mask,trg_position):
        enc = self.src_emb(src,src_mask,src_position)
        enc = self.encoder(enc,src_mask)
        
        dec = self.trg_emb(trg,trg_mask,trg_position)
        dec = self.decoder(dec,trg_mask,enc,src_mask)
        logit = self.cls(dec)
        
        return logit

    def inference(self,src,src_mask,src_position,trg,trg_mask,trg_position):
        enc = self.src_emb(src,src_mask,src_position)
        enc = self.encoder(enc,src_mask)

        for t in range(self.T-1):
            dec = self.trg_emb(trg,trg_mask,trg_position)
            dec = self.decoder(dec,trg_mask,enc,src_mask)
            logit = self.cls(dec)
            _,pred = logit.max(-1)
            trg[:,t+1] = pred[:,t]
            trg_mask[:,t+1,:] = 1

        return trg