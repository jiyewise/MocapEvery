# (FairMotion) Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn.init import xavier_uniform_

import random
from IPython import embed

# add transformer encoder module 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000): # d_model: ninp/hidden_dim in original
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [5000] -> [5000, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        ) # [0.5*d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # unsqueeze: [1, max_len, d_model] -> transpose: [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model] # self.pe size [max_len, 1, d_model]
        x = x + self.pe[:x.size(0), :] # cut pe into [seq_len, 1, d_model] and add to all batches
        return x
        # return self.dropout(x)

class TransformerEncoderModel(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.1, estimate_contact=False
    ):
        """
        input_dim: this is the dimension of the input
        ninp: 1024 this is the dimension of the hidden layer
        hidden_dim: same as ninp
        num_layers: the number of layers in transformer encoder and decoder. can be either 1 or 4

        """
        self.mid_dim = None
        if isinstance(input_dim, tuple):
            self.input_dim, self.mid_dim = input_dim

        self.hidden_dim = hidden_dim

        super(TransformerEncoderModel, self).__init__()
        self.model_type = "TransformerEncoder"

        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim, dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=LayerNorm(hidden_dim),
        )

        # Use Linear instead of Embedding for continuous valued input
        if self.mid_dim is not None:
            half_hidden_dim = int(hidden_dim/2)
            self.mid_encoder = nn.Linear(self.mid_dim, half_hidden_dim)
            self.input_encoder = nn.Linear(self.input_dim, half_hidden_dim)

        else:
            self.encoder = nn.Linear(input_dim, hidden_dim)

        self.hidden_dim = hidden_dim
        
        # foot fc 
        decode_dim = hidden_dim

        self.estimate_contact = estimate_contact
        if self.estimate_contact:
            self.contact_decoder = nn.Sequential(
                                nn.Linear(hidden_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, 2)
                )        
            decode_dim += 2

        # self.estimate_foot = estimate_foot
        # if self.estimate_foot:
            # self.foot_decoder = nn.Sequential(
            #                     nn.Linear(hidden_dim, 256),
            #                     nn.ReLU(),
            #                     nn.Linear(256, 6)
            #     )        
            # decode_dim += 6

        self.linear_decoder = nn.Sequential(
                            nn.Linear(decode_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, output_dim)
            )
        
        
        self.init_weights()

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def forward(self, src):
        # Transformer expects src and tgt in format (len, batch_size, dim)
        src = src.transpose(0, 1) # by transpose, [seq, batch, ninp]
        if self.mid_dim is None:
            projected_src = self.encoder(src) * np.sqrt(self.hidden_dim) # why add np.sqrt? [seq, batch, hidden_dim]
        else:
            half_hidden_dim = int(self.hidden_dim/2)

            src_input, src_mid = src[...,:self.input_dim], src[...,self.input_dim:]
            projected_input_src = self.input_encoder(src_input)
            projected_mid_src = self.mid_encoder(src_mid)
            projected_src = torch.cat((projected_input_src, projected_mid_src),-1) * np.sqrt(self.hidden_dim)

        pos_encoded_src = self.pos_encoder(projected_src) # [seq, batch, hidden_dim]
        encoder_output = self.transformer_encoder(pos_encoded_src) # [seq, batch, ninp] encoder output

        if self.estimate_contact:
            contact_output = self.contact_decoder(encoder_output) # [seq, batch, 18]
            encoder_output = torch.cat((encoder_output, contact_output), dim=2)

        # TODO check dimensions 
        output = self.linear_decoder(encoder_output) # [seq, batch, output_dim]

        if self.estimate_contact:
            return contact_output.transpose(0, 1), output.transpose(0, 1)

        return None, output.transpose(0, 1) # [batch, seq, output_dim]

        # return output.transpose(0, 1) # [batch, seq, output_dim]
