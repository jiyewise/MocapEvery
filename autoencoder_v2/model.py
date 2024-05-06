import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from IPython import embed



def load_model(data_config, model_config):
    if model_config['type'] == 'conv3':
        return Conv3AutoEncoder(input_dim=data_config['input_dim'], num_channels=model_config['num_channels'])
    else:
        raise NotImplementedError("Other types of autoencoder is not implemented!")
    
class Conv3AutoEncoder(nn.Module):
    def __init__(self, input_dim, num_channels, dropout=0.1):
        super(Conv3AutoEncoder, self).__init__()
        self.encoder_l1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(input_dim,out_channels=num_channels,kernel_size=25, padding=12),
            nn.ReLU(True),
            nn.BatchNorm1d(num_channels),
            nn.MaxPool1d(kernel_size=2, stride=2)   # (batch, num_channels, 120)
        )   

        self.encoder_l2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, out_channels=num_channels, kernel_size=25, padding=12),
            nn.ReLU(True),
            nn.BatchNorm1d(num_channels),
            nn.MaxPool1d(kernel_size=2, stride=2)   # (batch, num_channels, 60)
        )

        self.encoder_l3 = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Conv1d(num_channels, out_channels=num_channels, kernel_size=25, padding=12),
            nn.ReLU(True),
            nn.BatchNorm1d(num_channels),
            nn.MaxPool1d(kernel_size=2, stride=2, return_indices=False)   # (batch, num_channels, 30) 
        )

        # self.unpool = nn.MaxUnpool1d(kernel_size=2, stride=2)

        self.decoder_l1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.ConvTranspose1d(in_channels=num_channels, out_channels=num_channels, kernel_size=25, stride=2, padding=12, output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(num_channels)
        )

        self.decoder_l2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.ConvTranspose1d(in_channels=num_channels, out_channels=num_channels, kernel_size=25, stride=2, padding=12, output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(num_channels)
        )  
        
        self.decoder_l3 = nn.Sequential(
            # nn.Dropout(dropout),
            nn.ConvTranspose1d(in_channels=num_channels, out_channels=input_dim, kernel_size=25, stride=2, padding=12, output_padding=1)
            # nn.ReLU(True)
        )  

    def encode(self, x):
        x = x.permute(0,2,1)     # Input: (batch, original_input_dim, seq_len=200)

        x = self.encoder_l1(x)  # (batch, out_channels, 100) 28
        x = self.encoder_l2(x)  # (batch, out_channels, 50) 2
        x = self.encoder_l3(x)  # (batch, out_channels, 25)
        return x

    def decode(self, x):
        x = self.decoder_l1(x)  # (batch, out_channels, 50)
        x = self.decoder_l2(x)  # (batch, out_channels, 100)
        x = self.decoder_l3(x)  # (batch, original_input_dim, seq_len=200)
        return x.permute(0,2,1)     # Output: (batch, seq_len, original_input_dim)

    def forward(self, x):  
        # embed()
        x = x.permute(0,2,1)     # Input: (batch, original_input_dim, seq_len=200)

        x = self.encoder_l1(x)  # (batch, out_channels, 100) 28
        x = self.encoder_l2(x)  # (batch, out_channels, 50) 2
        x = self.encoder_l3(x)  # (batch, out_channels, 25)

        # x = self.unpool(x, ind)
        x = self.decoder_l1(x)  # (batch, out_channels, 50)
        x = self.decoder_l2(x)  # (batch, out_channels, 100)
        x = self.decoder_l3(x)  # (batch, original_input_dim, seq_len=200)
        return x.permute(0,2,1)     # Output: (batch, seq_len, original_input_dim)


class ConvAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvAutoencoder, self).__init__()
        self.encoder_c1 = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(input_dim, out_channels=256, kernel_size=25, padding=12),        # 256, 73, 200
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)   # 256, 73, 120
        )
        self.decoder = nn.Sequential(
            # nn.MaxUnpool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.ConvTranspose1d(in_channels=256, out_channels=input_dim, kernel_size=25, stride=2, padding=12, output_padding=1),
            # nn.ReLU(True)
          )  

    # def init_weights(self):
    #     for name, param in self.named_parameters():
    #         nn.init.uniform_(param.data, -0.08, 0.08)

    def encode(self, x):
        x = x.permute(0,2,1)
        x = self.encoder_c1(x) # [batch, out_channel, conv_seq_length]
        return x

    def decode(self, x):
        x = self.decoder(x) # [batch, 139, original_seq_length]
        return x.permute(0,2,1)

    def forward(self, x):
        # x shape [64, seq_len, 139]
        # should be converted to [64, 139, seq_len]
        x = x.permute(0,2,1)
        x = self.encoder_c1(x) # [batch, out_channel, conv_seq_length]
        x = self.decoder(x) # [batch, 139, original_seq_length]

        return x.permute(0,2,1)

class Seq2Seq(nn.Module):
    """Seq2Seq model for sequence generation. The interface takes predefined
    encoder and decoder as input.

    Attributes:
        encoder: Pre-built encoder
        decoder: Pre-built decoder
    """

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.1, 0.1)

    def encode(self, src):
        hidden, cell, outputs = self.encoder(src)
        return hidden, cell

    def decode(self, tgt, hidden, cell, max_len=None, teacher_forcing_ratio=0):
        outputs = self.decoder(
            tgt, hidden, cell, max_len, teacher_forcing_ratio,
        )
        return outputs

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        """
        Inputs:
            src: Source sequence provided as input to the encoder.
                Expected shape: (batch_size, seq_len, input_dim)
            tgt: Target sequence provided as input to the decoder. During
                training, provide reference target sequence. For inference,
                provide only last frame of source.
                Expected shape: (batch_size, seq_len, input_dim)
            max_len: Optional; Length of sequence to be generated. By default,
                the decoder generates sequence with same length as `tgt`
                (training).
            teacher_forcing_ratio: Probability of feeding gold target pose as
                decoder input instead of predicted pose from previous time step
        """
        # hidden [1, batch, 1024(hidden dim size)]
        # cell [1, batch, 1024(hidden dim size)]
        # outputs [batch, 120(seq_len), 1024(hidden_dim)]
        
        hidden, cell, outputs = self.encoder(src)
        self.latent_variables = (hidden, cell)
        outputs = self.decoder(
            tgt, hidden, cell, max_len, teacher_forcing_ratio,
        )
        # outputs shape [batch, 24(tgt_seq), 72(3J)]
        return outputs

class LSTMEncoder(nn.Module):
    def __init__(
        self, input_dim=None, hidden_dim=1024, num_layers=1, lstm=None,
    ):
        """LSTMEncoder encodes input vector using LSTM cells.

        Attributes:
            input_dim: Size of input vector 
            hidden_dim: Size of hidden state vector
            num_layers: Number of layers of LSTM units
            lstm: Optional; If provided, the lstm cell will be used in the
                encoder. This is useful for sharing lstm parameters with
                decoder.
        """
        # input dim (in aa representation) [3J]
        super(LSTMEncoder, self).__init__()
        self.lstm = lstm
        if not lstm:
            assert input_dim is not None
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
            )
        

    def forward(self, input):
        """
        Input:
            input: Input vector to be encoded.
                Expected shape is (batch_size, seq_len, input_dim)
        """
        input = input.transpose(0, 1) # input tensor shape changed to [seq_len, batch, 3J]
        # outputs [seq, batch, hidden_dim]
        # lstm_hidden [1, batch, hidden_dim]
        # lstm_cell [1, batch, hidden_dim]
        outputs, (lstm_hidden, lstm_cell) = self.lstm(input) 
        return lstm_hidden, lstm_cell, outputs.transpose(0, 1)


class DecoderStep(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, lstm=None):
        super(DecoderStep, self).__init__()
        self.lstm = (
            nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)
            if not lstm
            else lstm
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden=None, cell=None, encoder_outputs=None):
        if (hidden is None) and (cell is None):
            output, (hidden, cell) = self.lstm(input)
        else:
            output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output = output.squeeze(0)
        output = self.out(output)
        return output, hidden, cell


class LSTMDecoder(nn.Module):
    """Decoder to generate sequences using LSTM cells. Decoding is done in a
    greedy manner without attention mechanism.

    Attributes:
        input_dim: Size of input vector
        output_dim: Size of output to be generated at each time step
        hidden_dim: Size of hidden state vector
        device: Optional; Device to be used "cuda" or "cpu"
        lstm: Optional; If provided, the lstm cell will be used in the decoder.
            This is useful for sharing lstm parameters from encoder.
    """

    def __init__(
        self, input_dim, output_dim, hidden_dim, device="cuda", lstm=None
    ):
        super(LSTMDecoder, self).__init__()
        self.input_dim = input_dim
        self.decoder_step = DecoderStep(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            lstm=lstm,
        )
        self.device = device

    def forward(
        self,
        tgt,
        hidden=None,
        cell=None,
        max_len=None,
        teacher_forcing_ratio=0.5,
    ):
        """
        Inputs:
            tgt: Target sequence provided as input to the decoder. During
                training, provide reference target sequence. For inference,
                provide only last frame of source.
                Expected shape: (seq_len, batch_size, input_dim)
            hidden, cell: Hidden state and cell state to be used in LSTM cell
            max_len: Optional; Length of sequence to be generated. By default,
                the decoder generates sequence with same length as `tgt`
                (training).
            teacher_forcing_ratio: Probability of feeding gold target pose as
                decoder input instead of predicted pose from previous time step
        """
        # originally, tgt shape is [64, 24, 72] [batch, tgt_seq, 3J] in aa rep
        tgt = tgt.transpose(0, 1) # change to [tgt_seq, batch, 3J]
        max_len = max_len if max_len is not None else tgt.shape[0] # set to tgt_seq if none (none in training case)
        batch_size = tgt.shape[1]

        # input = tgt[0, :] # [batch, 3J] get the first pose of the target sequence
        outputs = torch.zeros(max_len, batch_size, self.input_dim,).to(
            self.device
        ) # [24, 64, 72] or [tgt_seq, batch, 3J]
        
        outputs[0] = tgt[0,:] 
        for t in range(max_len-1):
            if t == 0:
                input = tgt[0,:] # [batch, 3J] get the first pose of the target sequence
            else:
                teacher_force = random.random() < teacher_forcing_ratio
                input = tgt[t] if teacher_force else output
            input = input.unsqueeze(0) # change to [1, batch, 3J]
            output, hidden, cell = self.decoder_step(input, hidden, cell) # output [batch, 3J] hidden [1, batch, hidden_dim] cell [1, batch, hidden]
            outputs[t+1] = output

        outputs = outputs.transpose(0, 1) # convert to [batch, tgt_seq, 3J]
        return outputs


class DecoderStepWithAttention(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim, source_length, device="cuda",
    ):
        super(DecoderStepWithAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.source_length = source_length
        self.device = device

        self.attn = nn.Linear(
            self.hidden_dim + self.input_dim, self.source_length,
        )
        self.attn_combine = nn.Linear(
            self.hidden_dim + self.input_dim, self.input_dim,
        )
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        attn_weights = F.softmax(
            self.attn(torch.cat((input, hidden), 2)), dim=2,
        )
        attn_applied = torch.bmm(attn_weights.transpose(0, 1), encoder_outputs)

        output = torch.cat((input, attn_applied.transpose(0, 1)), 2)
        output = self.attn_combine(output)
        output = F.relu(output)

        if (hidden is None) and (cell is None):
            output, (hidden, cell) = self.lstm(output)
        else:
            output, (hidden, cell) = self.lstm(output, (hidden, cell))
        output = output.squeeze(0)
        output = self.out(output)
        return output, hidden, cell


class LSTMDecoderWithAttention(LSTMDecoder):
    def __init__(
        self,
        input_dim,
        output_dim,
        max_source_length,
        hidden_dim=128,
        device="cuda",
    ):
        """Extension of LSTMDecoder that uses attention mechanism to generate
        sequences.

        Attributes:
            input_dim: Size of input vector
            output_dim: Size of output to be generated at each time step
            max_source_length: Length of source sequence
            hidden_dim: Size of hidden state vector
            device: Optional; Device to be used "cuda" or "cpu"
        """
        super(LSTMDecoderWithAttention, self).__init__(
            input_dim, output_dim, hidden_dim, device
        )
        self.decoder_step = DecoderStepWithAttention(
            input_dim, output_dim, hidden_dim, max_source_length
        )
        self.device = device
