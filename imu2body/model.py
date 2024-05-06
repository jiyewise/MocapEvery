import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
from imu2body.model_base import TransformerEncoderModel
import torch
import torch.nn as nn
from IPython import embed

class IMU2BodyModel(nn.Module):
    def __init__(self, data_config, model_config):
        super(IMU2BodyModel, self).__init__()

        input_dim = data_config['input_dim']
        mid_dim = data_config['mid_dim']
        output_dim = data_config['output_dim']

        self.use_sep_encoder = model_config['sep_encoder']
        if self.use_sep_encoder:
            hand2body_input_dim = (input_dim, mid_dim)
        else:
            hand2body_input_dim = input_dim + mid_dim

        # imu + head -> ee pose 
        self.imu2hand = TransformerEncoderModel(
            input_dim=input_dim,
            output_dim=mid_dim,
            hidden_dim=model_config['hidden_dim1'],
            num_heads=model_config['num_head1']
        )

        # imu + head + ee pose -> contact, output
        self.hand2body = TransformerEncoderModel(
            input_dim=hand2body_input_dim,
            output_dim=output_dim,
            hidden_dim=model_config['hidden_dim2'],
            num_heads=model_config['num_head2'],
            estimate_contact=True
        )

    def init_weights(self):
        self.imu2hand.init_weights()
        self.hand2body.init_weights()
    
    def forward(self, input):
        _, ee = self.imu2hand(input) # hand: [batch, seq, 12]
        input_concat = torch.cat((input, ee), -1) # concatenate input and hand
        contact, output = self.hand2body(input_concat)
        return ee, contact, output
    

def load_model(data_config, model_config):
    return IMU2BodyModel(data_config=data_config, model_config=model_config)