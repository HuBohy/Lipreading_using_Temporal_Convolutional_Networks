import torch
import torch.nn as nn
import torch.nn.functional as F

from lipreading.model import MultiscaleMultibranchTCN

class FusionNet(nn.Module):
    def __init__(self, input_size = 1536, num_classes = 3, tcn_options = {}) -> None:
        super(FusionNet, self).__init__()

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(input_size, 1024)
        self.fc4 = nn.Linear(1024, num_classes)

        #other layers
        # hidden_dim = 256
        # num_channels = [hidden_dim*len(tcn_options['kernel_size'])*tcn_options['width_mult']]*tcn_options['num_layers']
        # self.tcn1 = MultiscaleMultibranchTCN(input_size,
        #     num_channels=num_channels,
        #     num_classes=num_classes,
        #     tcn_options=tcn_options,
        #     dropout=tcn_options['dropout'],
        #     relu_type='prelu',
        #     dwpw=tcn_options['dwpw'])

        # self.tcn_output = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x

        # B = x.size()[0]
        # lengths = [y.size()[-1] for y in x]
        # return self.tcn_output(self.tcn1(x, lengths, B))