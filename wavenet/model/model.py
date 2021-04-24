import torch
import torch.nn as nn


class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation, **kwargs)
    
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.conv.padding[0]]

class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, dilation):
        super().__init__()
        self.conv = CausalConv1d(residual_channels, residual_channels, 2, dilation=dilation)
        self.conv_residual = nn.Conv1d(residual_channels, residual_channels, 1)
        self.conv_skip = nn.Conv1d(residual_channels, skip_channels, 1)

        self.gate_tanh = nn.Tanh()
        self.gate_sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.conv(x)

        gated_tanh = self.gate_tanh(output)
        gated_sigmoid = self.gate_sigmoid(output)
        gated = gated_tanh * gated_sigmoid

        output = self.conv_residual(gated)
        
        output += x

        # Skip connection
        skip = self.conv_skip(gated)

        return output, skip

class DensNet(torch.nn.Module):
    def __init__(self, channels):
        """
        The last network of WaveNet
        :param channels: number of channels for input and output
        :return:
        """
        super(DensNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(channels, channels, 1)
        self.conv2 = torch.nn.Conv1d(channels, 1, 1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)


        return output

class WaveNet(nn.Module):
    def __init__(self, n_dilations, n_residuals, in_channels, res_channels, device):
        super().__init__()
        self.causal = CausalConv1d(in_channels, res_channels, 2, 1)
        self.residual_blocks = [[] for i in range(n_dilations)]
        self.densenet = DensNet(in_channels).to(device)
        self.n_residuals = n_residuals
        self.n_dilations = n_dilations

        for residual_id in range(n_residuals):
            for dilation_id in range(n_dilations):
                dilation = 2 ** dilation_id
                res_block = ResidualBlock(res_channels, in_channels, dilation).to(device)
                self.residual_blocks[dilation_id].append(res_block)

    def forward(self, x):
        skip_connections = []

        for dilation_id in range(self.n_dilations):
          output = self.causal(x)
          for block in self.residual_blocks[dilation_id]:
            output, skip = block(output)
            skip_connections.append(skip)

        output = torch.sum(torch.stack(skip_connections), dim=0)
        return self.densenet(output)