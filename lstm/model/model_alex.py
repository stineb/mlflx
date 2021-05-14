import math
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn
import torch.optim as optim
import torch



def loss_fn(x_decoded, x, mu, logvar, w, pz_mean, r_mean, r_logvar, r_input):
    kl_loss = w * (-0.5) * torch.mean(1 + logvar - (mu - pz_mean).pow(2) - logvar.exp())
    recon_loss = F.mse_loss(x_decoded, x)
    label_loss = torch.mean(0.5 * (r_mean - r_input).pow(2) / r_logvar.exp() + 0.5 * r_logvar)
    return kl_loss + recon_loss + label_loss, kl_loss, recon_loss, label_loss


class Encoder(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.rnn = nn.LSTM(input_size=input_features, hidden_size=output_features)
    
    def forward(self, x):
        outputs, (h, c) = self.rnn(x)

        return outputs.squeeze(1) #shape=(seq_len, num_dir * output_features)

class Reparametrize(nn.Module):
    def __init__(self, encoder_output, latent_size):
        super().__init__()
        self.encoder_output = encoder_output
        self.latent_size = latent_size

        self.fc_to_mean = nn.Linear(encoder_output, latent_size)
        self.fc_to_logvar = nn.Linear(encoder_output, latent_size)

        self.fc_to_r_mean = nn.Linear(encoder_output, 1)
        self.fc_to_r_logvar = nn.Linear(encoder_output, 1)

        self.fc_r_to_pzmean = nn.Linear(1, latent_size)

        nn.init.xavier_uniform_(self.fc_to_mean.weight)
        nn.init.xavier_uniform_(self.fc_to_logvar.weight)
        nn.init.xavier_uniform_(self.fc_to_r_mean.weight)
        nn.init.xavier_uniform_(self.fc_to_r_logvar.weight)

    def forward(self, x):
        self.mean = self.fc_to_mean(x)
        self.logvar = self.fc_to_logvar(x)

        self.r_mean = self.fc_to_r_mean(x)
        self.r_logvar = self.fc_to_r_logvar(x)

        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(self.mean)

        std = torch.exp(0.5 * self.r_logvar)
        eps = torch.randn_like(std)
        r = eps.mul(std).add_(self.r_mean)

        return z, self.mean, self.logvar, self.fc_r_to_pzmean(r), self.r_mean, self.r_logvar

class Decoder(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.rnn = nn.LSTM(input_size=input_features, hidden_size=output_features)

        self.rnn_to_output = nn.Sequential(
            nn.Linear(self.output_features, self.output_features),
            nn.Tanh(),
            nn.Linear(self.output_features, self.output_features),
        )

    def forward(self, x):
        outputs, (h, c) = self.rnn(x)
        outputs = outputs.squeeze(1)
        return self.rnn_to_output(outputs) #shape=(seq_len, num_dir * output_features)

class Model(nn.Module):
    def __init__(self, encoder, decoder, reparam):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reparam = reparam

    def forward(self, x, conditional):
        x = self.encoder(x) # x has the conditional
        z, mean, logvar, pz_mean, r_mean, r_logvar = self.reparam(x)

        # concat the conditoinal to z
        z1 = torch.cat([z, conditional], dim=1)
        
        # decode
        x = self.decoder(z1.unsqueeze(1))

        return x, mean, logvar, pz_mean, r_mean, r_logvar