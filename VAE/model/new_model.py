import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 48), 
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(32, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x # [seq_len, output_dim]

class TimeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_directions):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_directions = num_directions
        bidirectional = False if self.num_directions == 1 else True
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=bidirectional)

    def forward(self, x):
        output, (h, c) = self.rnn(x)
        output = output.squeeze(1) #[seq_len, num_directions * hidden_dim]
        return output

class Reparametrize(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.output_to_mean = nn.Linear(input_dim, latent_dim)
        self.output_to_logvar = nn.Linear(input_dim, latent_dim)

        nn.init.xavier_uniform_(self.output_to_mean.weight)
        nn.init.xavier_uniform_(self.output_to_logvar.weight)

    def forward(self, x):
        self.mean = self.output_to_mean(x) #[seq_len, latent_dim]
        self.logvar = self.output_to_logvar(x) #[seq_len, latent_dim]

        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(self.mean)
        return z, self.mean, self.logvar

class Regressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 18), 
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(18, 9), 
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(9, output_dim), 
        )
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, conditional_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conditional_dim = conditional_dim

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim + conditional_dim, 64), 
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(64, output_dim)
        )
    
    def forward(self, z, conditional):
        z = torch.cat([z, conditional], dim=1)
        x = self.fc1(z)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class Model(nn.Module):
    def __init__(self, encoder, timeencoder, reparametrize, decoder, regressor, device):
        super().__init__()
        self.encoder = encoder.to(device)
        self.timeencoder = timeencoder.to(device)
        self.reparametrize = reparametrize.to(device)
        self.decoder = decoder.to(device)
        self.regressor = regressor.to(device)
        self.device = device

    def forward(self, x, conditional):
        x = self.encoder(x)
        x = x.unsqueeze(1) # add batch dim=1
        x = self.timeencoder(x)
        z, mean, logvar = self.reparametrize(x)
        # maybe add a timedecoder here too? TODO
        output = self.decoder(z, conditional)
        y_pred = self.regressor(z)
        return output, mean, logvar, y_pred