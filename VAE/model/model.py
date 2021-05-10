import torch.nn as nn
import torch

class EncoderNoTime(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=256, out_features=output_features),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))

        return x

class EncoderWithTime(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.rnn = nn.LSTM(input_size=input_features, hidden_size=output_features)
    
    def forward(self, x):
        outputs, (h, c) = self.rnn(x)

        return outputs.squeeze(1) #shape=(seq_len, batch=1 i think, num_dir * output_features)

class Reparametrize(nn.Module):
    def __init__(self, encoder_output, latent_size):
        super().__init__()
        self.encoder_output = encoder_output
        self.latent_size = latent_size

        self.fc_to_mean = nn.Linear(encoder_output, latent_size)
        self.fc_to_logvar = nn.Linear(encoder_output, latent_size)

        nn.init.xavier_uniform_(self.fc_to_mean.weight)
        nn.init.xavier_uniform_(self.fc_to_logvar.weight)

    def forward(self, x):
        self.mean = self.fc_to_mean(x)
        self.logvar = self.fc_to_logvar(x)

        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(self.mean)
        return z, self.mean, self.logvar

class DecoderNoTime(nn.Module):
    def __init__(self, latent_size, input_features, condition_features, condition_decoder):
        super().__init__()
        self.latent_size = latent_size
        self.input_features = input_features
        self.condition_decoder = condition_decoder
    
        if self.condition_decoder:
            self.fc1 = nn.Sequential(
                nn.Linear(in_features=latent_size + condition_features, out_features=64),
                nn.ReLU()
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(in_features=latent_size, out_features=64),
                nn.ReLU()
            )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU()
        )
        self.fc5 = nn.Linear(in_features=64, out_features=input_features)

    def forward(self, z, condition):
        if self.condition_decoder:
            z = torch.cat([z, condition], dim=1)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        z = self.fc4(z)
        x = self.fc5(z)

        return x

class Regressor(nn.Module):
    def __init__(self, input_features, conditional_features):
        super().__init__()        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=input_features+conditional_features, out_features=32),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU()
        )

        self.fc5 = nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU()
        )
        
        self.fc6 = nn.Linear(16, 1)
    
    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        return x

class Model(nn.Module):
    def __init__(self, encoder, reparametrize, decoder, regressor):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reparametrize = reparametrize
        self.regressor = regressor

    def forward(self, x, conditional):
        x = self.encoder(x)
        z, mean, logvar = self.reparametrize(x)
        x = self.decoder(z, conditional)
        y = self.regressor(z)
        return x, mean, logvar, y
    def __init__(self, encoder, reparametrize, decoder, regressor):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reparametrize = reparametrize
        self.regressor = regressor

    def forward(self, x, conditional):
        x = self.encoder(x)
        z, mean, logvar = self.reparametrize(x)
        x = self.decoder(z, conditional)
        y = self.regressor(z, conditional)
        return x, mean, logvar, y