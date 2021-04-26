from model.new_model import Encoder,TimeEncoder, Reparametrize, Decoder, Regressor, Model
from sklearn.metrics import r2_score
from utils.preprocess import prepare_df, normalize
from model.loss import loss_fn
from tqdm import tqdm
import json
import torch
import pandas as pd
import argparse
import numpy as np

# Parse arguments 
parser = argparse.ArgumentParser(description='CV wavenet')

parser.add_argument('-gpu', '--gpu', default=None, type=str,
                      help='indices of GPU to enable ')

parser.add_argument('-e', '--n_epochs', default=None, type=int,
                      help='number of cv epochs (wavenet)')

parser.add_argument('-d', '--latent_dim', default=None, type=int,
                      help='latent dim')

args = parser.parse_args()
DEVICE = torch.device("cuda:" + args.gpu)
# Load Configs
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)


data = pd.read_csv(config_data["data_dir"], index_col=0).drop(columns=['lat', 'lon', 'elv','date','c4','whc'])

# Drop AR-Vir and CN-Cng
data = data[data.index != "AR-Vir"]
data = data[data.index != "CN-Cng"]
df_sensor, df_meta, df_gpp = prepare_df(data)



LATENT_SIZE = args.latent_dim
CONDITIONAL_FEATURES = len(df_meta[0].columns)
INPUT_FEATURES = len(df_sensor[0].columns) + CONDITIONAL_FEATURES



ENCODER_INPUT_DIM = INPUT_FEATURES
ENCODER_OUTPUT_DIM = 32
TIMEENCODER_INPUT_DIM = ENCODER_OUTPUT_DIM
TIMEENCODER_HIDDEN_DIM = 256
TIMEENCODER_NUM_DIRECTIONS = 2
REPARAMETRIZE_INPUT_DIM = TIMEENCODER_HIDDEN_DIM * TIMEENCODER_NUM_DIRECTIONS
REPARAMETRIZE_LATENT_DIM = 50
DECODER_INPUT_DIM = REPARAMETRIZE_LATENT_DIM
DECODER_OUTPUT_DIM = ENCODER_INPUT_DIM
DECODER_CONDITIONAL_DIM = CONDITIONAL_FEATURES





cv_r2 = []

for s in tqdm(range(len(df_sensor))):
  sites_to_train = list(range(len(df_sensor)))
  sites_to_train.remove(s)
  sites_to_test = [s]

  x_train = [pd.concat([df_sensor[i],df_meta[i]],axis=1).values for i in sites_to_train]
  conditional_train = [df_meta[i].values for i in sites_to_train]
  y_train = [df_gpp[i].values.reshape(-1,1) for i in sites_to_train]
 
  x_test = [pd.concat([df_sensor[i],df_meta[i]],axis=1).values for i in sites_to_test]
  conditional_test = [df_meta[i].values for i in sites_to_test]
  y_test = [df_gpp[i].values.reshape(-1,1) for i in sites_to_test]
 
  encoder = Encoder(ENCODER_INPUT_DIM, ENCODER_OUTPUT_DIM)
  time_encoder = TimeEncoder(TIMEENCODER_INPUT_DIM, TIMEENCODER_HIDDEN_DIM, TIMEENCODER_NUM_DIRECTIONS)
  reparametrize = Reparametrize(REPARAMETRIZE_INPUT_DIM, REPARAMETRIZE_LATENT_DIM)
  decoder = Decoder(DECODER_INPUT_DIM, DECODER_OUTPUT_DIM, DECODER_CONDITIONAL_DIM)
  regressor = Regressor(REPARAMETRIZE_LATENT_DIM, 1)
  model = Model(encoder, time_encoder, reparametrize, decoder, regressor, DEVICE)
  model.to(DEVICE)
  
  optimizer = torch.optim.Adam(model.parameters())
  r2 = []

  for epoch in range(args.n_epochs):

      model.train()
      for (x, y, conditional) in zip(x_train, y_train, conditional_train):
        x = torch.FloatTensor(x).to(DEVICE)
        y = torch.FloatTensor(y).to(DEVICE)
        conditional = torch.FloatTensor(conditional).to(DEVICE)
        outputs, mean, logvar, y_pred = model(x, conditional)

        optimizer.zero_grad()
        loss, recon_loss, kl_loss, reg_loss = loss_fn(outputs, x, y_pred, y, mean, logvar, 1)
        loss.backward()
        optimizer.step()
      
      model.eval()
      with torch.no_grad():
          for (x, y, conditional) in zip(x_test, y_test, conditional_test):
            x = torch.FloatTensor(x).to(DEVICE)
            y = torch.FloatTensor(y).to(DEVICE)
            conditional = torch.FloatTensor(conditional).to(DEVICE)

            outputs, mean, logvar, y_pred = model(x, conditional)

            
            loss, recon_loss, kl_loss, reg_loss = loss_fn(outputs, x, y_pred, y, mean, logvar, 1)
            r2.append(r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy()))
  cv_r2.append(max(r2))
  print(f"Test Site: {data.index.unique()[s]} R2: {cv_r2[s]}")
  print("CV R2 cumulative mean: ", np.mean(cv_r2), " +- ", np.std(cv_r2))
  print("-------------------------------------------------------------------")
 