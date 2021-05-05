from model.model import EncoderWithTime, Reparametrize, DecoderNoTime, Regressor, Model
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



ENCODER_OUTPUT_SIZE = 256
LATENT_SIZE = args.latent_dim
CONDITIONAL_FEATURES = len(df_meta[0].columns)
CONDITION_DECODER = True
INPUT_FEATURES = len(df_sensor[0].columns) + CONDITIONAL_FEATURES




cv_r2 = []

for s in tqdm(range(len(df_sensor))):
  X = pd.concat([df_sensor[s],df_meta[s]],axis=1).values
  Conditional = df_meta[s].values 
  Y = df_gpp[s].values.reshape(-1,1)
  
  
  x_train = X[:len(X)//2]
  conditional_train =  Conditional[:len(Conditional)//2]
  y_train =   Y[:len(Y)//2]
  
  x_test = X[len(X)//2:]
  conditional_test = Conditional[len(Conditional)//2:]
  y_test = Y[len(Y)//2:]

  encoder = EncoderWithTime(INPUT_FEATURES, ENCODER_OUTPUT_SIZE).to(DEVICE)
  reparam = Reparametrize(ENCODER_OUTPUT_SIZE, LATENT_SIZE).to(DEVICE)
  decoder = DecoderNoTime(LATENT_SIZE, INPUT_FEATURES, CONDITIONAL_FEATURES, CONDITION_DECODER).to(DEVICE)
  regressor = Regressor(LATENT_SIZE)
  model = Model(encoder, reparam, decoder, regressor).to(DEVICE)
  
  
  optimizer = torch.optim.Adam(model.parameters())
  r2 = []

  for epoch in range(args.n_epochs):
      model.train()
      x = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
      y = torch.FloatTensor(y_train).to(DEVICE)
      conditional = torch.FloatTensor(conditional_train).to(DEVICE)
      outputs, mean, logvar, y_pred = model(x, conditional)
      x = x.squeeze(1)

      optimizer.zero_grad()
      loss, recon_loss, kl_loss, reg_loss = loss_fn(outputs, x, y_pred, y, mean, logvar, 1)
      loss.backward()
      optimizer.step()
      
      model.eval()
      with torch.no_grad():
          x = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
          y = torch.FloatTensor(y_test).to(DEVICE)
          conditional = torch.FloatTensor(conditional_test).to(DEVICE)

          outputs, mean, logvar, y_pred = model(x, conditional)

          x = x.squeeze(1)
            
          loss, recon_loss, kl_loss, reg_loss = loss_fn(outputs, x, y_pred, y, mean, logvar, 1)
          r2.append(r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy()))
  cv_r2.append(max(r2))
  print(f"Test Site: {data.index.unique()[s]} R2: {cv_r2[s]}")
  print("CV R2 cumulative mean: ", np.mean(cv_r2), " +- ", np.std(cv_r2))
  print("-------------------------------------------------------------------")
 