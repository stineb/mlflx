from model.model import EncoderWithTime, Reparametrize, DecoderNoTime, Regressor, Model
from sklearn.metrics import r2_score
from utils.preprocess_time import prepare_df, normalize
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
#python cv-cutbytime.py -gpu 7 -e 15 -d 16
args = parser.parse_args()
DEVICE = torch.device("cuda:" + args.gpu)

# DEVICE = 'cpu'

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

# cv_r2 = [s]

sites_to_train = list(range(len(df_sensor)))
sites_to_test = list(range(len(df_sensor)))

x_train = [pd.concat([normalize(df_sensor[i][:round(len(df_sensor[i])/2)]),df_meta[i][:round(len(df_sensor[i])/2)]],axis=1).values for i in sites_to_train]
conditional_train = [df_meta[i][:round(len(df_sensor[i])/2)].values for i in sites_to_train]
y_train = [df_gpp[i][:round(len(df_sensor[i])/2)].values.reshape(-1,1) for i in sites_to_train]


x_test = [pd.concat([normalize(df_sensor[i][round(len(df_sensor[i])/2):]),df_meta[i][round(len(df_sensor[i])/2):]],axis=1).values for i in sites_to_test]
conditional_test = [df_meta[i][round(len(df_sensor[i])/2):].values for i in sites_to_test]
y_test = [df_gpp[i][round(len(df_sensor[i])/2):].values.reshape(-1,1) for i in sites_to_test]

encoder = EncoderWithTime(INPUT_FEATURES, ENCODER_OUTPUT_SIZE).to(DEVICE)
reparam = Reparametrize(ENCODER_OUTPUT_SIZE, LATENT_SIZE).to(DEVICE)
decoder = DecoderNoTime(LATENT_SIZE, INPUT_FEATURES, CONDITIONAL_FEATURES, CONDITION_DECODER).to(DEVICE)
regressor = Regressor(LATENT_SIZE)

model = Model(encoder, reparam, decoder, regressor).to(DEVICE)


optimizer = torch.optim.Adam(model.parameters())
r2 = []
r2_all=[]
for epoch in range(args.n_epochs):
  print(epoch)
  pred_all=[]
  y_all=[]
  model.train()
  for (x, y, conditional) in zip(x_train, y_train, conditional_train):
    x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
    y = torch.FloatTensor(y).to(DEVICE)
    conditional = torch.FloatTensor(conditional).to(DEVICE)
    outputs, mean, logvar, y_pred = model(x, conditional)
    x = x.squeeze(1)

    optimizer.zero_grad()
    loss, recon_loss, kl_loss, reg_loss = loss_fn(outputs, x, y_pred, y, mean, logvar, 1)
    loss.backward()
    optimizer.step()

  model.eval()
  with torch.no_grad():
      for (x, y, conditional) in zip(x_test, y_test, conditional_test):
        x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
        y = torch.FloatTensor(y).to(DEVICE)
        conditional = torch.FloatTensor(conditional).to(DEVICE)

        outputs, mean, logvar, y_pred = model(x, conditional)

        x = x.squeeze(1)

        loss, recon_loss, kl_loss, reg_loss = loss_fn(outputs, x, y_pred, y, mean, logvar, 1)
        r2.append(r2_score(y_true=y.detach().cpu().numpy(),y_pred=y_pred.detach().cpu().numpy()))
        
#         pred_all.append(y_pred)
#         y_all.append(y)
#   r2_all.append(r2_score(torch.cat(y_all),torch.cat(pred_all)))
# print(r2_all)
# # cv_r2.append(max(r2))
# # print(f"Test Site: {data.index.unique()[s]} R2: {cv_r2[s]}")
# # print("CV R2 cumulative mean: ", np.mean(cv_r2), " +- ", np.std(cv_r2))
# print("-------------------------------------------------------------------")

# r2_last=r2[-66:]
# for s in range(0,66):
#  print(f"Site: {data.index.unique()[s]} R2: {r2_last[s]}")
# print("r2_all",r2_all[-1:])
r2_last=r2[-64:]
for s in range(0,64):
 print(f"Site: {data.index.unique()[s]} R2: {r2_last[s]}")