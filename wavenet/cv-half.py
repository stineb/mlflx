from model.model import WaveNet
from model.metric import r2_score
from utils.preprocess import batch_by_site, normalize, make_batches
from tqdm import tqdm
import json
import torch
import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='CV wavenet')

parser.add_argument('-gpu', '--gpu', default=None, type=str,
                      help='indices of GPU to enable ')

parser.add_argument('-di', '--n_dilations', default=None, type=int,
                      help='number of dilations (wavenet)')

parser.add_argument('-rb', '--n_residuals', default=None, type=int,
                      help='number of residual blocks (wavenet)')
parser.add_argument('-e', '--n_epochs', default=None, type=int,
                      help='number of cv epochs (wavenet)')

args = parser.parse_args()
#-gpu 4 -di 4 -rb 2 -e2
DEVICE = torch.device("cuda:" + args.gpu)
# DEVICE = torch.device("cuda:2")
# Load Configs
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

df = pd.read_csv(config_data["data_dir"],index_col=0)
sites_df = batch_by_site(df)
n_features  = len(sites_df[0].columns)-1

cv_r2 = []
for s in tqdm(range(0,len(sites_df))):
#   sites_to_train = list(range(0, len(sites_df)))
#   sites_to_train = list(range(1, len(sites_df)))
#   sites_to_train.remove(s)
#   site_to_test = [s]

  train = normalize(sites_df[s].loc[:len(sites_df[s])//2])
  test= normalize(sites_df[s].loc[len(sites_df[s])//2:])

  X_train=[train.drop(columns=["GPP_NT_VUT_REF"]).to_numpy()]
  y_train=[train['GPP_NT_VUT_REF'].to_numpy()]
  X_test=[test.drop(columns=["GPP_NT_VUT_REF"]).to_numpy()]
  y_test=[test['GPP_NT_VUT_REF'].to_numpy()]
  model = WaveNet(args.n_dilations, args.n_residuals, n_features, 128, DEVICE).to(DEVICE)
  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters())

  train_losses = []
  test_losses = []
  r2 = []
  for epoch in range(args.n_epochs):
      train_loss = 0.0
      test_loss = 0.0
      train_mse = 0.0
      test_mse = 0.0

      model.train()
      for (x, y) in zip(X_train, y_train):
          x = torch.FloatTensor(x).unsqueeze(0).permute(0, 2, 1).to(DEVICE)
          y = torch.FloatTensor(y).to(DEVICE)

          pred = model(x)

          pred = pred.squeeze()
          
          optimizer.zero_grad()
          loss = criterion(pred, y)
          loss.backward()
          optimizer.step()

          train_loss += loss.item()
          train_mse += torch.mean((y - pred) ** 2)
      model.eval()
      with torch.no_grad():
          for (x, y) in zip(X_test, y_test):
              x = torch.FloatTensor(x).unsqueeze(0).permute(0, 2, 1).to(DEVICE)
              y = torch.FloatTensor(y).to(DEVICE)

              pred = model(x)

              pred = pred.squeeze()
              
              loss = criterion(pred, y)

              test_loss += loss.item()
              test_mse += torch.mean((y - pred) ** 2)
              r2.append(r2_score(pred,y))
                  
      
  cv_r2.append(max(r2))
  print(f"Test Site: {df.index.unique()[s]}  R2: {cv_r2[s]}")
  print("CV R2 cumulative mean: ", np.mean(cv_r2), " +- ", np.std(cv_r2))
  print("-------------------------------------------------------------------")
