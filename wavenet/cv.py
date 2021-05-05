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
DEVICE = torch.device("cuda:" + args.gpu)
# DEVICE = torch.device("cuda:2")
# Load Configs
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

df = pd.read_csv(config_data["data_dir"],index_col=0)
sites_df = batch_by_site(df)
n_features  = len(sites_df[0].columns)-1
cv_mse = []
cv_r2 = []
for s in tqdm(range(0,len(sites_df))):
  sites_to_train = list(range(0, len(sites_df)))
#   sites_to_train = list(range(1, len(sites_df)))
  sites_to_train.remove(s)
  site_to_test = [s]

  train = [sites_df[i] for i in sites_to_train]
  test = [sites_df[i] for i in site_to_test]

  for i in range(len(train)):
    train[i] = normalize(train[i])

  for i in range(len(test)):
    test[i] = normalize(test[i])

  X_train, y_train = make_batches(train)
  X_test, y_test = make_batches(test)
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
                  
      train_losses.append(train_loss / len(X_train))
      test_losses.append(test_loss / len(X_test))
  cv_r2.append(max(r2))
  cv_mse.append(min(test_losses))
  print(f"Test Site: {df.index.unique()[s]}  MSE: {cv_mse[s]} R2: {cv_r2[s]}")
  print("CV MSE cumulative mean: ", np.mean(cv_mse)," +-", np.std(cv_mse))
  print("CV R2 cumulative mean: ", np.mean(cv_r2), " +- ", np.std(cv_r2))
  print("-------------------------------------------------------------------")
