from model.model import WaveNet
from model.metric import r2_score
from utils.preprocess import batch_by_site, normalize, make_batches
from tqdm import tqdm
import json
import torch
import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='onesite wavenet')

parser.add_argument('-gpu', '--gpu', default=None, type=str,
                      help='indices of GPU to enable ')

parser.add_argument('-di', '--n_dilations', default=None, type=int,
                      help='number of dilations (wavenet)')

parser.add_argument('-rb', '--n_residuals', default=None, type=int,
                      help='number of residual blocks (wavenet)')
parser.add_argument('-e', '--n_epochs', default=None, type=int,
                      help='number of cv epochs (wavenet)')
#-gpu 2 -di 4 -rb 2 -e 50
args = parser.parse_args()
print(args)
DEVICE = torch.device("cuda:"+ args.gpu)

# # Load Configs
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

df = pd.read_csv(config_data["data_dir"],index_col=0)
sites_df = batch_by_site(df)
n_features  = len(sites_df[0].columns)-1
cv_mse = []
cv_r2 = []
# for s in tqdm(range(len(sites_df))):
# sites_to_train = list(range(0, len(sites_df)))
# sites_to_train.remove(s)
# site_to_test = [s]

# train = [sites_df[i] for i in sites_to_train]
# test = [sites_df[i] for i in site_to_test]


sites_to_train = list(range(0, len(sites_df)))
site_to_see = [sites_df[i] for i in sites_to_train]
train=site_to_see
test=site_to_see

for i in range(len(site_to_see)):
 train[i] = site_to_see[i].loc[:len(site_to_see[i])/2]

for i in range(len(site_to_see)):
 test[i] = site_to_see[i].loc[len(site_to_see[i])/2:]

for i in range(len(train)):
 train[i] = normalize(train[i])

for i in range(len(test)):
 test[i] = normalize(test[i])

X_train, y_train = make_batches(train)
X_test, y_test = make_batches(test)

# train=sites_df[0].loc[:len(sites_df[0])/2]
# test=sites_df[0].loc[len(sites_df[0])/2:]
    
# train=normalize(train)
# test=normalize(test)

# X_train=train.drop(columns=["GPP_NT_VUT_REF"]).to_numpy()
# y_train=train['GPP_NT_VUT_REF'].to_numpy()
# X_test=test.drop(columns=["GPP_NT_VUT_REF"]).to_numpy()
# y_test=test['GPP_NT_VUT_REF'].to_numpy()


model = WaveNet(args.n_dilations, args.n_residuals, n_features, 128, DEVICE).to(DEVICE)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

train_losses = []
test_losses = []
r2 = []
r2_all=[]

for epoch in range(args.n_epochs):
  train_loss = 0.0
  test_loss = 0.0
  train_mse = 0.0
  test_mse = 0.0
  pred_all=[]
  y_all=[]
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
          pred_all.append(pred)
          y_all.append(y)
  
  r2_all.append(r2_score(torch.cat(pred_all),torch.cat(y_all)))            
  train_losses.append(train_loss / len(X_train))
  test_losses.append(test_loss / len(X_test))
# cv_r2.append(max(r2))
# cv_mse.append(min(test_losses))

# print("CV MSE cumulative mean: ", np.mean(cv_mse)," +-", np.std(cv_mse))
# print("CV R2 cumulative mean: ", np.mean(cv_r2), " +- ", np.std(cv_r2))
# print("-------------------------------------------------------------------")
r2_last=r2[-66:]
for s in range(0,66):
 print(f"Site: {df.index.unique()[s]} R2: {r2_last[s]}")
# print("r2",r2[-66:])
# print("train_losses",min(train_losses))
# print("test_losses",min(test_losses))
# print("r_overall",r_overall)
print("r2_all",r2_all[-1:])

