from model.model import EncoderWithTime, Reparametrize, DecoderNoTime
from sklearn.metrics import r2_score

from utils.preprocess import batch_by_site, normalize
from tqdm import tqdm
import json
import torch
import pandas as pd
import argparse
import numpy as np
import torch.nn.functional as F

def loss_fn(x_decoded, x, y_pred, y, mu, logvar, w):
    kl_loss = w * (-0.5) * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    recon_loss = F.mse_loss(x_decoded, x)
    regression_loss = F.mse_loss(y_pred, y)
    return kl_loss + recon_loss + regression_loss, recon_loss, kl_loss, regression_loss


parser = argparse.ArgumentParser(description='CV wavenet')

parser.add_argument('-gpu', '--gpu', default=None, type=str,
                      help='indices of GPU to enable ')

parser.add_argument('-e', '--n_epochs', default=None, type=int,
                      help='number of cv epochs (wavenet)')

args = parser.parse_args()
DEVICE = torch.device("cuda:" + args.gpu)
# Load Configs
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)


x_time_dep_cols = ['TA_F', 'TA_F_DAY', 'TA_F_NIGHT', 'SW_IN_F', 'LW_IN_F', 'VPD_F', 'PA_F', 'P_F', 'WS_F', 'LE_F_MDS', 'NEE_VUT_REF', 'wscal', 'fpar', 'whc']
x_time_invariant_cols = ['classid_CRO', 'classid_CSH', 'classid_DBF',
       'classid_EBF', 'classid_ENF', 'classid_GRA', 'classid_MF',
       'classid_SAV', 'classid_WET', 'classid_WSA', 'c4_False', 'c4_True',
       'koeppen_code_-', 'koeppen_code_Aw', 'koeppen_code_BSh',
       'koeppen_code_BSk', 'koeppen_code_BWh', 'koeppen_code_Cfa',
       'koeppen_code_Cfb', 'koeppen_code_Csa', 'koeppen_code_Csb',
       'koeppen_code_Dfb', 'koeppen_code_Dfc', 'koeppen_code_ET',
       'igbp_land_use_Cropland/Natural Vegetation Mosaic',
       'igbp_land_use_Croplands', 'igbp_land_use_Deciduous Broadleaf Forest',
       'igbp_land_use_Evergreen Broadleaf Forest',
       'igbp_land_use_Evergreen Needleleaf Forest', 'igbp_land_use_Grasslands',
       'igbp_land_use_Mixed Forests', 'igbp_land_use_Open Shrublands',
       'igbp_land_use_Savannas', 'igbp_land_use_Water',
       'igbp_land_use_Woody Savannas', 'plant_functional_type_Cereal crop',
       'plant_functional_type_Deciduous Broadleaf Trees',
       'plant_functional_type_Evergreen Broadleaf Trees',
       'plant_functional_type_Evergreen Needleleaf Trees',
       'plant_functional_type_Grass', 'plant_functional_type_Shrub',
       'plant_functional_type_Water']
x_cols = x_time_dep_cols + x_time_invariant_cols
y_col = ['GPP_NT_VUT_REF']

ENCODER_OUTPUT_SIZE = 256
LATENT_SIZE = 8
CONDITIONAL_FEATURES = len(x_time_invariant_cols)

INPUT_FEATURES = len(x_cols)





df = pd.read_csv(config_data["data_dir"],index_col=0).drop(columns=['lat', 'lon', 'elv'])
df = df[df.index != "AR-Vir"]
df = df[df.index != "CN-Cng"]
df = pd.get_dummies(df, columns=['classid', 'c4', 'koeppen_code', 'igbp_land_use', 'plant_functional_type'])
sites_df = batch_by_site(df)


n_features  = len(sites_df[0].columns)-1
cv_mse = []
cv_r2 = []
for s in tqdm(range(len(sites_df))):
  sites_to_train = list(range(0, len(sites_df)))
  sites_to_train.remove(s)
  site_to_test = [s]
  train = [sites_df[i] for i in sites_to_train]
  test = [sites_df[i] for i in site_to_test]
  c_train_dfs = pd.concat(train, ignore_index=True)
  c_test_dfs = pd.concat(test, ignore_index=True)

  X_train = [pd.concat([normalize(df[x_time_dep_cols]), df[x_time_invariant_cols]], axis=1).values for df in train]
  conditional_train = [df[x_time_invariant_cols].values for df in train]
  y_train = [normalize(df[y_col]).values for df in train]

  X_test = [pd.concat([normalize(df[x_time_dep_cols]), df[x_time_invariant_cols]], axis=1).values for df in test]
  conditional_test = [df[x_time_invariant_cols].values for df in test]
  y_test = [normalize(df[y_col]).values for df in test]


  encoder = EncoderWithTime(INPUT_FEATURES, ENCODER_OUTPUT_SIZE).to(DEVICE)
  reparam = Reparametrize(ENCODER_OUTPUT_SIZE, LATENT_SIZE).to(DEVICE)
  decoder = DecoderNoTime(LATENT_SIZE, INPUT_FEATURES, CONDITIONAL_FEATURES, CONDITION_DECODER).to(DEVICE)
  regressor = Regressor(LATENT_SIZE)
  model = Model(encoder, reparam, decoder, regressor).to(DEVICE)  
  optimizer = torch.optim.Adam(model.parameters())

  train_losses = []
  test_losses = []
  r2 = []
  for epoch in range(args.n_epochs):
      test_loss = 0.0
      test_r2= 0.0

      model.train()
      for (x, y, conditional) in zip(X_train, y_train, conditional_train):
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
          for (x, y, conditional) in zip(X_test, y_test, conditional_test):
            x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
            y = torch.FloatTensor(y).to(DEVICE)
            conditional = torch.FloatTensor(conditional).to(DEVICE)

            outputs, mean, logvar, y_pred = model(x, conditional)

            x = x.squeeze(1)
            
            loss, recon_loss, kl_loss, reg_loss = loss_fn(outputs, x, y_pred, y, mean, logvar, 1)
            r2.append(r2_score(y_true=y.detach().numpy(), y_pred=y_pred.detach().numpy()))



  cv_r2.append(max(r2))
  print(f"Test Site: {df.index.unique()[s]}  MSE: {cv_mse[s]} R2: {cv_r2[s]}")
  print("CV R2 cumulative mean: ", np.mean(cv_r2), " +- ", np.std(cv_r2))
  print("-------------------------------------------------------------------")
 