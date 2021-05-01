from model.model import EncoderWithTime, Reparametrize, DecoderNoTime, Regressor, Model
from sklearn.metrics import r2_score
from utils.preprocess_veg import prepare_df, normalize
from model.loss import loss_fn
from tqdm import tqdm
import json
import torch
import pandas as pd
import argparse
import numpy as np
import faulthandler

faulthandler.enable()
# Parse arguments 
parser = argparse.ArgumentParser(description='CV wavenet')

parser.add_argument('-gpu', '--gpu', default=0, type=str,
                      help='indices of GPU to enable ')

parser.add_argument('-e', '--n_epochs', default=10, type=int,
                      help='number of cv epochs (wavenet)')

parser.add_argument('-d', '--latent_dim', default=16, type=int,
                      help='latent dim')

args = parser.parse_args()
DEVICE = torch.device("cuda:4" )

#-gpu 2 -e 30 -d 36
# Load Configs
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)


data = pd.read_csv(config_data["data_dir"], index_col=0).drop(columns=['lat', 'lon', 'elv','date','c4','whc'])
# Drop AR-Vir and CN-Cng
data=data[data.index!= "AR-Vir"]
data=data[data.index!= "CN-Cng"]

veg_type=data['classid'].unique()
veg_type=veg_type[veg_type!='CSH']

for veg in veg_type:
    veg_df = data[data.classid == veg]

    df_sensor, df_meta, df_gpp = prepare_df(veg_df)

    ENCODER_OUTPUT_SIZE = 256
    LATENT_SIZE = args.latent_dim
    CONDITIONAL_FEATURES = len(df_meta[0].columns)
    CONDITION_DECODER = True
    INPUT_FEATURES = len(df_sensor[0].columns) + CONDITIONAL_FEATURES
    if len(df_sensor)< 2:
        continue
    

    cv_r2=[]

    for s in range(len(df_sensor)):

      sites_to_train = list(range(len(df_sensor)))
      sites_to_train.remove(s)
      sites_to_test = [s]

      x_train = [pd.concat([df_sensor[i],df_meta[i]],axis=1).values for i in sites_to_train]
      conditional_train = [df_meta[i].values for i in sites_to_train]
      y_train = [df_gpp[i].values.reshape(-1,1) for i in sites_to_train]



      x_test = [pd.concat([df_sensor[i],df_meta[i]],axis=1).values for i in sites_to_test]
      conditional_test = [df_meta[i].values for i in sites_to_test]
      y_test = [df_gpp[i].values.reshape(-1,1) for i in sites_to_test]

      encoder = EncoderWithTime(INPUT_FEATURES, ENCODER_OUTPUT_SIZE).to(DEVICE)
      reparam = Reparametrize(ENCODER_OUTPUT_SIZE, LATENT_SIZE).to(DEVICE)
      decoder = DecoderNoTime(LATENT_SIZE, INPUT_FEATURES, CONDITIONAL_FEATURES, CONDITION_DECODER).to(DEVICE)
      regressor = Regressor(LATENT_SIZE)
      model = Model(encoder, reparam, decoder, regressor).to(DEVICE)


      optimizer = torch.optim.Adam(model.parameters())
      r2 = []

      for epoch in range(args.n_epochs):
        model.train()
        train_loss = 0.0
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
#             print('train')
        

        model.eval()
        with torch.no_grad():
            for (x, y, conditional) in zip(x_test, y_test, conditional_test):
                x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
                y = torch.FloatTensor(y).to(DEVICE)
                conditional = torch.FloatTensor(conditional).to(DEVICE)

                outputs, mean, logvar, y_pred = model(x, conditional)

                x = x.squeeze(1)

                loss, recon_loss, kl_loss, reg_loss = loss_fn(outputs, x, y_pred, y, mean, logvar, 1)
                r2.append(r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy()))
                print('test')
#       cv_r2.append(max(r2))
#     print(f"Classif: {veg}", np.mean(cv_r2), " +- ", np.std(cv_r2))
#     print("-------------------------------------------------------------------")
