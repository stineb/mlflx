from model.model_alex import Encoder, Decoder, Reparametrize,Model, loss_fn
from preprocess import prepare_df
from sklearn.metrics import r2_score
import torch
import pandas as pd
import argparse
import torch.nn.functional as F
import numpy as np
import operator
from plotly import graph_objects as go


# Parse arguments 
parser = argparse.ArgumentParser(description='CV LSTM')

parser.add_argument('-gpu', '--gpu', default=None, type=str,
                      help='indices of GPU to enable ')

parser.add_argument('-e', '--n_epochs', default=None, type=int,
                      help='number of cv epochs ()')


args = parser.parse_args()
DEVICE = torch.device("cuda:" + args.gpu)
torch.manual_seed(40)
data = pd.read_csv('utils/df_imputed.csv', index_col=0)
data = data.drop(columns='date')
raw = pd.read_csv('../data/df_20210510.csv', index_col=0)['GPP_NT_VUT_REF']
raw = raw[raw.index != 'CN-Cng']

df_sensor, df_meta, df_gpp = prepare_df(data)
sites = raw.index.unique()
masks = []
for s in sites:
    mask = raw[raw.index == s].isna().values
    masks.append(list(map(operator.not_, mask)))
    
    
cv_r2 = []
cv_pred = []
for s in range(len(df_sensor)):
    sites_to_train = list(range(len(df_sensor)))
    sites_to_train.remove(s)
    sites_to_test = [s]
    x_train = [pd.concat((df_sensor[i], df_meta[i]), axis=1).values for i in sites_to_train]
    conditional_train = [df_meta[i].values for i in sites_to_train]
    y_train = [df_gpp[i].values.reshape(-1,1) for i in sites_to_train]

    x_test = [pd.concat((df_sensor[i],df_meta[i]), axis=1).values for i in sites_to_test]
    conditional_test = [df_meta[i].values for i in sites_to_test]
    y_test = [df_gpp[i].values.reshape(-1,1) for i in sites_to_test]
    
    ENCODER_INPUT_DIM = x_train[0].shape[1]
    ENCODER_OUTPUT_DIM = 16
    REPARAM_INPUT_DIM = ENCODER_OUTPUT_DIM
    LATENT_DIM = 32
    REPARAM_OUTPUT_DIM = LATENT_DIM
    DECODER_INPUT_DIM = LATENT_DIM + conditional_train[0].shape[1]
    DECODER_OUTPUT_DIM = len(df_sensor[0].columns)

    encoder = Encoder(ENCODER_INPUT_DIM, ENCODER_OUTPUT_DIM).to(DEVICE)
    decoder = Decoder(DECODER_INPUT_DIM, DECODER_OUTPUT_DIM).to(DEVICE)
    reparam = Reparametrize(REPARAM_INPUT_DIM, REPARAM_OUTPUT_DIM).to(DEVICE)
    model = Model(encoder, decoder, reparam).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    r2 = []
    
    for epoch in range(args.n_epochs):
        model.train()
        for (x, c, y) in zip(x_train, conditional_train, y_train):
            # Convert to tensors
            x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
            c = torch.FloatTensor(c).to(DEVICE)
            y = torch.FloatTensor(y).to(DEVICE)
            # Get predictions
            out, mean, logvar, pz_mean, r_mean, r_logvar = model(x, c)
                
            # Remove the conditional from x
            x = x.squeeze(1)[:, :len(df_sensor[0].columns)]

            # Get loss and update
            optimizer.zero_grad()
            loss, kl_loss, recon_loss, label_loss = loss_fn(out, x, mean, logvar, 1, pz_mean, r_mean, r_logvar, y)
            loss.backward()
            optimizer.step()
         
        model.eval()
        with torch.no_grad():
                for (x, c, y) in zip(x_test, conditional_test, y_test):
                    # Convert to tensors
                    x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
                    c = torch.FloatTensor(c).to(DEVICE)
                    y = torch.FloatTensor(y).to(DEVICE)

                    # Get predictions
                    out, mean, logvar, pz_mean, r_mean, r_logvar = model(x, c)

                    # Remove the conditional from x
                    x = x.squeeze(1)[:, :len(df_sensor[0].columns)]

                    # Get loss
                    loss, kl_loss, recon_loss, label_loss = loss_fn(out, x, mean, logvar, 1, pz_mean, r_mean, r_logvar, y)
                    test_r2 = r2_score(y.detach().cpu().numpy()[masks[s]], r_mean.detach().cpu().numpy()[masks[s]])
                    r2.append(test_r2)
    
    cv_r2.append(max(r2))
    print(f"Test Site: {s} R2: {cv_r2[s]}")
    print("CV R2 cumulative mean: ", np.mean(cv_r2), " +- ", np.std(cv_r2))
    print("-------------------------------------------------------------------")
    
fig = go.Figure()
fig.add_trace(go.Histogram(x=cv_r2,nbinsx=20))
fig.add_vline(x=np.mean(cv_r2))
fig.write_html("histogram_alex_cv.html")
