from model.model_notime import Model
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
sites = raw.index.unique()

masks = []
for s in sites:
    mask = raw[raw.index == s].isna().values
    masks.append(list(map(operator.not_, mask)))


df_sensor, df_meta, df_gpp = prepare_df(data)

INPUT_FEATURES = len(df_sensor[0].columns) 

cv_r2 = []
cv_pred = []
for s in range(len(df_sensor)):
    sites_to_train = list(range(len(df_sensor)))
    sites_to_train.remove(s)

    x_train = [df_sensor[i].values for i in sites_to_train]
    y_train = [df_gpp[i].values.reshape(-1,1) for i in sites_to_train]

    x_test = df_sensor[s].values 
    y_test = df_gpp[s].values.reshape(-1,1) 

    model = Model(INPUT_FEATURES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())

    r2 = []
    pred = []
    
    for epoch in range(args.n_epochs):
        train_loss = 0.0
        train_r2 = 0.0
        model.train()
        for (x, y) in zip(x_train, y_train):
            x = torch.FloatTensor(x).to(DEVICE)
            y = torch.FloatTensor(y).to(DEVICE)
            y_pred = model(x)
            optimizer.zero_grad()
            loss = F.mse_loss( y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_r2 += r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy())
        
        model.eval()
        with torch.no_grad():
                x = torch.FloatTensor(x_test).to(DEVICE)
                y = torch.FloatTensor(y_test).to(DEVICE)
                y_pred = model(x)
                test_loss = F.mse_loss( y_pred, y)
                test_r2 = r2_score(y_true=y.detach().cpu().numpy()[masks[s]], y_pred=y_pred.detach().cpu().numpy()[masks[s]])
                r2.append(test_r2)
                pred.append(y_pred)
    
    cv_r2.append(max(r2))
    cv_pred.append(pred[np.argmax(r2)])
    print(f"Test Site: {s} R2: {cv_r2[s]}")
    print("CV R2 cumulative mean: ", np.mean(cv_r2), " +- ", np.std(cv_r2))
    print("-------------------------------------------------------------------")
    
fig = go.Figure()
fig.add_trace(go.Histogram(x=cv_r2,nbinsx=20))
fig.add_vline(x=np.mean(cv_r2))
fig.write_html("histogram_cv_notime.html")