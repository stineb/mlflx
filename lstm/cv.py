from model import Model
from preprocess import prepare_df
from sklearn.metrics import r2_score
import torch
import pandas as pd

# Parse arguments 
parser = argparse.ArgumentParser(description='CV LSTM')

parser.add_argument('-gpu', '--gpu', default=None, type=str,
                      help='indices of GPU to enable ')

parser.add_argument('-e', '--n_epochs', default=None, type=int,
                      help='number of cv epochs ()')

parser.add_argument('-c', '--conditional', default=True, type=bool,
                      help='enable conditioning')

args = parser.parse_args()
DEVICE = torch.device("cuda:" + args.gpu)
torch.manual_seed(40)
np.random.seed(40)

data = pd.read_csv('../data/df_final.csv', index_col=0).drop(columns=['lat', 'lon', 'elv','date','c4','whc'])
good_sites = pd.read_csv("../data/df_20210507.csv", low_memory=False )['sitename'].unique()

df_sensor, df_meta, df_gpp = prepare_df(data,sites=good_sites)

INPUT_FEATURES = len(df_sensor[0].columns) 
HIDDEN_DIM = 256
CONDITIONAL_FEATURES = len(df_meta[0].columns)

cv_r2 = []
cv_pred = []
for s in range(len(df_sensor)):
    sites_to_train = list(range(len(df_sensor)))
    sites_to_train.remove(s)
    sites_to_test = [s]

    x_train = [df_sensor[i].values for i in sites_to_train]
    conditional_train = [df_meta[i].values for i in sites_to_train]
    y_train = [df_gpp[i].values.reshape(-1,1) for i in sites_to_train]

    x_test = df_sensor[i].values for i in sites_to_test
    conditional_test = df_meta[i].values for i in sites_to_test
    y_test = df_gpp[i].values.reshape(-1,1) for i in sites_to_test

    model = Model(INPUT_FEATURES, CONDITIONAL_FEATURES, HIDDEN_DIM, arg.conditional).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())

    r2 = []
    pred = []
    
    for epoch in range(args.n_epochs):
        train_loss = 0.0
        train_r2 = 0.0
        model.train()
        for (x, y, conditional) in zip(x_train, y_train, conditional_train):
            x = torch.FloatTensor(x).to(DEVICE)
            y = torch.FloatTensor(y).to(DEVICE)
            c = torch.FloatTensor(conditional).to(DEVICE)
            
            y_pred = model(x, c)
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
                c = torch.FloatTensor(conditional_test).to(DEVICE)
                y_pred = model(x, c)
                test_loss = F.mse_loss( y_pred, y)
                test_r2 = r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy())
                r2.append(test_r2)
                pred.append(y_pred)
    
    cv_r2.append(max(r2))
    cv_pred.append(pred[argmax(r2)])
    print(f"Test Site: {s} R2: {cv_r2[s]}")
    print("CV R2 cumulative mean: ", np.mean(cv_r2), " +- ", np.std(cv_r2))
    print("-------------------------------------------------------------------")