from torch.utils.data import Dataset
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, x, y, conditional):
        super(MyDataset, self).__init__()
        self.x = np.array(x, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)
        self.conditional = np.array(conditional, dtype=np.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index, :], self.y[index], self.conditional[index, :]


def batch_by_site(df):
    sites = df.index.unique()
    sites_df = [df[df.index.isin([site])] for site in sites]
    for i in range(len(sites_df)):
        sites_df[i]['date'] = pd.to_datetime(sites_df[i]['date'], format="%Y-%m-%d")
        sites_df[i] = sites_df[i].set_index("date")
        sites_df[i] = sites_df[i].reset_index(drop=True)
    return sites_df

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name != 'GPP_NT_VUT_REF':
          result[feature_name] = (df[feature_name] - df[feature_name].mean()) / df[feature_name].std()
    return result
