from sklearn.impute import KNNImputer
import pandas as pd

data = pd.read_csv('../../data/df_20210510.csv', index_col=0).drop(columns=['lat', 'lon', 'elv','c4','whc','LE_F_MDS','NEE_VUT_REF'])

data = data[data.index != 'CN-Cng'] # missing meta-data info for this site

sites = data.index.unique()

# Impute TA_F_DAY (with TA_F and SW_IN_F)
df =  data[['TA_F','SW_IN_F','TA_F_DAY', 'TA_F_NIGHT']]
for s in sites:
    impute = KNNImputer()
    x = df[df.index == s].values
    x = impute.fit_transform(x)
    data.loc[data.index == s, 'TA_F_DAY'] = x[:,2]
    data.loc[data.index == s, 'TA_F_NIGHT'] = x[:,3]

# Impute GPP (with SW_IN_F, LW_IN_F, TA_F, WS_F, VPD_F, P_F, TA_F_DAY)
df =  data[['TA_F','SW_IN_F','TA_F_DAY', 'LW_IN_F','WS_F','P_F', 'VPD_F', 'GPP_NT_VUT_REF']]
for s in sites:
    impute = KNNImputer()
    x = df[df.index == s].values
    x = impute.fit_transform(x)
    data.loc[data.index == s, 'GPP_NT_VUT_REF'] = x[:,-1]
    
data.to_csv('df_imputed.csv')