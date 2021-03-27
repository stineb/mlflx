library(Amelia)
data_file <- "data/ddf_combined_mlflx_20210323.csv"
df <- read.csv(data_file)
keep_cols <- c("sitename", "date", "TA_F", "SW_IN_F", "LW_IN_F", "VPD_F", "PA_F",
               "P_F", "WS_F", "USTAR", "CO2_F_MDS", "GPP_NT_VUT_REF")
df <- df[keep_cols]

idvars <- c("sitename", "date")
ncpus = 8
m = ncpus * 10      # 80 sets, see Bagging
df.out = amelia(df, idvars = idvars, m = m, parallel = 'multicore', ncpus = ncpus)

write.amelia(df.out, file.stem = "data_imp")
