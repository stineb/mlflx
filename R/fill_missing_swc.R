fill_missing_swc <- function(df){

  # df <- df %>% 
  #   mutate(LE_F_MDS_clean = LE_F_MDS, NEE_VUT_REF_clean = NEE_VUT_REF, GPP_NT_VUT_REF_clean = GPP_NT_VUT_REF)
  
  cont <- TRUE
  
  if (nrow(dplyr::filter(df, !is.na(wscal) & !is.na(SWC_F_MDS_1))) == 0){
    if (any(!is.na(df$wscal))){
      df <- df %>% 
        mutate(SWC_F_MDS_1 = wscal)
    } else {
      rlang::warn(paste("Removing entire site because no water balance data is available:", df$sitename[1], "\n"))
      df <- tibble()
      cont <- FALSE
    }
  } else {
    pp <- recipe(GPP_NT_VUT_REF ~ ., data = df) %>%
      step_impute_linear(SWC_F_MDS_1, impute_with = imp_vars(wscal)) %>%
      prep(training = df)
    df <- bake(pp, new_data = df)
  }

  if (cont){
    if (nrow(dplyr::filter(df, !is.na(SWC_F_MDS_2) & !is.na(SWC_F_MDS_1))) == 0){
      df <- df %>% 
        mutate(SWC_F_MDS_2 = SWC_F_MDS_1)
    } else {
      pp <- recipe(GPP_NT_VUT_REF ~ ., data = df) %>%
        step_impute_linear(SWC_F_MDS_2, impute_with = imp_vars(wscal, SWC_F_MDS_1)) %>%
        prep(training = df)
      df <- bake(pp, new_data = df)
    }
    
    if (nrow(dplyr::filter(df, !is.na(SWC_F_MDS_2) & !is.na(SWC_F_MDS_1) & !is.na(SWC_F_MDS_3))) == 0){
      df <- df %>% 
        mutate(SWC_F_MDS_3 = SWC_F_MDS_2)
    } else {
      pp <- recipe(GPP_NT_VUT_REF ~ ., data = df) %>%
        step_impute_linear(SWC_F_MDS_3, impute_with = imp_vars(wscal, SWC_F_MDS_1, SWC_F_MDS_2)) %>%
        prep(training = df)
      df <- bake(pp, new_data = df)
    }
    
    if (nrow(dplyr::filter(df, !is.na(SWC_F_MDS_2) & !is.na(SWC_F_MDS_1) & !is.na(SWC_F_MDS_3) & !is.na(SWC_F_MDS_4))) == 0){
      df <- df %>% 
        mutate(SWC_F_MDS_4 = SWC_F_MDS_3)
    } else {
      pp <- recipe(GPP_NT_VUT_REF ~ ., data = df) %>%
        step_impute_linear(SWC_F_MDS_4, impute_with = imp_vars(wscal, SWC_F_MDS_1, SWC_F_MDS_2, SWC_F_MDS_3)) %>%
        prep(training = df)
      df <- bake(pp, new_data = df)
    }
  }

  
  # pp <- recipe(GPP_NT_VUT_REF ~ ., data = df) %>% 
  #   step_impute_linear(SWC_F_MDS_4, impute_with = imp_vars(wscal, SWC_F_MDS_1, SWC_F_MDS_2, SWC_F_MDS_3)) %>%
  #   prep(training = df)
  # df <- bake(pp, new_data = df)
  # 
  # pp <- recipe(GPP_NT_VUT_REF ~ ., data = df) %>% 
  #   step_impute_linear(SWC_F_MDS_5, impute_with = imp_vars(wscal, SWC_F_MDS_1, SWC_F_MDS_2, SWC_F_MDS_3, SWC_F_MDS_4)) %>%
  #   prep(training = df)
  # df <- bake(pp, new_data = df)
  # 
  # pp <- recipe(GPP_NT_VUT_REF ~ ., data = df) %>% 
  #   step_impute_linear(SWC_F_MDS_6, impute_with = imp_vars(wscal, SWC_F_MDS_1, SWC_F_MDS_2, SWC_F_MDS_3, SWC_F_MDS_4, SWC_F_MDS_5)) %>%
  #   prep(training = df)
  # df <- bake(pp, new_data = df)
  # 
  # pp <- recipe(GPP_NT_VUT_REF ~ ., data = df) %>% 
  #   step_impute_linear(SWC_F_MDS_7, impute_with = imp_vars(wscal, SWC_F_MDS_1, SWC_F_MDS_2, SWC_F_MDS_3, SWC_F_MDS_4, SWC_F_MDS_5, SWC_F_MDS_6)) %>%
  #   prep(training = df)
  # df <- bake(pp, new_data = df)
  
  return(df)
}