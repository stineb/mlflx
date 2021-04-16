clean_le <- function(df){
  
  df <- df %>% 
    mutate(LE_F_MDS_raw = LE_F_MDS)
  
  ## clean based on qc only if qc is available
  if (any(!is.na(df$LE_F_MDS_QC)) && ("LE_F_MDS_QC" %in% names(df))){
    df <- df %>% 
      mutate(LE_F_MDS = ifelse(LE_F_MDS_QC > 0.6, LE_F_MDS, NA))
  }
  
  ## remove spurious values
  df <- df %>% 
    mutate(LE_F_MDS = identify_pattern(LE_F_MDS))
  
  return(df)
}

identify_pattern <- function( vec ){
  
  spurious_values <- vec %>% 
    table() %>% 
    as_tibble() %>% 
    setNames(c("value", "n")) %>% 
    arrange(desc(n)) %>% 
    dplyr::filter(n>1) %>% 
    pull(value) %>% 
    as.numeric()
  
  vec[vec %in% spurious_values] <- NA
  
  return( vec )
  
}