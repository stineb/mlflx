#' Calibrates SOFUN model parameters
#'
#' This is the main function that handles the 
#' calibration of SOFUN model parameters. 
#' 
#' @param drivers asdf
#' @param obs A data frame containing observational data used for model
#'  calibration. Created by function \code{get_obs_calib2()}
#' @param settings A list containing model calibration settings. 
#'  See vignette_rsofun.pdf for more information and examples.
#'
#' @return A complemented named list containing 
#'  the calibration settings and optimised parameter values.
#' @export
#' @importFrom magrittr %>%
#' @import GenSA BayesianTools
#' 
#' @examples 
#' \dontrun{ 
#' calib_sofun(
#'   drivers = filter(drivers,
#'           sitename %in% settings$sitenames),
#'   obs = obs_calib,
#'   settings = settings)
#' }

lsocv_sofun <- function(
  drivers,
  obs,
  settings
){
  
  ## get full list of sites
  sites <- drivers$sitename %>% unique()
  
  ## for each site, train at remaining sites and evaluate at this site
  purrr::map(
    as.list(sites),
    ~lsocv_sofun_bysite(., drivers, obs, settings)
  )
  
}

lsocv_sofun_bysite <- function(
  thissite,
  drivers,
  obs,
  settings
  ){
  
  ## calibrate at all but this site
  pars <- suppressWarnings(
    calib_sofun(
      drivers = p_model_fluxnet_drivers %>% 
        dplyr::filter(sitename != thissite),  
      obs = df_calib %>% 
        dplyr::filter(sitename != thissite),
      settings = settings
    )
  )

  ## update parameters  
  params_modl <- list(
    kphio           = pars$par[1],
    soilm_par_a     = pars$par[2],
    soilm_par_b     = pars$par[3],
    tau_acclim_tempstress = pars$par[4],
    par_shape_tempstress  = pars$par[5]
  )
  
  ## predict at this site
  output <- rsofun::runread_pmodel_f(
    p_model_fluxnet_drivers %>% 
      dplyr::filter(sitename == thissite),
    par = params_modl
  )
  
  ## create reduced data frame
  df_modobs <- df_calib %>% 
    dplyr::filter(sitename == thissite) %>% 
    unnest(data) %>% 
    left_join(
      output %>% 
        unnest(data) %>% 
        dplyr::select(sitename, date, mod = gpp),
      by = c("sitename", "date")
    )
  
  return(df_modobs)  
}