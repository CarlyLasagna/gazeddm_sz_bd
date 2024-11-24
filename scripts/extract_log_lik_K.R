extract_log_lik_K <- function(list_of_stanfits, list_of_holdout){
  theK <- length(list_of_stanfits)
  list_of_log_liks <- plyr::llply(1:theK, function(k){
    extract_log_lik(list_of_stanfits[[k]])
  })
  # log_lik_heldout - will include the loglik of all the held out data of all the folds; (samples x N_obs) matrix
  log_lik_heldout <- list_of_log_liks[[1]] * NA
  for(k in 1:theK){
    log_lik <- list_of_log_liks[[k]]
    samples <- dim(log_lik)[1] 
    N_obs <- dim(log_lik)[2]
    # matrix of same size as log_lik_heldout (value = 1 if data was held out in fold k)
    heldout <- matrix(rep(list_of_holdout[[k]], each = samples), nrow = samples)
    # Sanity check that previous log_lik is not being overwritten:
    if(any(!is.na(log_lik_heldout[heldout==1]))){warning("Heldout log_lik has been overwritten!!!!")}
    # add the log_lik of fold k to the overall matrix:
    log_lik_heldout[heldout==1] <- log_lik[heldout==1]
  }
  return(log_lik_heldout)
}