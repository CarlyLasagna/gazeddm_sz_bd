########################################################################
# This script (from hbayesdm toolbox (Ahn et al 2014)) calculates a HDI for a 
# vector of MCMC samples and a user-specified credible interval.
########################################################################
# INPUT VARIABLES:
# sampleVec =  vector of posterior mcmc samples for a given parameter
# credMass = credible interval (e.g., .95 or .9)
########################################################################
HDIofMCMC <- function(sampleVec,credMass = 0.95) {
  sortedPts = sort(sampleVec)
  ciIdxInc = floor(credMass * length(sortedPts))
  nCIs = length(sortedPts) - ciIdxInc
  ciWidth = rep(0 , nCIs)
  for (i in 1:nCIs) {
    ciWidth[i] = sortedPts[i + ciIdxInc] - sortedPts[i]
  }
  HDImin = sortedPts[which.min(ciWidth)]
  HDImax = sortedPts[which.min(ciWidth) + ciIdxInc]
  HDIlim = c(HDImin , HDImax)
  return(HDIlim)
}