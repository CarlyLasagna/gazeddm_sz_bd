#!/usr/bin/env Rscript
#########################################################################
##
##   This code models choices/RTs on a YES/NO detection task using the 
##   Ratcliff (1978) drift diffusion model (DDM). Accuracy on this YES/NO 
##   detection task is determined by the response  (yes [1], no [2]) and 
##   condition (signal present [1], not present [2]) .
##
##   note:
##   for responses (resp): 1=upper bound resp 'YES'; 2=lower bound resp 'NO'
##   for conditions (cond; ie gaze condition): 1=signal present; 2=signal absent
##   for condition other; ie head and emo conditions)" 1=forward head + neutral emo, 2=forward head+fearful emo, 3=deviated head+neutral emo, 4=deviated head +fearful emo
##
########################################################################

library(cmdstanr)
#library(rstan)

rm(list=ls())     # clear environment
seed <- 42        # set random seed

#set TASK parameters
rtBound <- 0      # lowest rt we allow in seconds (e.g., if you threw out trials with RTs<200ms, this would be"2")

#set MCMC SAMPLER parameters
modelname <-'hddm_m10'# choose: hddm_m1, hddm_m2, hddm_m5,hddm_m6,hddm_m7,hddm_m8,hddm_m9,hddm_m10
warmup <- 2500        # number of warmup samples
cores <- 36           # number of cores to use (default=1)
iter <- 6000          # number of postwarmup samples per chain (
chains <- 36          # number of chains
adapt_delta <- 0.95   # default=.95 (target mean proposal acceptance probability during adaptation period)
max_treedepth <- 12   # default=10 (when max_treedepth reached, sampler terminates prematurely)
stepsize <- 1         # default=1 (discretization interval)

# set PATH to parent directory
fitpath <- '/nfs/turbo/lsa-clasagna/gazeddm/'
beh_datafile <- 'gaze_beh.csv' #gaze_beh.csv = actual data, example_beh_data.csv = fake data w/ same structure as real data

options(mc.cores = cores)

#############################################################################

#source relevant scripts
scriptpath<-paste(fitpath,"scripts/",sep="")
setwd(scriptpath)
file.sources = list.files(pattern="*.R")
sapply(file.sources,source,.GlobalEnv)

#number of postwarmup draws
postwarmup <-chains*(iter-warmup)

#create directory where outputs will be saved
dirname=paste(fitpath,"output/",sep="") 
if (file.exists(dirname) == FALSE){dir.create(dirname)} #if output directory doesn't exist, create it
  
#load RESPONSE data (raw trial level data should be saved in ./data directory as .csv)
datapath=paste(fitpath,"data/",beh_datafile,sep="") 
data<-read.csv(datapath)

N_cond<-length(unique(data$cond))
data$subj<-match(data$subj, unique(data$subj)) #turn subj id into sequential indexes 
N_groups<-length(unique(data$group))
data$group<-match(data$group, unique(data$group)) #turn group # into sequential indexes 
N_subj<-length(unique(data$subj)) #get number of subjects
N_cond_other<-length(unique(data$cond_other_emo_head)) #

if (max(data$rt)>100){data$rt<-as.numeric(data$rt)/1000} #if rt col is in ms, convert to seconds

#calculate min rt's for each subject
mintimes <- rep(0,N_subj)
for (j in 1:N_subj){ #get min times for each subject
  tmp <- subset(data, data$subj == j)
  mintimes[j]<-min(tmp$rt)
}

# prep data for stan 
data_stan<-list(
  N_obs=dim(data)[1], # number of observations [integer]
  N_subj=N_subj,# Number of subjects [integer]
  N_groups=N_groups,# Number of groups [integer]
  N_cond=length(unique(data$cond)), #number of conditions (determines resp accuracy)
  N_cond_other=length(unique(data$cond_other_emo_head)), #number of other task conditions (does NOT determine resp accuracy) - incl. just head orientation for this model
  N_choice=length(unique(data$choice)), #number of choices [integer]
  RT=data$rt, # RT for each observation [vector of doubles of length 'N_obs']
  choice=data$choice, # choice for each observation [integer vector of length 'N_obs']
  cond=data$cond, # gaze condition for each observation [integer vector of length 'N_obs']; this is the task condition that determines whether responses are correct or not
  cond_other=data$cond_other_emo_head, #other within-subj task conditions (gaze x emo) for each obs. [integer vector of length 'N_obs']; 1=neutral direct 2=neutral indirect, 3=fearful direct, 4=fearful indirect.
  subj=data$subj, # subject for each observation [integer vector length 'N_obs']
  group=data$group, # group for each observation [integer vector of length 'N_obs']
  minRT=as.numeric(mintimes),# minimum RT for each subject [vector of doubles of length 'N_subjs']
  rtBound=rtBound) # lower RT boundary that was allowed 

# define name of model you want to run
stanmodelname=paste(fitpath,"scripts/",modelname,'.stan',sep="") 

#create directory where outputs will be saved
modelpath=paste(fitpath,"output/",modelname,"/",sep="") 
if (file.exists(modelpath) == FALSE){dir.create(modelpath)} #if output directory doesn't exist, create it

# run the model
print("Running model in cmdstan") 

model <- cmdstan_model(stanmodelname)

fit <- model$sample(data=data_stan, 
                    iter_warmup=warmup, 
                    iter_sampling=iter,
                    init=0, 
                    chains=chains, 
                    parallel_chains=chains, #num cores
                    save_warmup = FALSE,
                    adapt_delta=adapt_delta,
                    max_treedepth=max_treedepth,
                    step_size=stepsize,
                    refresh=100,
                    seed=seed,
                    output_dir=modelpath,
                    diagnostics = c("divergences", "treedepth", "ebfmi"))

fitname<-paste(modelpath,"stanfit",".RData",sep = "") #save workspace as .RData file
fit$save_object(fitname)

