// STAN MODEL 1
// This program defines a hierarchical drift diffusion model (Ratcliffe 1978), 
// with a non-centered parameterization. YES responses are 
// modeled as upper boundary responses and NO responses are modeled as
// lower boundary responses. In this version, drift rate (delta),
// threshold separation (alpha), start point (beta), and NDT are fixed across
// both task conditions (signal present, signal absent). 
functions{
  
  // wiener RNG using stochastic differential equation
  row_vector wiener1_rng(int cond, real alpha, real ndt, real beta, real delta) {
    real h;
    real sigma;
    real z;
    real i;
    real theEvidence;
    real theTime;
    real noise;
    real choice;
    real rt;
    row_vector[2] rand_trial;
    h = .001; //step size
    sigma = 1; //drift scaling coefficient
    z = beta * alpha;  // absolute starting point
    i = 1; //iteration to start at
    theEvidence = z; //begin accumulating evidence @ absolute start point
    while(theEvidence<=alpha && theEvidence>=0){ //while evidence is between 0 and alpha
      i=i+1;
      noise = normal_rng(0,1); //sample rand noise from standard normal dist 
      theEvidence = theEvidence + (delta*h+(sigma*sqrt(h))*noise);
    } // exit once evidence passes either boundary
    
    // define the choice (1 <- Yes (upper) 2 <- No (lower))
    if(cond==1){//if signal = present
      if(theEvidence>=alpha){ 
      choice = 1;
      }else{
        choice = 2;
      }
    }else{//if signal = absent
      if(theEvidence>=alpha){ 
      choice = 2;
      }else{
        choice = 1;
      }
    }
    
    
    rand_trial[1] = choice;
    rand_trial[2] = i*h+ndt; // define the decision time
    return rand_trial;
  }
  // wiener RNG using rejection based method
  row_vector wiener2_rng(int cond, real alpha, real ndt, real beta, real delta) {
    real h = .001;//step size
    real sigma = 1;//drift scaling coefficient
    real z = beta * alpha;  // absolute starting point;
    real i = 1; //iteration to start at
    real rand;//rand value from uniform dist
    real choice;
    real rt;
    row_vector[2] rand_trial;
    //real theTime=0;
    real theEvidence = z; //begin accumulating evidence @ absolute start point
    real p = .5 * (1+((delta*sqrt(h))/sigma)); //probability cutoff
    while(theEvidence < alpha && theEvidence > 0){
      rand = uniform_rng(0,1);
      if(rand <= p){
        theEvidence = theEvidence + sigma*sqrt(h);
      }else{
        theEvidence = theEvidence - sigma*sqrt(h);
      }
      i=i+1;
    }
  
    // define the choice (1 <- Yes (upper) 2 <- No (lower))
    if(cond==1){//if signal = present
      if(theEvidence>=alpha){ 
      choice = 1;
      }else{
        choice = 2;
      }
    }else{//if signal = absent
      if(theEvidence>=alpha){ 
      choice = 2;
      }else{
        choice = 1;
      }
    }
    
    rand_trial[1] = choice;
    rand_trial[2] = i*h+ndt;// define the decision time
    return rand_trial;
  }
}
data {
  int N_obs;          // number of observations [single integer]                           
  int N_subj;         // number of subjects [single integer]
  int N_cond;         // number of conditions [single integer]. **Not used here                       
  int N_cond_other;   // number of additional task conditions [single integer] that don't determine accuracy of responses responses
  int N_choice;       // number of choice alternatives [single integer]
  int N_groups;       // number of groups [single integer]
  vector[N_obs] RT;   // RT in seconds for each trial [vector of reals; length N_obs]
  int subj[N_obs];    // subj id for each trial [integer vector; length N_obs]
  int choice[N_obs];  // response for each trial [integer vector; length N_obs]
  int cond[N_obs];    // cond id for each trial [integer vector; length N_obs] **Not used here  
  int cond_other[N_obs];// cond for other within-subjects task conditions [integer vector; length N_obs]
  int group[N_obs];   // group id for each trial [integer vector; length N_obs]
  real minRT[N_subj]; // minimum RT for each subject [vector of reals; length N_subj]
  real rtBound;       // lowest rt allowed [single number] 
}
transformed data {
  int subj_group[N_subj];   // gives a vector of group id's at the subject-level (as opposed to the observation level as in 'group' above)
  for (i in 1:N_obs){
    subj_group[subj[i]]=group[i];
  }
}

parameters { 
  
  // GROUP-level parameters
  vector[N_groups] mu_grp_alpha_pr;           // threshold sep. group mean, all conditions
  vector[N_groups] mu_grp_beta_pr;            // start point group mean, all conditions
  vector[N_groups] mu_grp_delta_pr;           // drift rate group mean, all conditions
  vector[N_groups] mu_grp_ndt_pr;             // non-decision time group mean in sec, all conditions
  
  vector<lower=0>[N_groups] sig_grp_alpha_pr; // threshold sep. group SD, all conditions
  vector<lower=0>[N_groups] sig_grp_beta_pr;  // start point group SD, all conditions
  vector<lower=0>[N_groups] sig_grp_delta_pr; // drift rate group SD, all conditions
  vector<lower=0>[N_groups] sig_grp_ndt_pr;   // non-decision time group SD, all conditions
  
  //SUBJECT-level parameters
  vector[N_subj] sub_alpha_pr;  // threshold sep. subject mean, all conditions
  vector[N_subj] sub_beta_pr;   // start point subject mean, all conditions
  vector[N_subj] sub_delta_pr;  // drift rate subject mean, all conditions
  vector[N_subj] sub_ndt_pr;    // non-decision time subject mean in sec, all conditions
}

transformed parameters { 
  
  // SUBJECT-level transformed pars for non-centered parameterization
  vector<lower=0,upper=4>[N_subj] sub_alpha;        // threshold sep. TRANSFORMED subject mean, all conditions
  vector<lower=0,upper=1>[N_subj] sub_beta;         // start point TRANSFORMED subject mean, all conditions
  vector<lower=-4,upper=4>[N_subj] sub_delta;       // drift rate TRANSFORMED subject mean, all conditions
  vector<lower=rtBound,upper=max(minRT)*0.9>[N_subj] sub_ndt;   // non-decision time in sec TRANSFORMED subject mean, all conditions
  
  // loop through all observations (subject parameter scaled by group mean/sigma, phi transformed (std normal cdf), and re-scaled)
  for (i in 1:N_subj) { 
    // threshold sep. bound between 0.1 and 3.9
    sub_alpha[i] = 0.1 + 3.9 * Phi(mu_grp_alpha_pr[subj_group[i]] + sig_grp_alpha_pr[subj_group[i]] * sub_alpha_pr[i]); 
    // start point bound between 0 and 1 (< 0.5 lower boundary bias; >0.5 upper boundary bias)
    sub_beta[i] = Phi(mu_grp_beta_pr[subj_group[i]] + sig_grp_beta_pr[subj_group[i]] * sub_beta_pr[i]);             
    // drift rate bound between -4 and 4 (<0 for lower boundary responses, >0 for upper boundary responses)
    sub_delta[i] = -4 + 8 * Phi(mu_grp_delta_pr[subj_group[i]] + sig_grp_delta_pr[subj_group[i]] * sub_delta_pr[i]);
    // non-decision time (in seconds) bound between lower RT boundary and subject's fastest RT
    sub_ndt[i] = ((minRT[i]*0.9 - rtBound) * Phi(mu_grp_ndt_pr[subj_group[i]] + sig_grp_ndt_pr[subj_group[i]] * sub_ndt_pr[i]))+rtBound;    
  }
}

model {
  
  // GROUP-level hyperpriors
  mu_grp_alpha_pr ~ normal(0, 1);   // prior on threshold sep group mean, all conditions
  mu_grp_beta_pr ~ normal(0, 1);    // prior on start point group mean, all conditions
  mu_grp_delta_pr ~ normal(0, 1);   // prior on drift rate group mean, all conditions
  mu_grp_ndt_pr ~ normal(0, 1);     // prior on NDT group mean, all conditions
  
  sig_grp_alpha_pr ~ normal(0, .2); // prior on threshold sep group SD, all conditions
  sig_grp_beta_pr ~ normal(0, .2);  // prior on start point group SD, all conditions
  sig_grp_delta_pr ~ normal(0, .2); // prior on drift rate group SD, all conditions
  sig_grp_ndt_pr ~ normal(0, .2);   // prior on NDT group SD, all conditions
  
  //SUBJECT-level priors
  sub_alpha_pr ~ normal(0, 1);      // prior on untransformed threshold sep subj mean, all conditions
  sub_beta_pr  ~ normal(0, 1);      // prior on untransformed start point subj mean, all conditions
  sub_delta_pr  ~ normal(0, 1);     // prior on untransformed drift rate subj mean, all conditions
  sub_ndt_pr  ~ normal(0, 1);       // prior on untransformed NDT subj mean, all conditions
  
  // loop through observations
  for (i in 1:N_obs){ 
    if(choice[i]==1){ //if response = YES 
      RT[i] ~ wiener(sub_alpha[subj[i]], sub_ndt[subj[i]], sub_beta[subj[i]], sub_delta[subj[i]]);
    } else { //if response is NO, invert beta and delta
      RT[i] ~ wiener(sub_alpha[subj[i]], sub_ndt[subj[i]], 1-sub_beta[subj[i]], -sub_delta[subj[i]]);
    }
  }
}

generated quantities {
  vector[N_obs] log_lik = rep_vector(0, N_obs); // log liklihood for each observation
  
  // GROUP-level transformed parameters
  vector<lower=0,upper=4>[N_groups] mu_alpha = 0.1 + 3.9*Phi(mu_grp_alpha_pr); // threshold sep group mean, all conditions
  vector<lower=0,upper=1>[N_groups] mu_beta = Phi(mu_grp_beta_pr);            // start point group mean, all conditions
  vector<lower=-4,upper=4>[N_groups] mu_delta = -4 + 8*Phi(mu_grp_delta_pr);   // drift rate group mean, all conditions
  vector<lower=rtBound, upper=max(minRT)*0.9>[N_groups] mu_ndt = ((max(minRT)*0.9 - rtBound)*Phi(mu_grp_ndt_pr))+rtBound; // NDT group mean in sec, all conditions
  
  { // local section. calculate log_lik for each subject
    for (i in 1:N_obs){ 
      if(choice[i]==1){ // if response = YES
        log_lik[i] += wiener_lpdf(RT[i] | sub_alpha[subj[i]], sub_ndt[subj[i]], sub_beta[subj[i]], sub_delta[subj[i]]);
      } else { // if response = NO, invert beta and delta
        log_lik[i] += wiener_lpdf(RT[i] | sub_alpha[subj[i]], sub_ndt[subj[i]], 1-sub_beta[subj[i]], -sub_delta[subj[i]]);
      }
    }
  }
}