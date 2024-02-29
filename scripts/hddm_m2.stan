// STAN MODEL 2
// This program defines a hierarchical drift diffusion model (Ratcliffe 1978), 
// with a non-centered parameterization. YES responses are 
// modeled as upper boundary responses and NO responses are modeled as
// lower boundary responses. In this version, drift rate (delta) is allowed to 
// vary by task condition (signal present, signal absent). But the start point 
// (beta), threshold sep (alpha), and NDT are NOT allowed to vary by condition.

functions{
  
  // wiener RNG using stochastic differential equation
  vector wiener1_rng(int cond, real alpha, real ndt, real beta, real delta) {
    real h;
    real sigma;
    real z;
    real i;
    real theEvidence;
    real theTime;
    real noise;
    real choice;
    real rt;
    vector[2] rand_trial;
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
  vector wiener2_rng(int cond, real alpha, real ndt, real beta, real delta) {
    real h = .001;//step size
    real sigma = 1;//drift scaling coefficient
    real z = beta * alpha;  // absolute starting point;
    real i = 1; //iteration to start at
    real rand;//rand value from uniform dist
    real choice;
    real rt;
    vector[2] rand_trial;
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
  int N_cond;         // number of conditions [single integer] that determine accuracy of responses
  int N_cond_other;   // number of additional task conditions [single integer] that don't determine accuracy of responses responses
  int N_choice;       // number of choice alternatives [single integer]
  int N_groups;       // number of groups [single integer]
  vector[N_obs] RT;   // RT in seconds for each trial [vector of reals; length N_obs]
  int subj[N_obs];    // subj id for each trial [integer vector; length N_obs]
  int choice[N_obs];  // response for each trial [integer vector; length N_obs]
  int cond[N_obs];      // cond id for each trial [integer vector; length N_obs]
  int cond_other[N_obs];// cond for other within-subjects task conditions [integer vector; length N_obs]
  int group[N_obs];     // group id for each trial [integer vector; length N_obs]
  real minRT[N_subj];   // minimum RT for each subject [vector of reals; length N_subj]
  real rtBound;         // lowest rt allowed [single number] 
}
transformed data {
  int subj_group[N_subj];   // gives a vector of group id's at the subject-level (as opposed to the observation level as occurs with 'group' specified above)
  for (i in 1:N_obs){
    subj_group[subj[i]]=group[i];
  }
}
parameters { 
  
  // GROUP-level parameters
  vector[N_groups] mu_grp_alpha_pr;           // threshold sep. group mean, all conditions
  vector[N_groups] mu_grp_beta_pr;            // start point group mean, all conditions
  vector[N_groups] mu_grp_delta_present_pr;   // drift rate group mean, stim present
  vector[N_groups] mu_grp_delta_absent_pr;    // drift rate group mean, stim absent
  vector[N_groups] mu_grp_ndt_pr;             // non-decision time group mean in sec, all conditions
  
  vector<lower=0>[N_groups] sig_grp_alpha_pr; // threshold sep. group SD, all conditions
  vector<lower=0>[N_groups] sig_grp_beta_pr;  // start point group SD, all conditions
  vector<lower=0>[N_groups] sig_grp_delta_pr; // drift rate group SD, all conditions
  vector<lower=0>[N_groups] sig_grp_ndt_pr;   // non-decision time group SD, all conditions
  
  //SUBJECT-level parameters
  vector[N_subj] sub_alpha_pr;                // threshold sep. subject mean, all conditions
  vector[N_subj] sub_beta_pr;                 // start point subject mean, all conditions
  vector[N_subj] sub_delta_present_pr;        // drift rate subject mean, stim present
  vector[N_subj] sub_delta_absent_pr;         // drift rate subject mean, stim absent
  vector[N_subj] sub_ndt_pr;                  // non-decision time subject mean in sec, all conditions
}

transformed parameters { 
  
  // SUBJECT-level transformed pars for non-centered parameterization
  vector<lower=0,upper=4>[N_subj] sub_alpha;                // threshold sep. TRANSFORMED subject mean, all conditions
  vector<lower=0,upper=1>[N_subj] sub_beta;                 // start point TRANSFORMED subject mean, all conditions
  vector<lower=-4,upper=4>[N_subj] sub_delta_present;       // drift rate TRANSFORMED subject mean, stim present
  vector<lower=-4,upper=4>[N_subj] sub_delta_absent;        // drift rate TRANSFORMED subject mean, stim absent
  vector<lower=rtBound,upper=max(minRT)*0.9>[N_subj] sub_ndt;   // non-decision time in sec TRANSFORMED subject mean, all conditions
  
  // loop through all observations (phi transform (std normal cdf) subj param scaled by group mean/sigma, and re-scale)
  for (i in 1:N_obs) { 
    // threshold sep. bound between 0.1 and 3.9
    sub_alpha[subj[i]] = 0.1 + 3.9 * Phi(mu_grp_alpha_pr[group[i]] + sig_grp_alpha_pr[group[i]] * sub_alpha_pr[subj[i]]); 
    // start point bound between 0 and 1 (< 0.5 lower boundary bias; >0.5 upper boundary bias)
    sub_beta[subj[i]] = Phi(mu_grp_beta_pr[group[i]] + sig_grp_beta_pr[group[i]] * sub_beta_pr[subj[i]]);             
    // drift rate bound between -4 and 4 (<0 for lower boundary responses, >0 for upper boundary responses)
    sub_delta_present[subj[i]] = -4 + 8 * Phi(mu_grp_delta_present_pr[group[i]] + sig_grp_delta_pr[group[i]] * sub_delta_present_pr[subj[i]]);
    sub_delta_absent[subj[i]] = -4 + 8 *  Phi(mu_grp_delta_absent_pr[group[i]] + sig_grp_delta_pr[group[i]] * sub_delta_absent_pr[subj[i]]);
    // non-decision time (in seconds) bound between lower RT boundary and subject's fastest RT
    sub_ndt[subj[i]] = ((minRT[subj[i]]*0.9 - rtBound) * Phi(mu_grp_ndt_pr[group[i]] + sig_grp_ndt_pr[group[i]] * sub_ndt_pr[subj[i]]))+rtBound;    
  }
}

model {
  
  // GROUP-level hyperpriors
  mu_grp_alpha_pr ~ normal(0, 1);           // prior on threshold sep group mean, all conditions
  mu_grp_beta_pr ~ normal(0, 1);            // prior on start point group mean, all conditions
  mu_grp_delta_present_pr ~ normal(0, 1);   // prior on drift rate group mean, stim present
  mu_grp_delta_absent_pr ~ normal(0, 1);    // prior on drift rate group mean, stim absent
  mu_grp_ndt_pr ~ normal(0, 1);             // prior on NDT group mean, all conditions
  
  sig_grp_alpha_pr ~ normal(0, .2);         // prior on threshold sep group SD, all conditions
  sig_grp_beta_pr ~ normal(0, .2);          // prior on start point group SD, all conditions
  sig_grp_delta_pr ~ normal(0, .2);         // prior on drift rate group SD, all conditions
  sig_grp_ndt_pr ~ normal(0, .2);           // prior on NDT group SD, all conditions
  
  //SUBJECT-level priors
  sub_alpha_pr ~ normal(0, 1);              // prior on untransformed threshold sep subj mean, all conditions
  sub_beta_pr  ~ normal(0, 1);              // prior on untransformed start point subj mean, all conditions
  sub_delta_present_pr  ~ normal(0, 1);     // prior on untransformed drift rate subj mean, stim present
  sub_delta_absent_pr  ~ normal(0, 1);      // prior on untransformed drift rate subj mean, stim absent
  sub_ndt_pr  ~ normal(0, 1);               // prior on untransformed NDT subj mean, all conditions
  
  // loop through observations
  for (i in 1:N_obs){ 
    if (cond[i]==1){ // if signal = present
      if(choice[i]==1){ //if response = YES (correct)
        RT[i] ~ wiener(sub_alpha[subj[i]], sub_ndt[subj[i]], sub_beta[subj[i]], sub_delta_present[subj[i]]);
      } else { //if response is NO (incorrect), invert beta and delta
        RT[i] ~ wiener(sub_alpha[subj[i]], sub_ndt[subj[i]], 1-sub_beta[subj[i]], -sub_delta_present[subj[i]]);
      }
    } else { // if signal = absent 
      if(choice[i]==1){ // if response = YES (incorrect)
        RT[i] ~ wiener(sub_alpha[subj[i]], sub_ndt[subj[i]], sub_beta[subj[i]], sub_delta_absent[subj[i]]);
      } else { // if response = NO (correct), invert beta and delta
        RT[i] ~ wiener(sub_alpha[subj[i]], sub_ndt[subj[i]], 1-sub_beta[subj[i]], -sub_delta_absent[subj[i]]);
      }
    }
  }
}

generated quantities {
  vector[N_obs] log_lik = rep_vector(0, N_obs); // log liklihood for each observation
  
  // GROUP-level transformed parameters
  vector<lower=0,upper=4>[N_groups] mu_alpha = 0.1 + 3.9*Phi(mu_grp_alpha_pr);         // threshold sep group mean, all conditions
  vector<lower=0,upper=1>[N_groups]  mu_beta = Phi(mu_grp_beta_pr);                    // start point group mean, all conditions
  vector<lower=-4,upper=4>[N_groups] mu_delta_present = -4 + 8*Phi(mu_grp_delta_present_pr);   // drift rate group mean, stim present
  vector<lower=-4,upper=4>[N_groups] mu_delta_absent = -4 + 8*Phi(mu_grp_delta_absent_pr);     // drift rate group mean, stim absent
  vector<lower=rtBound, upper=max(minRT)*0.9>[N_groups] mu_ndt = ((max(minRT)*0.9 - rtBound)*Phi(mu_grp_ndt_pr))+rtBound; // NDT group mean in sec, all conditions
  
  { // local section. calculate log_lik for each subject
    for (i in 1:N_obs){ 
      if(cond[i]==1){ // if signal = present
        if(choice[i]==1){ // if response = YES (correct)
          log_lik[i] += wiener_lpdf(RT[i] | sub_alpha[subj[i]], sub_ndt[subj[i]], sub_beta[subj[i]], sub_delta_present[subj[i]]);
        } else { // if response = NO (incorrect), invert beta and delta
          log_lik[i] += wiener_lpdf(RT[i] | sub_alpha[subj[i]], sub_ndt[subj[i]], 1-sub_beta[subj[i]], -sub_delta_present[subj[i]]);
        }
      } else{ // if signal = absent 
        if(choice[i]==1){ // if response = YES (incorrect)
          log_lik[i] += wiener_lpdf(RT[i] | sub_alpha[subj[i]], sub_ndt[subj[i]], sub_beta[subj[i]], sub_delta_absent[subj[i]]);
        } else { // if response = NO (correct), invert beta and delta
          log_lik[i] += wiener_lpdf(RT[i] | sub_alpha[subj[i]], sub_ndt[subj[i]], 1-sub_beta[subj[i]], -sub_delta_absent[subj[i]]);
        }
      }
    }
  }
}


