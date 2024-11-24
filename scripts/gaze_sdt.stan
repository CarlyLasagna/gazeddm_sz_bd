// Equal variance gaussian SDT model

data {
  int nWCon; // Number of conditions
  int nSub;  // Number of subjects
  array[nWCon,nSub] int s; // Number of signal trials
  array[nWCon,nSub] int n; // Number of noise trials
  array[nWCon,nSub] int h; // Number of hits
  array[nWCon,nSub] int f; // Number of false alarms
}

parameters {
  vector[nWCon] muc; //mu criterion
  vector[nWCon] mud; //mu discriminability
  real<lower=0> lambdac; //sigma criterion
  real<lower=0> lambdad; //sigma discriminability
  
  array[nWCon,nSub] real c; //criterion
  array[nWCon,nSub] real d; //discriminability
}

transformed parameters {
  real sigmac = 1 / sqrt(lambdac); //convert to precision
  real sigmad = 1 / sqrt(lambdad); //convert to precision
}

model {
  
  // Hyperpriors
  for (i in 1:nWCon) {
    muc[i] ~ normal(0, 0.001);
    mud[i] ~ normal(0, 0.001);
  }

  lambdac ~ gamma(0.001, 0.001);
  lambdad ~ gamma(0.001, 0.001);

  // Main model
  for (i in 1:nWCon) {
    for (j in 1:nSub) {
      
      // Priors
      real thetah;
      real thetaf;
      
      c[i,j] ~ normal(muc[i],lambdac);
      d[i,j] ~ normal(mud[i],lambdad);

      // Reparameterization using equal variance Gaussian SDT
      thetah = Phi(d[i,j] / 2 - c[i,j]);
      thetaf = Phi(-d[i,j] / 2 - c[i,j]);

      // Likelihood
      h[i, j] ~ binomial(s[i, j],thetah);
      f[i, j] ~ binomial(n[i, j],thetaf);

    }
  }
}

