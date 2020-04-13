data {
  // Data matrix
  int<lower = 0>        N_row;
  int<lower = 0>        N_col;
  matrix[N_row, N_col]  y;

  // LFO parameter
  int<lower=0> L; // How many time steps are we estimating with
}
parameters{
  // Hierarchical parameters
  real              mean_i;
  real<lower=0>     sd_i;
  row_vector[N_col] beta_i_raw;  // Intercept

  matrix[N_col, N_col] beta_c;   // Parameters for VAR(1)
}
transformed parameters{
  row_vector[N_col]    beta_i; // Individual intercepts

  for (c in 1:N_col)
  {
    beta_i[c]   = mean_i + sd_i * beta_i_raw[c];
  }
}
model{
  // Priors
  for (c in 1:N_col)
  {
    beta_c[c,:] ~ std_normal();
  }

  beta_i_raw ~ std_normal();
  mean_i     ~ std_normal();
  sd_i       ~ cauchy(0, 1);

  // Model
  for (t in 2:L)
  {
    y[t,:] ~ normal(beta_i + y[t-1,:]*beta_c, 1);
  }
}
generated quantities {
  real y_ll[N_row, N_col];           // The log likelihood for each individual data point
  real y_hat[N_row+1, N_col];        // The posterior predictive distribution
  row_vector[N_col] y_mean[N_row+1]; // The posterior predictive distribution of the mean of y

  for (t in 2:(N_row+1))
  {
    y_mean[t,:] = beta_i + y[t-1,:]*beta_c;       // Mean of the observations
    y_hat[t,:]  = normal_rng(y_mean[t,:], 1); // Sample from the observations


    if (t <= N_row) // The log-lik should be done for all but the first and the predicted one
    {
      for (c in 1:N_col)
      {
        y_ll[t,c] = normal_lpdf(y[t,c] | y_mean[t,c], 1); // Evaluating single log-likelihoods for PSIS-LOO
      }
    }
  }
}
