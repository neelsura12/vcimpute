library(MASS)
library(devtools)
install_github("udellgroup/mixedgcImp")
library(mixedgcImp)
library(rvinecopulib)

n = 100
p = 2
rho = 0.25
missingness = 0.2

# generate missing data, an n x p matrix X with missingness%  --------------

Sigma = matrix(data=c(1, rho, rho, 1), ncol=2, nrow=2)
X = mvrnorm(n, mu=rep(0, 2), Sigma=Sigma)
X_obs = X
loc = sample(1:prod(n*p), size=floor(prod(n*p)*missingness))
X_obs[loc] = NA

# drop empty rows  ---------------------------------------------------------

idx = c()
for (i in 1:nrow(X_obs)) {
  if (!all(is.na(X_obs[i,]))) {
    idx = c(idx, i)
  }
}
X_obs = X_obs[idx,]
X = X[idx,]
n = nrow(X_obs)

# fit GC imputation ---------------------------------------------------------

fit = impute_mixedgc(X_obs)
err_imp = cal_mae_scaled(xhat=fit$Ximp, xobs=X_obs, xtrue=X)

# fit bivariate copula ------------------------------------------------------

U = matrix(0, n, p)
fn1 = ecdf(X_obs[,1])
U1 = fn1(X_obs[,1]) * n/(n+1)
fn2 = ecdf(X_obs[,2])
U2 = fn2(X_obs[,2]) * n/(n+1)
U = matrix(c(U1, U2), n,p)

fit = bicop(U, family_set="onepar", keep_data=TRUE)

# impute ---------------------------------------------------------------------

X_hat = X_obs
for (i in 1:n) {
  if (!is.na(U[i,1]) & is.na(U[i, 2]))
  {
    cond_mean_fix1 = matrix(c(rep(U[i,1], n), 1:n/n), n, p)
    prob = mean(predict(fit, cond_mean_fix1, "hfunc1"))
    cat(prob, quantile(X_obs[,2], prob, na.rm=TRUE), U[i,1], U[i,2], '\n')
  }
  #else if
  #{
    # TODO use hfunc2
  #}
}
