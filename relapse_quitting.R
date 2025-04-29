library(Grind)

# number of scenarios
n <- 3

# define model
budworm <- function(t, state, parms) {
  with(as.list(c(state,parms)), {
    
    r <- state[which(names(state) == "r")]
    N <- state[which(names(state) == "N")]
    
    dN <- r * N * (1-N/k) - (b*N^2)/(a^2+N^2)
    dr <- r_growth * N - r_decay * (r - r_base) 
    
    return(list(c(dN, dr)))
  }) }
model <- budworm

# parameter settings
parms <- p <- c(b = 2, a = 1, k = 10)

r_base <- c(0.6, 0.7, 0.8) # different starting values for r
r_growth <- 0.0006
r_decay <- 0.0025

# define inital states
s <- state <- c(rep(8, n), r_base+0.3)
names(s) <- names(state) <- c(rep("N", n), rep("r", n))


# run model with temporary surpression of K
dat <- run(tmax = 800, table = TRUE, timeplot = FALSE, after = "if(t > 200 & t < 500) parms[3] <- 0.01;
    if(t > 500 ) parms[3] <- 10")


# plot consumption with 3 scenarios
matplot(dat[,which(names(dat) == "N")], type = "l")
