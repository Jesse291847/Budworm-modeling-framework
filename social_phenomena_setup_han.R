library(qgraph)
library(Grind)
library(tidyverse)
source("helper_functions.R")

Budworm <- function(t, state, parms) {
  with(as.list(c(state,parms)), {
    
    m <- matrix(state[which(names(state) == "network")], n, n)
    N <- state[which(names(state) == "N")]
    r <- state[which(names(state) == "r")]
    
    b_base <- parms[which(names(parms) == "b_base")]
    k_base <- parms[which(names(parms) == "k_base")]
    a <- parms[which(names(parms) == "a")]
    
    
    b <- b_base + beta * (1/(1+exp(((N) - h_b)/t_b))) %*% m
    k <- pmin(parms[which(names(parms) == "k_base")] + state[which(names(state) == "N")] %*% m * kappa, 10)
  
    dN <- r * N * (1-N/k) - (b*N^2)/(a^2+N^2)
    dr <- r_growth * N - r_decay * (r - r_base) 
    dnetwork <- rep(0, n*n) 
    
    return(list(c(dN, dr, dnetwork)))
  }) }
model <- Budworm

# number of agents
n <- 50

# number of time points
ntime <- 1000


#set.seed(1978)


# parameter settings
h_b <- 1
t_b <- 0.1
k_physical <- 10
r_growth <- 0.0005
r_decay <- 0.003
r_base <- runif(n, 0.5, 1.5)

# will only change in the disruption social network phenomenon
beta <- 0.1

a <- runif(n, 0.8, 2.3)
b_base <- runif(n, 0.3, 2.3)


# initial states
network <- matrix(sample(0:1, n*n, TRUE, prob = c(0.95, 0.05)), n,n)
network[upper.tri(network)] <- t(network)[upper.tri(network)]
diag(network) <- 0
#s <- state <- c(rep(0.0001, n), r_base , as.vector(network))
s <- state <- c(rep(.6, n), r_base , as.vector(network))
names(s) <- names(state) <- c(rep("N", n), rep("r", n), rep("network", n*n))



