library(Grind)

# define model
budworm <- function(t, state, parms) {
  with(as.list(c(state,parms)), {
    dN <- r * N * (1-N/k) - (b*N^2)/(a^2+N^2)
    return(list(c(dN)))
  }) }
model <- budworm

# try different parameter settings here 
parms <- p <-  c(r = 1.1 , b = 3, a = 1, k = 10)

# set starting values
state <- s <- c(N = 0.1)

# run model
run(tmax = 30, table = TRUE, timeplot = TRUE)

# example to try: set b to 1
parms <- p <-  c(r = 1.1 , b = 1, a = 1, k = 10)

# we see the that the system now changes to a different stable state
run(tmax = 30, table = TRUE, timeplot = TRUE)





