library(Grind)

# optional: set number of timepoints and number of agents
ntime <- 30 # defaults to 300 if not specified
n <- 20 # defaults to 100 if not specified 

# setup a the full model with parameters
source("social_phenomena_setup.R")

# optional: change parameters by adding/substracting or redefining distribution

# option 1
b_base <- b_base + 0.3

# option 2
a <- runif(n, 0.5, 1)


# update the changed parameters
p <- parms <- c(k_base, b_base, a)
names(p) <- names(parms) <- c(rep("k_base", n), rep("b_base", n), rep("a", n))

# run simulation
res <- run(tmax = ntime, timeplot = FALSE, table = TRUE)

# extract consumption data
dat <- res[,which(names(res) == "N")]

# plot the consumption data
matplot(dat, type = "l", ylab = "Consumption", xlab = "Time")


# # possible experiment: increase A
# a <- runif(n, 1, 2)
# 
# # update the changed parameters
# p <- parms <- c(k_base, b_base, a)
# names(p) <- names(parms) <- c(rep("k_base", n), rep("b_base", n), rep("a", n))
# 
# 
# # run simulation
# res <- run(tmax = ntime, timeplot = FALSE, table = TRUE)
# 
# # extract consumption data
# dat <- res[,which(names(res) == "N")]
# 
# # plot the consumption data: More escalation to heavy use!
# matplot(dat, type = "l", ylab = "Consumption")














