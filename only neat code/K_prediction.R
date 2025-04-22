# Change default settings
n <- 100
ntime <- 100
# setup social simulation
source("social_phenomena_setup.R")

# load helper function for network simulation
source("helper_functions.R")


k_base <- runif(n, 0, 0.00011) # low base availability
kappa <- 0.3 # strong social influence on availability



# get parameters ready for Grind
p <- parms <- c(k_base, b_base, a,kappa,beta)
names(p) <- names(parms) <- c(rep("k_base", n), rep("b_base", n), rep("a", n),"kappa","beta")

# the first number determines how much the variable will change maximum
d_x <- 5/ntime/2

d_x <- 0
data <- run(tmax = ntime, after = "distance <- outer(state[which(names(state) == \"N\")],
 state[which(names(state) == \"N\")], FUN = function(x, y) abs(x - y))
    net <- matrix(state[which(names(state) == \"network\")], n ,n )
    state[which(names(state) == \"network\")] <- as.vector(utility_network(net, distance, h_l = n*0.1));
  if(t > ntime/4 & t < 2*ntime/4) parms[which(names(parms) == \"k_base\")] <- parms[which(names(parms) == \"k_base\")]+d_x
     if(t > 2*ntime/4 & t < 3*ntime/4) parms[which(names(parms) == \"k_base\")] <- parms[which(names(parms) == \"k_base\")]-d_x" ,
            table = TRUE,timeplot = T)

dat1 <- data[,which(names(data) == "N")]


k_base <- runif(n, 0, 1)
# get parameters ready for Grind
p <- parms <- c(k_base, b_base, a,kappa,beta)
names(p) <- names(parms) <- c(rep("k_base", n), rep("b_base", n), rep("a", n), "kappa", "beta")

data2 <- run(tmax = ntime, after = "distance <- outer(state[which(names(state) == \"N\")],
 state[which(names(state) == \"N\")], FUN = function(x, y) abs(x - y))
    net <- matrix(state[which(names(state) == \"network\")], n ,n )
    state[which(names(state) == \"network\")] <- as.vector(utility_network(net, distance, h_l = n*0.1));
  if(t > ntime/4 & t < 2*ntime/4) parms[which(names(parms) == \"k_base\")] <- parms[which(names(parms) == \"k_base\")]+d_x
     if(t > 2*ntime/4 & t < 3*ntime/4) parms[which(names(parms) == \"k_base\")] <- parms[which(names(parms) == \"k_base\")]-d_x" ,
            table = TRUE,timeplot = T)

dat2 <- data[,which(names(data) == "N")]



b_base <- b_base * 2

# get parameters ready for Grind
p <- parms <- c(k_base, b_base, a,kappa,beta)
names(p) <- names(parms) <- c(rep("k_base", n), rep("b_base", n), rep("a", n),"kappa","beta")


data3 <- run(tmax = ntime, after = "distance <- outer(state[which(names(state) == \"N\")],
 state[which(names(state) == \"N\")], FUN = function(x, y) abs(x - y))
    net <- matrix(state[which(names(state) == \"network\")], n ,n )
    state[which(names(state) == \"network\")] <- as.vector(utility_network(net, distance, h_l = n*0.1));
  if(t > ntime/4 & t < 2*ntime/4) parms[which(names(parms) == \"k_base\")] <- parms[which(names(parms) == \"k_base\")]+d_x
     if(t > 2*ntime/4 & t < 3*ntime/4) parms[which(names(parms) == \"k_base\")] <- parms[which(names(parms) == \"k_base\")]-d_x" ,
            table = TRUE,timeplot = T)

dat3 <- data[,which(names(data) == "N")]
