# This script generates Figure 9 

# setup social simulation
source("social_phenomena_setup.R")

# load helper function for network simulation
source("helper_functions.R")

# specific settings for legal substance
k_base <- runif(n, 3, 8) # high base availability
kappa <- 0.05 # low social influence on K

# get parameters ready for Grind
p <- parms <- c(k_base, b_base, a)
names(p) <- names(parms) <- c(rep("k_base", n), rep("b_base", n), rep("a", n))

# run the simulation
data <- run(tmax = ntime, after = "distance <- outer(state[which(names(state) == \"N\")], state[which(names(state) == \"N\")], FUN = function(x, y) abs(x - y))
    net <- matrix(state[which(names(state) == \"network\")],n, n)
    state[which(names(state) == \"network\")] <- as.vector(utility_network(net, distance, h_l = n*0.1))"
            ,table = TRUE)

# extract consumption data
dat1 <- data[,which(names(data) == "N")]

# generate colors according to consumption status
colors <- get_colors(dat1)

# extract all networks
networks  <- list()
for(i in 1:ntime){
  networks[[i]] <- matrix(as.numeric(data[i, which(names(data) == "network")]), n, n)
}

# save the final network for a legal substance
net_legal <- qgraph(networks[[300]], color = colors[300,], esize = 0.5, title = "Legal Substance", title.cex = 1.4)

# change the parameters to an illegal substance
k_base <- runif(n, 0, 1) # low base availability
kappa <- 0.3 # strong social influence on availability

# get parameters ready for Grind
p <- parms <- c(k_base, b_base, a)
names(p) <- names(parms) <- c(rep("k_base", n), rep("b_base", n), rep("a", n))


# run simulation
data2 <- run(tmax = ntime, after = "distance <- outer(state[which(names(state) == \"N\")], 
state[which(names(state) == \"N\")], FUN = function(x, y) abs(x - y))
    net <- matrix(state[which(names(state) == \"network\")], n ,n )
    
    state[which(names(state) == \"network\")] <- as.vector(utility_network(net, distance, h_l = n*0.1))"
            ,table = TRUE)

# extract consumption data
dat2 <- data2[,which(names(data2) == "N")]

colors2 <- get_colors(dat2)
networks2  <- list()
for(i in 1:ntime){
  networks2[[i]] <- matrix(as.numeric(data2[i, which(names(data2) == "network")]), n, n)
}

net_illegal <- qgraph(networks2[[300]], color = colors2[300,], esize = 0.5, title = "Illegal Substance", title.cex = 1.4)

# create Figure 

pdf("figures/plot_legal_illegal.pdf", height = 5, width = 10)
par(mfrow = c(1,2))
plot(net_legal)
plot(net_illegal)
dev.off()











