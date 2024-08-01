library(qgraph)
library(Grind)
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

n <- 100

# parameters that are not manipulated in any of the simulations can go here

h_b <- 1
t_b <- 0.1
k_physical <- 10
r_growth <- 0.0005
r_decay <- 0.003
r_base <- runif(n, 0.5, 1.5)

a <- runif(n, 0.8, 2.3)


#parameters that are manipulated throughout the simulations can go here
k_base <- runif(n, 0.000001, 0.0000011)
b_base <- runif(n, 0.3, 2.3 )
beta <- 0.1
kappa <- 0.3


#states: things for which you want to follow the progression

network <- matrix(sample(0:1, n*n, TRUE, prob = c(0.95, 0.05)), n,n)
network[upper.tri(network)] <- t(network)[upper.tri(network)]
diag(network) <- 0
s <- state <- c(rep(0.0001, n) ,r_base , as.vector(network))
names(s) <- names(state) <- c(rep("N", n), rep("r", n), rep("network", n*n))

p <- parms <- c(k_base, b_base, a)
names(p) <- names(parms) <- c(rep("k_base", n), rep("b_base", n), rep("a", n))


# addiction spreads
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

ntime <- 300
data <- run(tmax = ntime, after = "distance <- outer(state[which(names(state) == \"N\")], state[which(names(state) == \"N\")], FUN = function(x, y) abs(x - y))
    net <- matrix(state[which(names(state) == \"network\")], n ,n )
    
    state[which(names(state) == \"network\")] <- as.vector(utility_network(net, distance, h_l = n*0.1));
            if(t == 20) {parms[which(names(parms) == \"k_base\")][c(10, 15, 20)] <- 5}" , table = TRUE)

dat1 <- data[,which(names(data) == "N")]


colors <- get_colors(dat1)
networks  <- list()

for(i in 1:ntime){
  networks[[i]] <- matrix(as.numeric(data[i, which(names(data) == "network")]), n, n)
}

#just some random layout that is the same for all networks
av <- averageLayout(networks[1:30])


# pdf("epidemic_networks.pdf", width = 15, height = 5)
# par(mfrow = c(2,3))
# for (i in 1:6) {
#   qgraph(networks[[10]], color = colors[10,], layout = av, title = "t = 10", title.cex = 2, esize = 0.5)
#   qgraph(networks[[20]], color = colors[20,], layout = av, title = "t = 20", title.cex = 2, esize = 0.5 )
#   qgraph(networks[[50]], color = colors[50,], layout = av, title = "t = 50", title.cex = 2, esize = 0.5)
#   qgraph(networks[[100]], color = colors[75,], layout = av, title = "t = 75", title.cex = 2, esize = 0.5)
#   qgraph(networks[[150]], color = colors[100,], layout = av, title = "t = 100", title.cex = 2, esize = 0.5)
#   qgraph(networks[[200]], color = colors[150,], layout = av, title = "t = 150", title.cex = 2, esize = 0.5)}
# dev.off()


#legal - illegal
k_base <- runif(n, 3, 8)
kappa <- 0.05

#update parameters
p <- parms <- c(k_base, b_base, a)
names(p) <- names(parms) <- c(rep("k_base", n), rep("b_base", n), rep("a", n))


data <- run(tmax = ntime, after = "distance <- outer(state[which(names(state) == \"N\")], state[which(names(state) == \"N\")], FUN = function(x, y) abs(x - y))
    net <- matrix(state[which(names(state) == \"network\")], n ,n )
    
    state[which(names(state) == \"network\")] <- as.vector(utility_network(net, distance, h_l = n*0.1))"
           ,table = TRUE)

dat1 <- data[,which(names(data) == "N")]


colors <- get_colors(dat1)
networks  <- list()

for(i in 1:ntime){
  networks[[i]] <- matrix(as.numeric(data[i, which(names(data) == "network")]), n, n)
}
net_legal <- qgraph(networks[[300]], color = colors[300,], esize = 0.5, title = "Legal Substance")

#illegal
k_base <- runif(n, 0, 1)
kappa <- 0.25

#update parameters
p <- parms <- c(k_base, b_base, a)
names(p) <- names(parms) <- c(rep("k_base", n), rep("b_base", n), rep("a", n))
data <- run(tmax = ntime, after = "distance <- outer(state[which(names(state) == \"N\")], state[which(names(state) == \"N\")], FUN = function(x, y) abs(x - y))
    net <- matrix(state[which(names(state) == \"network\")], n ,n )
    
    state[which(names(state) == \"network\")] <- as.vector(utility_network(net, distance, h_l = n*0.1))"
            ,table = TRUE)

dat1 <- data[,which(names(data) == "N")]

matplot(dat1, type = "l")
colors <- get_colors(dat1)
networks  <- list()
for(i in 1:ntime){
  networks[[i]] <- matrix(as.numeric(data[i, which(names(data) == "network")]), n, n)
}


net_illegal <- qgraph(networks[[300]], color = colors[300,], esize = 0.5, title = "Illegal Substance")

# pdf("Illegal_vs_legal.pdf", width = 10, height= 5)
# par(mfrow = c(1,2))
# plot(net_legal)
# plot(net_illegal)
# dev.off()

saved_data <- list()
for (i in 1:100) {
#social network aids recovery settings
k_base <- runif(n, 1,5)
beta <-  0.2
kappa <- 0.2

#update parameters
p <- parms <- c(k_base, b_base, a)
names(p) <- names(parms) <- c(rep("k_base", n), rep("b_base", n), rep("a", n))


data <- run(tmax = ntime, timeplot = FALSE, after = "distance <- outer(state[which(names(state) == \"N\")], state[which(names(state) == \"N\")], FUN = function(x, y) abs(x - y))
    net <- matrix(state[which(names(state) == \"network\")], n ,n )
    
    state[which(names(state) == \"network\")] <- as.vector(utility_network(net, distance, h_l = n*0.1));
            if(t == 100) {heavy_user <- sample(which(state[which(names(state) == \"N\")] > 6), 1)
                          net[heavy_user,] <- net[,heavy_user] <- 0
                          state[which(names(state) == \"network\")] <- as.vector(net)
            }"
            ,table = TRUE)

dat1 <- data[,which(names(data) == "N")]

network <- matrix(as.numeric(data[101, which(names(data) == "network")]), n, n)

saved_data[[i]] <- dat1[,which(rowSums(network) == 0)]

}

#saveRDS(saved_data, "crucial_aid.RDS")