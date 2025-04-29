# set number of timepoints and number of agents
ntime <- 1000
n <- 100

source("social_phenomena_setup.R")

# setup parameters
k_base <- runif(n, 3.00001, 8.00001) 
k_base <- runif(n, 3.00001, 8.00001) 

kappa <- 0.2
beta <- 0.1

# set b_base high: norms against consumption
b_base <- b_base + 2 


# get parameters ready for Grind
p <- parms <- c(k_base, b_base, a,kappa,beta)
names(p) <- names(parms) <- c(rep("k_base", n), rep("b_base", n), rep("a", n),"kappa","beta")

# set by how much b_base will decrease
d_x <- -2.3/ntime*4

# run model (alternatively the data can be loaded in from the sim_data folder)
data <- run(tmax = ntime, after = "distance <- outer(state[which(names(state) == \"N\")],
 state[which(names(state) == \"N\")], FUN = function(x, y) abs(x - y))
    net <- matrix(state[which(names(state) == \"network\")], n ,n )
    state[which(names(state) == \"network\")] <- as.vector(utility_network(net, distance, h_l = n*0.1));
  if(t > ntime/4 & t < 2*ntime/4) parms[which(names(parms) == \"b_base\")] <- parms[which(names(parms) == \"b_base\")]+d_x
     if(t > 2*ntime/4 & t < 3*ntime/4) parms[which(names(parms) == \"b_base\")] <- parms[which(names(parms) == \"b_base\")]-d_x" ,
            table = TRUE,timeplot = FALSE)

# extract the consumption data
dat1 <- data[,which(names(data) == "N")]


# create the plot showing b
up=(ntime/4):(2*ntime/4)
down=(2*ntime/4):(3*ntime/4)
means=apply(dat1[c((ntime/4):(2*ntime/4),(2*ntime/4):(3*ntime/4)),],1,mean)
x=mean(b_base) + d_x*c(1:length(up),length(down):1)



b_plot <- ggplot(data = data.frame(x, time = c(up, down))) + 
  geom_line(aes(y = x, x = time), color = "#1C110A", linewidth = 1) +
  theme_classic() + 
  labs(y = "Average B base")


# extract the starting network and the final network
colors <- get_colors(dat1)


# end and start network
networks  <- list()
for(i in c(max(up), max(down))) {
  networks[[i]] <- matrix(as.numeric(data[i, which(names(data) == "network")]), n, n)
}

pre_network <- matrix(as.numeric(data[min(up), which(names(data) == "network")]), n, n)
post_network <- matrix(as.numeric(data[max(down), which(names(data) == "network")]), n, n)


network_1 <- qgraph(pre_network, color = colors[min(up),], esize = 0.5)
network_2 <- qgraph(post_network, color = colors[max(down),],esize = 0.5 )

# create plot with consumption data
plot_1 <- dat1 %>%
  as.data.frame() %>%
  gather(variable, value) %>%
  mutate(x = rep(1:(n()/length(unique(variable))), times = length(unique(variable)))) %>%
  ggplot(aes(x = x, y = value, group = variable)) +
  geom_line(color = "#1C110A") +
  theme_classic()