# This script generates Figures 7 and E1

# setup social simulation
source("social_phenomena_setup.R")

# load helper function for network simulation
source("helper_functions.R")


# specific settings for this simulation
k_base <- runif(n, 0.000001, 0.0000011) # k_base starts low
kappa <- 0.3 # strong social influence


# get parameters ready for Grind
p <- parms <- c(k_base, b_base, a)
names(p) <- names(parms) <- c(rep("k_base", n), rep("b_base", n), rep("a", n))


# run the simulation 
data <- run(tmax = ntime, after = "distance <- outer(state[which(names(state) == \"N\")], state[which(names(state) == \"N\")], FUN = function(x, y) abs(x - y))
    net <- matrix(state[which(names(state) == \"network\")], n ,n )
    
    state[which(names(state) == \"network\")] <- as.vector(utility_network(net, distance, h_l = n*0.1));
            if(t == 20) {parms[which(names(parms) == \"k_base\")][c(10, 15, 20)] <- 5};
            if(t == 150) {parms[which(names(parms) == \"k_base\")][c(10, 15, 20)] <- 0}" , table = TRUE)

# extract the consumption data
dat1 <- data[,which(names(data) == "N")]

# Generate Figure 8 
fig_8 <- data.frame(value = rowSums(dat1 > 5), time = 1:nrow(dat1)) %>%  ggplot(mapping = aes(x = time, y = value)) +
  geom_line(linewidth = 1.2, color =  "#C18203") +
  labs(y = "Number of people with N > 5", x = "Time") +
  geom_vline(xintercept = 20, linetype = "dotted", color = "darkred", linewidth = 1.5) + 
  annotate("text", x = 20, y = 40, label = "Intervention", angle = 90, vjust = -0.5, size = 8) +
  geom_vline(xintercept = 150, linetype = "dashed", color = "#1C110A", linewidth = 1.5) + 
  annotate("text", x = 150, y = 40, label = "Intervention reverted", angle = 90, vjust = -0.5, size = 6) +
  theme_classic() +
  theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text = element_text(size = 12)
  )

## to track the spread through the network over time
# get colors for the nodes based on their consumption status
colors <- get_colors(dat1)


# extract all networks from the data as a matrix
networks  <- list()
for(i in 1:ntime){
  networks[[i]] <- matrix(as.numeric(data[i, which(names(data) == "network")]), n, n)
}

# plot all networks with the same layout
av <- averageLayout(networks[1:30])

# Generate Figure E1 for the Appendix
# pdf("epidemic_networks.pdf", width = 15, height = 5)
par(mfrow = c(2,3))
for (i in 1:6) {
  qgraph(networks[[10]], color = colors[10,], layout = av, title = "t = 10", title.cex = 2, esize = 0.5)
  qgraph(networks[[20]], color = colors[20,], layout = av, title = "t = 20", title.cex = 2, esize = 0.5 )
  qgraph(networks[[50]], color = colors[50,], layout = av, title = "t = 50", title.cex = 2, esize = 0.5)
  qgraph(networks[[100]], color = colors[75,], layout = av, title = "t = 75", title.cex = 2, esize = 0.5)
  qgraph(networks[[150]], color = colors[100,], layout = av, title = "t = 100", title.cex = 2, esize = 0.5)
  qgraph(networks[[200]], color = colors[150,], layout = av, title = "t = 150", title.cex = 2, esize = 0.5)}
# dev.off()




