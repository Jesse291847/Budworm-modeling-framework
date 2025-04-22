# This script generate Figures 8 and F1
# setup social simulation
source("social_phenomena_setup.R")

# load helper function for network simulation
source("helper_functions.R")


# we repeat this simulation 100 times, disruption the ties of a random agent each time
saved_data <- list()

# since this simulation takes very long an alternative is to not run the for loop
# instead an example dataset can be loaded with:
saved_data <- readRDS("sim_data/disruption_social.RDS") %>% do.call(what = cbind) 


#ONLY RUN this loop if you did not load in the sim_data
# for (i in 1:100) {
#   
#   # specific settings for this simulation
#   k_base <- runif(n, 1, 5)
#   beta <-  0.2
#   kappa <- 0.2
#   
#   # update parameters
#   p <- parms <- c(k_base, b_base, a)
#   names(p) <- names(parms) <- c(rep("k_base", n), rep("b_base", n), rep("a", n))
#   
#   
#   data <- run(tmax = ntime, timeplot = FALSE, after = "distance <- outer(state[which(names(state) == \"N\")], state[which(names(state) == \"N\")], FUN = function(x, y) abs(x - y))
#     net <- matrix(state[which(names(state) == \"network\")], n ,n )
#     
#     state[which(names(state) == \"network\")] <- as.vector(utility_network(net, distance, h_l = n*0.1));
#             if(t == 100) {heavy_user <- sample(which(state[which(names(state) == \"N\")] > 6), 1)
#                           net[heavy_user,] <- net[,heavy_user] <- 0
#                           state[which(names(state) == \"network\")] <- as.vector(net)
#             }"
#               ,table = TRUE)
#   
#   dat1 <- data[,which(names(data) == "N")]
#   
#   network <- matrix(as.numeric(data[101, which(names(data) == "network")]), n, n)
#   
#   saved_data[[i]] <- dat1[,which(rowSums(network) == 0)]
#   
# }

# hand picked example trajectories of three scenarios (these will depend on seed)
examples <- data.frame(relapse = saved_data[,12], delayed = saved_data[,82], recovery = saved_data[,22], 
                       time = 1:nrow(saved_data))

#Generate Figure 9
fig_9 <- examples %>% ggplot(aes(x = time)) + 
  geom_line(aes(y = relapse, color = "Relapse"), linewidth = 1.2, linetype = 1) + 
  geom_line(aes(y = recovery, color = "Recovery"), linewidth = 1.2, linetype = 2) + 
  geom_line(aes(y = delayed, color = "Delayed relapse"), linewidth = 1.2, linetype = 3) +
  scale_color_manual(values = c("Relapse" = "#880D1E", "Recovery" = "#C18203", 
                                "Delayed relapse" = "#1C110A")) +
  labs(y = "Consumption", x = "Time") + 
  theme_classic() +
  theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text = element_text(size = 12), 
    legend.text = element_text(size = 13),
    legend.title = element_blank(),
    legend.key.size = unit(1.2, "lines"),  # Adjust legend key size
    legend.position = "top",
    legend.box.margin = margin(6, 6, 6, 6),
    legend.background = element_rect(fill = NA, color = "#C18203")
  ) 

# Generate Figure F1
figure_F1 <- saved_data %>% 
  tail(1) %>% 
  as.vector() %>% 
  hist(breaks = seq(0, 10, 0.5), 
       main = "Distribution of Consumption at the End of the Simulation", 
       xlab = "Consumption")


