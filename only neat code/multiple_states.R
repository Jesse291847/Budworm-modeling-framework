library(Grind)
library(tidyverse)

set.seed(1978)

# define model
budworm <- function(t, state, parms) {
  with(as.list(c(state,parms)), {
    N <- state[1:n]
    
    r <- parms[which(names(parms) == "r")]
    a <- parms[which(names(parms) == "a")]
    b <- parms[which(names(parms) == "b")]
    k <- parms[which(names(parms) == "k")]
    
    dN <- r * N * (1-N/k) - (b*N^2)/(a^2+N^2)
    return(list(c(dN)))
  }) }
model <- budworm


# number of states to plot
n <- 8

# sample random parameters
r <- c(runif(n-1, 0.5, 1.3), 0)
k <- runif(n, 2, 8)
a <- runif(n, 0.5, 2)
b <- runif(n, 0.5, 1.5)


# set b and a to 0, mimicking the absence of control
parms <- p <-  c(r, k, a, b)
names(p) <- names(p) <- c(rep("r", n), rep("k", n), rep("a", n), rep("b", n))

# set starting values
state <- s <- c(0, runif(n-1, 2, 7))

# run the model from with differen states and parms
data <- run(tmax = 40, tstep = 0.1, table = TRUE, timeplot = FALSE)


# color palette for plotting
gold_black_palette <- rep(c("#1C110A", "#C18203","#880D1E" , "#A663CC", "#BFDBF7", "#0B5351"),2)

# Generate Figure 5
fig_5 <- data %>% gather(key = "p", value = "consumption", -time) %>%
  ggplot(mapping = aes(x = time, y = consumption)) +
  geom_line(aes(color = p), linewidth = 1) +
  scale_color_manual(values = gold_black_palette) +
  labs(
    title = "",
    x = "Time (t)",
    y = "Consumption") + 
  theme_classic() +
  theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text = element_text(size = 12),
    legend.position = "none")
  
ggsave(fig_5.pdf)

# define model with sensitization
budworm <- function(t, state, parms) {
  with(as.list(c(state,parms)), {
    dN <- r * N * (1-N/k) - (b*N^2)/(a^2+N^2)
    dr <- r_growth * N - r_decay * (r_base - r)
    return(list(c(dN, dr)))
  }) }
model <- budworm


state <- s <-c(N = 0.1, r = 0)
parms <- p <- c(r_base = 0, k = 10, b = 1, a = 3, r_growth = 0.005, r_decay = 0.03)

run()

