library(Grind)
library(tidyverse)
library(patchwork)


# define model
budworm <- function(t, state, parms) {
  with(as.list(c(state,parms)), {
    r <- parms[1:3] 
    N <- state[1:3]
    dN <- r * N * (1-N/k) - (b*N^2)/(a^2+N^2)
    return(list(c(dN)))
  }) }
model <- budworm

# choose different values for sensitivity to consumption r
r_values <- c(0.3, 0.45, 0.7)

# set b and a to 0, mimicking the absence of control
parms <- p <-  c(r_values, b = 0, a = 0, k = 10)

# set starting values
state <- s <- c(0.1,0.1,0.1)

# run model
dat1 <- run(tmax = 30, tstep = 0.01, table = TRUE, timeplot = FALSE)
colnames(dat1) <- c("time", "low_r", "medium_r", "high_r")

# plot for different values r for left panel
panel_1 <- dat1 %>% ggplot(mapping = aes(x = time)) + 
  geom_line(aes(y = low_r, color = "low_r"), linewidth = 1.2) + 
  geom_line(aes(y = medium_r, color = "medium_r"), linewidth = 1.2) + 
  geom_line(aes(y = high_r, color = "high_r"), linewidth = 1.2) +
  geom_hline(yintercept = 10, linetype = 2, color = "#A663CC", linewidth = 1.5) +
  scale_color_manual(values = c("low_r" = "#880D1E", "medium_r" = "#C18203", 
                                "high_r" = "#1C110A"), 
                     labels = c("low_r" = "r = 0.3", "medium_r" = "r = 0.45", 
                                "high_r" = "r = 0.7"),   breaks = c("low_r", "medium_r", "high_r") ) + 
  labs(y = "Consumption", x = "Time (t)") + 
  theme_classic() +
  theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text = element_text(size = 12), 
    legend.text = element_text(size = 16),
    legend.title = element_blank(),
    legend.key.size = unit(1, "lines"),  
    legend.position = c(0.8, 0.15),
    legend.box.margin = margin(6, 6, 6, 6),
    legend.background = element_rect(fill = NA, color = "#C18203") 
  )  +
  annotate("text", x = 0.5, y = 9.5, label = "K", color ="#A663CC", size = 8)

# plot control for different values of consumption

# define range of values for N
N_values <- seq(0, 5, by = 0.001)

# define range of values for A
A_values <- c(0.5, 1, 1.5)


# define function to go from N to control
control <- function(N, B = 1, A = 1) {
  (B*N^2)/(A^2 + N^2)
}

y <- cbind(N_values, "y1" = control(N_values, A = 0.5),  "y2" = control(N_values, A = 1), 
           "y3" = control(N_values, A = 1.5)) 


            
panel_2 <- ggplot(y, aes(x = N_values)) +
  geom_line(aes(y = y1, color = "A = 0.5"), size = 1) +
  geom_line(aes(y = y2, color = "A = 1"), size = 1) +
  geom_line(aes(y = y3, color = "A = 1.5"), size = 1) + 
  scale_color_manual(values = c("#1C110A", "#C18203","#880D1E"))  +        
  labs(x = "Consumption", y = "Control") +
  theme_classic() +
  geom_hline(yintercept = 1, linetype = 2, color = "#A663CC", linewidth = 1.5) +
  theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text = element_text(size = 12), 
    legend.text = element_text(size = 16),
    legend.title = element_blank(),
    legend.key.size = unit(1, "lines"),  
    legend.position = c(0.8, 0.15),
    legend.box.margin = margin(6, 6, 6, 6),
    legend.background = element_rect(fill = NA, color = "#C18203")) +
  annotate("text", x = 0.1, y = 0.95, label = "B", color ="#A663CC", size = 8) 
 


# generate Figure 2
panel_1 + panel_2


# ggsave("combined_panels.pdf", panel_1 + panel_2, width = 12, height = 6)


