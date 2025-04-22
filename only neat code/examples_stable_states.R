library(Grind)
library(tidyverse)
library(patchwork)

a <- 0.67
c <- 0.38
d <- 0.25


get_fig_data <- function(r_setting) {

# number of states to plot
n <- 8

# parameters
p <- parms <- c(r = r_setting, k = 10, b = 1, a = 1)
s <- state <- seq(0.3, 6, length.out = 8)
  
# define model
budworm <- function(t, state, parms) {
  with(as.list(c(state,parms)), {
    N <- state[1:n]
    dN <- r * N * (1-N/k) - (b*N^2)/(a^2+N^2)
    return(list(c(dN)))
  }) }
model <- budworm

# simulate model
return(run(ode = model, state = s, parms = p, tmax = 25, tstep = 0.1, table = TRUE, timeplot = FALSE))

}

# run simulations
data_a <- get_fig_data(a)
data_c <- get_fig_data(c)
data_d <- get_fig_data(d)


plot_panel <- function(dat, panel = "a - High Consumption") {
  dat %>% gather(key = "variable", value = "value", -time) %>% 
    ggplot(mapping = aes(y = value, x = time, group = variable)) +
    geom_line(color = "#C18203", linetype = 2, linewidth = 1.1) + theme_classic() +
  labs(
    title = panel,
    x = "Time (t)",
    y = "N",
    color = "'Holling parameter'"
  ) +
    theme(plot.title = element_text(hjust = 0.5, size = 12))
}


# create the panels
p1 <- plot_panel(data_a)
p2 <- plot_panel(data_c, panel = "c - Bistable Consumption")
p3 <- plot_panel(data_d, panel = "d - Low Consumption")


# combine the panels into Figure 4
fig_4 <- p1 / p2/  p3








