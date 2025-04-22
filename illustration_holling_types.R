library(ggplot2)
library(dplyr)

Holling_all <- function(N, B = 1, A = 1, c = 2) {
  
  B*N^c/(A^2 + N^2)
}


c_values <- seq(2, 1, by = -0.2)
N_values <- seq(0, 5, by = 0.01)

df <- expand.grid(N = N_values, c = c_values) %>%
  mutate(Control = Holling_all(N, c = c))

# This looks like the colors in Maarten's plot
gold_black_palette <- c("#1C110A", "#C18203","#880D1E" , "#A663CC", "#BFDBF7", "#0B5351")


fig_A1 <- ggplot(df, aes(x = N, y = Control, color = factor(c))) +
  geom_line(size = 1) +
  scale_color_manual(values = gold_black_palette, labels = paste("c =", c_values)) +
  labs(
    title = "",
    x = "Consumption (N)",
    y = "Control",
    color = "'Holling parameter'"
  ) + theme_classic() 
 
