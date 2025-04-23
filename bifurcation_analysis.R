library(rootSolve)
library(dplyr)
library(tidyr)
library(ggplot2)
library(grid)
library(patchwork)

# Parameters
A <- 1
K <- 1

# Define ranges for r and q
r_vals <- seq(0.15, .8, length.out = 1000) # we used 1000 for the manuscript
q_vals <- seq(3.5, 12, length.out = 1000)

# Define the budworm model Stable state function
budworm_eq <- function(N, r, q) {
  r * N * (1 - N / q) - (1 * N^2) / (1^2 + N^2)
}

# Find equilibria for each r, q pair (take max real root as representative)
results <- expand.grid(r = r_vals, q = q_vals)
results$N <- NA_real_

for (i in 1:nrow(results)) {
  r <- results$r[i]
  q <- results$q[i]
  
  # Try multiple guesses
  guesses <- seq(0.01, 12, length.out = 5)
  roots <- c()
  
  for (guess in guesses) {
    root <- tryCatch({
      uniroot.all(function(N) budworm_eq(N, r, q), c(0.001, 15))
    }, error = function(e) NULL)
    
    if (!is.null(root)) {
      root <- root[root > 0]
      roots <- unique(c(roots, round(root, 4)))
    }
  }

  # For plotting: pick max real root (e.g., high-population Stable state)
  if (length(roots) > 0) {
    #results$N[i] <- mean(roots)
    results$N[i] <- ifelse(i%%4 < 2,min(roots),max(roots))
  }
}

# Remove NA values
results_clean <- na.omit(results)


# RHS of differential equation
dN_dt <- function(N, r, q) {
  r * N * (1 - N / q) - (N^2) / (1 + N^2)
}

# Derivative for stability analysis
d_dN <- function(N, r, q) {
  term1 <- r * (1 - 2 * N / q)
  term2 <- (2 * N) / (1 + N^2)^2
  term1 - term2
}

# Parameters
q <- 10
r_vals <- seq(0.2, .8, length.out = 1000)
results <- data.frame()

for (r in r_vals) {
  guesses <- seq(0.01, 20, length.out = 100)
  eqs <- c()
  
  for (guess in guesses) {
    root <- tryCatch({
      uniroot.all(function(N) dN_dt(N, r, q), c(0.001, 30))
    }, error = function(e) NULL)
    
    if (!is.null(root)) {
      root <- round(root[root > 0], 5)
      eqs <- c(eqs, root)
    }
  }
  
  eqs <- unique(eqs)
  for (N in eqs) {
    deriv <- d_dN(N, r, q)
    stability <- ifelse(deriv < 0, "Stable", "Unstable")
    results <- rbind(results, data.frame(r = r, N = N, Stability = stability))
  }
}

a=.67;d=.25

# get coordinates for a and d
aN=results$N[which.min(abs(results$r-a))]
dN=results$N[which.min(abs(results$r-d))]

# get coordinates for b 
b_1 <- results %>%
  filter(Stability == "Stable", N < 2.5) %>%
  slice_max(N)

b_2 <- results %>%
  filter(abs(r - b_1$r) < 1e-6) %>%  slice_max(N) 

# get coordinates for c 
c_1 <- results %>%
  filter(Stability == "Stable", N > 2.5) %>%
  slice_min(N)

c_2 <- results %>%
  filter(abs(r - c_1$r) < 1e-6) %>%  slice_min(N) 

# Create main panel of Figure 3
g1 = ggplot(results_clean, aes(x = q, y = r, fill = N)) +
  geom_tile() +
  scale_fill_gradientn(
    name = "Stable state N",
    colors = c("#BFDBF7", "#A663CC", "#C18203", "#880D1E"),
    na.value = "white") +
  labs(
    y = "Ar/B",
    x = "K/A"
  ) +
  theme_minimal(base_size = 16) +  # Set base font size
  theme(
    plot.title = element_text(size = 18, face = "bold"),
    plot.subtitle = element_text(size = 14),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 13),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12)
  )  +
  annotate("segment",
    x = 10, xend = 10, y = 0.2, yend = .75,
    arrow = arrow(ends = "both", type = "closed", length = unit(0.2, "cm")),
    color = "#A663CC",    size = 1) +
  annotate("segment",x = 4, xend = 8,y = 0.6, yend = .6,  arrow = arrow(ends = "both", type = "closed", length = unit(0.2, "cm")),
    color = "#880D1E",    size = 1
  ) + 
  annotate("segment",x = 4, xend = 8, y = 0.3, yend = 0.3,  arrow = arrow(ends = "both", type = "closed", length = unit(0.2, "cm")),
           color = "#1C110A", size = 1
  )+
  
  annotate("point",    x = 10,y = .67,size = 3,color = "#1C110A"
  ) +
  annotate( "text", x = 9.9, y = .67,label = "a",    size = 7,
    hjust = 1  
  )+
  annotate("point",  x = 10,y = b_1$r, size = 3,color = "#1C110A"
  ) +
  annotate(    "text", x = 9.9,y = b_1$r,label = "b",    size = 7,
    hjust = 1  # aligns text to the right, toward the point
  )+
  annotate(  "point",    x = 10,y = c_1$r,size = 3,color = "#1C110A"
  ) +
  annotate(    "text", x = 9.9,y = c_1$r,label = "c", size = 7,
    hjust = 1  # aligns text to the right, toward the point
  )+
  annotate( "point",    x = 10,y = .25,size = 3,color = "#1C110A"
  ) +
  annotate( "text", x = 9.9,y = .25,label = "d", size = 7,
    hjust = 1  
  )


# Create bottom right panel in Figure 3
g3 <- results %>% 
  filter(Stability == "Stable") %>%
  ggplot(aes(x = r, y = N)) +
  geom_point(size = 0.8, alpha = 0.8, color = "#A663CC") +
  labs(
    title = "",
    x = "Ar/B",
    y = "Stable state N",
    color = "Stability"
  ) +
  theme_classic(base_size = 16) +
  theme(legend.position = "none", coord_cartesian(ylim = c(0, max(N) + 1)) ) +
  
  # Point a
  annotate("point", x = a, y = aN, size = 3, color = "#1C110A") +
  annotate("text", x = a, y = aN, label = "a", size = 7, hjust = 1, vjust = -1) +
  
  # Point b_1
  annotate("point", x = b_1$r, y = b_1$N, size = 3, color = "#1C110A") +
  annotate("text", x = b_1$r, y = b_1$N, label = "b", size = 7, hjust = 1, vjust = -1) +
  
  # Point c_1
  annotate("point", x = c_1$r, y = c_1$N, size = 3, color = "#1C110A") +
  annotate("text", x = c_1$r, y = c_1$N, label = "c", size = 7, hjust = 1, vjust = -1) +
  
  # Point d
  annotate("point", x = d, y = dN, size = 3, color = "#1C110A") +
  annotate("text", x = d, y = dN, label = "d", size = 7, hjust = 1, vjust = -1) +
  
  # Point b_2
  annotate("point", x = b_2$r, y = b_2$N, size = 3, color = "#1C110A") +
  annotate("text", x = b_2$r, y = b_2$N, label = "b", size = 7, hjust = 1, vjust = -1) +
  
  # Point c_2
  annotate("point", x = c_2$r, y = c_2$N, size = 3, color = "#1C110A") +
  annotate("text", x = c_2$r, y = c_2$N, label = "c", size = 7, hjust = 1, vjust = -1)



# Parameters
r <- .60
q_vals <- seq(4, 7.5, length.out = 1000)
results <- data.frame()

for (q in q_vals) {
  guesses <- seq(0.01, 20, length.out = 100)
  eqs <- c()
  
  for (guess in guesses) {
    root <- tryCatch({
      uniroot.all(function(N) dN_dt(N, r, q), c(0.001, 30))
    }, error = function(e) NULL)
    
    if (!is.null(root)) {
      root <- round(root[root > 0], 5)
      eqs <- c(eqs, root)
    }
  }
  
  eqs <- unique(eqs)
  for (N in eqs) {
    deriv <- d_dN(N, r, q)
    stability <- ifelse(deriv < 0, "Stable", "Unstable")
    results <- rbind(results, data.frame(q = q, N = N, Stability = stability))
  }
}

results <- results %>% filter(Stability == "Stable")
# Create bottom left panel of Figure 3
g2 = ggplot(results, aes(x = q, y = N)) +
  geom_point(size = 0.8, alpha = 0.8, color = "#880D1E") +
  labs(
    title = "",
    x = "K/A",
    y = "Stable state N",
    color = "Stability"
  ) +
  theme_classic(base_size = 16) +theme(legend.position="none")



final_plot <- g1 / (g2 | g3) +
  plot_layout(heights = c(1.7, 1.3)) 
  
ggsave("figure3.png", plot = final_plot, width = 15, height = 12, dpi = 300)
