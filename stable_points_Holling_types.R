library(Grind)
Budworm <- function(t, state, parms) {
  with(as.list(c(state,parms)), {
    dN <- r * N * (1-N/K) - (b*N^c)/(a^2+N^2)
    
    return(list(c(dN)))
  }) }
model <- Budworm

p <- parms <- c(K = 10, b = 1, a = 1, r = 0.5, c = 1)

#check 1
steady <- newton(s = c(N = 0)) # stable 0
steady <- round(newton(s = c(N = 8)), 3) # stable near K

p <- parms <- c(K = 10, b = 1, a = 1, r = 0.5, c = 1.2)

#check 1.2
steady <- newton(s = c(N = 0.03)) # stable near 0
steady <- newton(s = c(N = 8)) # stable near K


p <- parms <- c(K = 10, b = 1, a = 1, r = 0.5, c = 1.4)
#check 1.4
steady <- newton(s = c(N = 0.2)) # stable near 0
steady <- newton(s = c(N = 8)) # stable near K


p <- parms <- c(K = 10, b = 1, a = 1, r = 0.5, c = 1.6)
#check 1.6
steady <- newton(s = c(N = 0.37)) # stable near 0
steady <- newton(s = c(N = 8)) # stable near K

p <- parms <- c(K = 10, b = 1, a = 1, r = 0.5, c = 1.8)
#check 1.8
steady <- newton(s = c(N = 0.55)) # stable near 0
steady <- newton(s = c(N = 8)) # stable near K

p <- parms <- c(K = 10, b = 1, a = 1, r = 0.5, c = 2)
#check 2
steady <- newton(s = c(N = 0.55)) # stable near 0
steady <- newton(s = c(N = 8)) # stable near K




# plot stable states
p <- parms <- c(K = 10, b = 1, a = 1, r = 0.5, c = 1)
steady <- newton(s = c(N = 0))
p <- parms <- c(K = 10, b = 1, a = 1, r = 0.5, c = 1.1)
steady2 <- newton(s = c(N = 0.03))

p <- parms <- c(K = 10, b = 1, a = 1, r = 0.5, c = 1)
steady3 <- newton(s = c(N = 1.1))
steady4 <- newton(s = c(N = 8))


continue(steady, x = "c", y = "N", ymin= -0.1, ymax = 10)
continue(steady2, x = "c", y = "N", ymin= -0.1, ymax = 10, add = TRUE)
continue(steady3, x = "c", y = "N", ymin= -0.1, ymax = 10, add = TRUE)
continue(steady4, x = "c", y = "N", ymin= -0.1, ymax = 10, add = TRUE)




newton(s= c(N = 1))
