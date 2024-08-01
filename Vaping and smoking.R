library(Grind)

#only difference between dS and dV is in the b-parameter
smoking <- function(t, state, parms) {
  with(as.list(c(state,parms)), {
    dS <- r*S*(1-(S)/k)-(b_S*S^2)/(a^2+S^2)
    
    dr <- (growth_r * (S) - decay_r * r) * 0.001
    return(list(c(dS, dr)))
  }) }


vaping_smoking <- function(t, state, parms) {
  with(as.list(c(state,parms)), {
    dr <- (growth_r * (S + V) - decay_r * r) * 0.001
    dS <- r*S*(1-(S+V)/k)-(b_S*S^2)/(a^2+S^2)
    dV <- r*V*(1-(V+S)/k)-(b_V*V^2)/(a^2+V^2)
    
     #is dit heel erg om gewoon zo te doen
    return(list(c(dS, dV, dr)))
  }) }


# without vaping smoking escalates
model <- smoking
state <- s <- c(S = 0.1, r = 1.2)
parms <- p <- c(k = 10, b_S = 2.1, decay_r = 2, growth_r = 1, a = 1.5)

smoking_data <- run(tmax = 50, table = TRUE)


# vaping suppresses smoking, so in this setup good alternative
model <- vaping_smoking
state <- s <- c(S = 0.1, V = 0.1, r = 1.2)
parms <- p <- c(k = 10, b_S = 2.1, b_V = 2, decay_r = 2, growth_r = 1, a = 1.5)

vape_save <- run(table = TRUE, tmax = 50)


# vaping bad
model <- smoking
state <- s <- c(S = 0.1, r = 1)
parms <- c(k = 10, b_S = 2.4, decay_r = 2, growth_r = 1, a = 0.9)

no_smoke <- run(tmax = 100, table =TRUE)

model <- vaping_smoking
state <- s <- c(S = 0.1, V = 0.1, r = 1.2)
parms <- p <- c(k = 10, b_S = 2, b_V = 1.5, decay_r = 2, growth_r = 1.4, a = 0.9)

vape_bad <- run(tmax = 100, after =  "if(t > 50 & t < 60) {state[\"V\"] <- 0.01}", table = TRUE)


# # Resort to cocaine use after alcohol use
# coke_alcohol <- function(t, state, parms) {
#   with(as.list(c(state,parms)), {
#     dN <- r_N*N*(1-N/k)-(b_N*N^2)/(a^2+N^2)
#     dC <- r_C*C*(1-C/k)-((b_C + sin(N/2)*1.5)*C^2)/(a^2+C^2)
#     
#     dr_N <- (growth_r * N - decay_r * r_N) * 0.001
#     dr_C <- (growth_r * C - decay_r * r_C) * 0.001
#     return(list(c(dN, dC, dr_N, dr_C)))
#   }) }
# 
# 
# model <- coke_alcohol
# state <- s <- c(N = 0.1, C = 0.1, r_N = 1.2, r_C = 1.2)
# parms <- p <- c(k = 10, b_C = 4, b_N = 2, decay_r = 2, growth_r = 1, a = 1.5)
# 
# #coke use because of alcohol
# coke_escalation <- run(table = TRUE)
# 
# #Without alcohol also no coke use
# no_coke <- run(after = "state[\"N\"] <- 0", table = TRUE)
# 
# # Without coke use still alcohol use
# run(after = "state[\"C\"] <- 0")


