
A_values <- seq(0.2, 5, 0.1)
z <- numeric()



for (i in 1:length(A_values)) {
A_value <- A_values[i]

control <- function(N, A = A_value, B =2) {
  
  (B*N^2)/(A^2 + N^2)
  
}

N_values <- seq(0, 10, 0.01)

y <- control(N_values)

l_c <- which.max(diff(y))
s_c <- which.min(diff(y))

z[i] <- N_values[l_c]/A_value

}


