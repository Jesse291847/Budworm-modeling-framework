library(scales)

get_colors<- function(input_data) {
  color_gradient <- col_numeric(palette = c("lightblue", "darkred"), domain = c(min(input_data), max(input_data)))
  colors <- apply(input_data,2, color_gradient)
  return(colors)
}

utility_network <- function(adj_matrix, dist_matrix, t_c = 1, t_l = 0.3, h_c = 2, h_l = 10, encounters = "all") {
  n <- nrow(adj_matrix) 
  all_nodes <- 1:n

  if (is.numeric(encounters)) {  
for (i in 1:encounters) {

  node1 <- sample(all_nodes, 1)
  node2 <- sample(all_nodes[!all_nodes %in% node1],1)

  connections_1 <- length(which(adj_matrix[node1, ] == 1))
  connections_2 <- length(which(adj_matrix[node2, ] == 1))

  utility <- 1/(1+exp((dist_matrix[node1,node2] - h_u) / t_u)) - 1/(1+exp(-(connections_1+connections_2 - h_c)/ t_c))


  if(utility > 0) {
    adj_matrix[node1, node2] <- adj_matrix[node2, node1] <- 1
  } else {
    adj_matrix[node1, node2] <- adj_matrix[node2, node1] <- 0
  }}
  }
if ( encounters == "all") {
  
  for (i in all_nodes) {
  node1 <- all_nodes[i]
  node2 <- sample(all_nodes[!all_nodes %in% node1],1)
  
  connections_1 <- length(which(adj_matrix[node1, -node2 ] == 1))
  connections_2 <- length(which(adj_matrix[node2, -node1 ] == 1))
  
  utility <- 1/(1+exp((dist_matrix[node1,node2] - h_c)/ t_c))
  cost <- 1/(1+exp(((-(connections_1 +connections_2 - h_l))/t_l)))
  p <- utility * (1-cost) 
  
  if(p > runif(1, 0, 1)) {
    adj_matrix[node1, node2] <- adj_matrix[node2, node1] <- 1
  } else {
    adj_matrix[node1, node2] <- adj_matrix[node2, node1] <- 0
  }  
    
  }
  
  
} else {
  print("encounters argument misspecified")
}
  
  return(adj_matrix)
}


