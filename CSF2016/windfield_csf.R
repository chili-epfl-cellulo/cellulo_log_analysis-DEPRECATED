windfield_o <- windfield_game1_orange_withindex_distances

attach(windfield_o)
library(ggplot2)
ggplot(windfield_o, aes(x = windfield_o$V1, y =windfield_o$V2)) + geom_line() 

p <- ggplot(windfield_game1_orange_withindex_velocity, 
       aes(x=windfield_game1_orange_withindex_velocity$V1, y=windfield_game1_orange_withindex_velocity$V3))
p+geom_line()+xlim(0,2000)
