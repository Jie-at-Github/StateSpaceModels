#============ Standard  import of data from python results ====================
#rm(list = ls())
#gc()
graphics.off()
resultsfolder <- "/.../results/"
resultsfile <- "/.../results/temp.RData"
library(ggplot2)
setwd(resultsfolder)
cat("working with file", resultsfile, "...")

#update this every time a new result comes
load(file = resultsfile)
graphics.off()

#============ Plot parameters ====================
indexhistory <- length(savingtimes)
w <- weighthistory[indexhistory,]
w <- w / sum(w)
i <- 1
xLab=c('tau', 'sigma', 'r', 'b')
g <- qplot(x = thetahistory[indexhistory,i,], weight = w, geom = "blank")
#g <- g + geom_histogram(aes(y = ..density..), binwidth = 1) + geom_density(fill = "blue", alpha = 0.5)
g <- g + geom_histogram(aes(y = ..density..)) + geom_density(fill = "blue", alpha = 0.5)
g <- g + xlab(xLab[i])
if (exists("trueparameters")){
  g <- g + geom_vline(xintercept = trueparameters[i], linetype = 2, size = 1)
}
print(g)


print(lScore) #log score at each time 

print(hScore) #H score at each time 

