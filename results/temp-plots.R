
rm(list = ls())
gc()
graphics.off()
resultsfolder <- "/Users/jieding/Dropbox/MyResearch2016/Hrule_SSM_Pierre/py-smc2-master/results/"
resultsfile <- "/Users/jieding/Dropbox/MyResearch2016/Hrule_SSM_Pierre/py-smc2-master/results/temp.RData"
pdffile <- "/Users/jieding/Dropbox/MyResearch2016/Hrule_SSM_Pierre/py-smc2-master/results/temp.pdf"
library(ggplot2)
setwd(resultsfolder)
cat("working with file", resultsfile, "...
")
load(file = resultsfile)
pdf(file = pdffile, useDingbats = FALSE, title = "SMC2 results")

nbparameters <- 3
graphics.off()

g <- qplot(x = resamplingindices, y = acceptratios, geom = "line", colour = "acceptance rates")
#g <- g + geom_line(aes(y = guessAR, colour = "guessed acceptance rates"))
g <- g + xlim(0, T) + ylim(0, 1) +  opts(legend.position = "bottom", legend.direction = "horizontal")
g <- g + scale_colour_discrete(name = "") + xlab("time") + ylab("acceptance rates")
print(g)

Ntheta <- dim(thetahistory)[3]
ESSdataframe <- as.data.frame(cbind(1:length(ESS), ESS))
g <- ggplot(data = ESSdataframe, aes(x = V1, y= ESS))
g <- g + geom_line() + xlab("iterations") + ylab("ESS") + ylim(0, Ntheta)
print(g)
#g <- ggplot(data = ESSdataframe, aes(x = V1, y= ESS))
#g <- g + geom_line() + xlab("iterations (square root scale)") + ylab("ESS (log)") + ylim(0, Ntheta)
#g <- g + scale_x_sqrt() + scale_y_log()
#print(g)

g <- qplot(x = 1:T, y = cumsum(computingtimes), geom = "line",
           ylab = "computing time (square root scale)", xlab = "iteration")
g <- g + scale_y_sqrt()
print(g)

evidencedataframe <- as.data.frame(cbind(1:length(evidences), evidences))
g <- ggplot(data = evidencedataframe, aes(x = V1, y= evidences))
g <- g + geom_line() + xlab("iterations") + ylab("evidence")
print(g)

#indexhistory <- length(savingtimes)
#t <- savingtimes[indexhistory]
#w <- weighthistory[indexhistory,]
#w <- w / sum(w)

indexhistory <- length(savingtimes)
w <- weighthistory[indexhistory,]
w <- w / sum(w)
i <- 1
g <- qplot(x = thetahistory[indexhistory,i,], weight = w, geom = "blank")
g <- g + geom_histogram(aes(y = ..density..)) + geom_density(fill = "blue", alpha = 0.5)
g <- g + xlab(expression(B))
if (exists("trueparameters")){
    g <- g + geom_vline(xintercept = trueparameters[i], linetype = 2, size = 1)
}
g <- g + opts(legend.position = "bottom", legend.direction = "horizontal")

priorfunction <- function(x) dexp(x, rate = 2.00000)
g <- g + stat_function(fun = priorfunction, aes(colour = "prior"), linetype = 1, size = 1)
g <- g + scale_colour_discrete(name = "")

print(g)

observationsDF <- cbind(data.frame(observations), 1:length(observations))
names(observationsDF) <- c("y", "index")
g <- ggplot(data = observationsDF, aes(x = index, y = y)) 
g <- g + geom_line() + ylab("observations")
print(g)

if (exists("truestates") && is.null(dim(truestates))){
    truestates <- as.matrix(truestates, ncol = 1)
}
predictedquantities <- grep(patter="predicted", x = ls(), value = TRUE)
if (T > 25){
    start <- 10
} else {
    start <- 1
}
for (name in predictedquantities){
    ystr <- paste(name, "[start:T,1]", sep = "")
    yqt1 <- paste(name, "[start:T,2]", sep = "")
    yqt2 <- paste(name, "[start:T,3]", sep = "")
    g <- qplot(x = start:T, geom = "blank")
    g <- g + geom_line(aes_string(y = ystr, colour = paste("'", name, "'", sep = "")))
    g <- g + geom_line(aes_string(y = yqt1, colour = paste("'", name, "quantile'", sep = "")))
    g <- g + geom_line(aes_string(y = yqt2, colour = paste("'", name, "quantile'", sep = "")))
    if (name == "predictedstate1" && exists("truestates")){
            g <- g + geom_line(aes(y = truestates[start:T,1], colour = "True states"))
    } else {
        if (name == "predictedstate2" && exists("truestates")){
            g <- g + geom_line(aes(y = truestates[start:T,2], colour = "True states"))
        } else {
            if (name == "predictedobservations"){
                g <- g + geom_line(aes(y = observations[start:T,1], colour = "observations"))
            }
            if (name == "predictedsquaredobs"){
                g <- g + geom_line(aes(y = observations[start:T,1]**2, colour = "squared observations"))
            }
        }
    }
    g <- g + opts(legend.position = "bottom", legend.direction = "horizontal")
    g <- g + xlab("time") + ylab(name) + scale_colour_discrete(name = "")
    print(g)
}

if (exists("truestates") && is.null(dim(truestates))){
    truestates <- as.matrix(truestates, ncol = 1)
}
filteredquantities <- grep(patter="filtered", x = ls(), value = TRUE)
for (name in filteredquantities){
    g <- qplot(x = 1:T, geom = "blank") 
    g <- g + geom_line(aes_string(y = name,
    colour = paste("'",name, "'", sep = "")))
    if (name == "filteredstate1" && exists("truestates")){
            g <- g + geom_line(aes(y = truestates[,1], colour = "True states"))
    } else {
        if (name == "filteredstate2" && exists("truestates"))
            g <- g + geom_line(aes(y = truestates[,2], colour = "True states"))
    }
    g <- g + xlab("time") + ylab(name) + scale_colour_discrete(name = "")
    g <- g + opts(legend.position = "bottom", legend.direction = "horizontal")
    print(g)
}

dev.off()
