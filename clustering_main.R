# Clustering - Problem 2

# Clean up
library(rstudioapi)
library(dplyr)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
rm(list=ls())

set.seed(122446)

load("cluster_data.RData")
dim(y)
n = nrow(y)
p = ncol(y)

y.std = scale(y)

alp = (p-2)/(2*p)
bet = (n-p-3)/(2*(n-p-1))
a   = p/2
b   = (n-p-1)/2

pr = (1:n - alp)/(n - alp - bet + 1)
quantiles = qbeta(pr, a, b)

y.std = scale(y, center=T, scale = F)
S = cov(y.std)
S_inv = solve(S)
dsq = diag(y.std%*% tcrossprod(S_inv, y.std))

u = (n *dsq) / (n-1)^2

pdf("beta_quant.pdf")
par(pin=c(2.75, 2.75))
plot(quantiles,sort(u),type='l',xlab='beta quantile',ylab = 'u quantile', main="Checking Multivariate Normality")
dev.off()

# Marginal normality check
marginals = y.std[,sample(ncol(y), 25, replace=F)]

par(mfrow= c(5,5), pin=c(.75,.75))
check.norm = function(variable){
  qqnorm(variable, main="")
  qqline(variable, main="")
}


apply(marginals, 2, check.norm)
dev.copy(pdf, "marginals.pdf")
dev.off()

###Start CLustering here
# PCA to get lower dimensions for clustering 

svd.y = svd(y.std)

u = svd.y$u
d = svd.y$d
v = svd.y$v
PCs = u%*%diag(d)

eigs = d**2/(n-1)
varprop = cumsum(eigs)/sum(eigs)


# Get Subsets of PCs
pc.ind=c(5,20,40)
PCs.sub = lapply(pc.ind, function(x) PCs[,1:x])
names(PCs.sub) = paste0("y.pc.", pc.ind)

# Proportion of variance and scree plots
pdf("pve_scree.pdf")
nvars = c(p, 100, 40)
par(mfrow = c(2,3), pin=c(1.25,1.25),mar=c(3.5,3.5,2.5,1))
for(i in nvars){
  if(i==head(nvars,1)){
    plot(seq_len(i), varprop[1:i], type = "l", xlab="", ylab="")
    title(ylab="Prop of Variance Explained", cex.lab=1.2, line=2)
    abline(v=pc.ind[3], col=4)
  } else if(i==nvars[2]) {
    plot(seq_len(i), varprop[1:i], type = "p", xlab="", ylab="")
    abline(v=pc.ind, col=2:4)
  } else{
    plot(seq_len(i), varprop[1:i], type = "p", xlab="", ylab="")
  }
}

for(i in nvars){
  if(i==nvars[1]){
    plot(seq_len(i), eigs[1:i], type="l", xlab="", ylab="")
    title(ylab="Eigenvalue", cex.lab=1.2, line=2)
    abline(v=pc.ind[3], col=4)
  } else if(i==nvars[2]){
    plot(seq_len(i), eigs[1:i], type="p", xlab="", ylab="")
    abline(v=pc.ind, col=2:4)
  } else if(i==nvars[3]){
    plot(seq_len(i), eigs[1:i], type="p", xlab="", ylab="")
  }
}
title(main= "Scree and Proportion of Variance Plots for PCA", outer=T, line = -2)
title(xlab = "Principal Component Number", outer=T, line=-1, cex.lab=1.4)
dev.off()

#kmeans # use Calinski and Harabasz (1974) ch index
ch.index = function(x,kmax,iter.max=100,nstart=25,
                    algorithm="Lloyd")
{
  ch = numeric(length=kmax-1)
  n = nrow(x)
  for (k in 2:kmax) {
    a = kmeans(x,k,iter.max=iter.max,nstart=nstart,
               algorithm=algorithm)
    w = a$tot.withinss
    b = a$betweenss
    ch[k-1] = (b/(k-1))/(w/(n-k))
  }
  return(list(k=2:kmax,ch=ch))
}

ch = lapply(PCs.sub, ch.index, kmax=25)

pdf("kmeans_ch.pdf")
par(mfrow=c(1,3))
for(i in 1:3){
  ch.i = ch[[i]]
  plot(2:25, ch.i$ch, xlab="", ylab="", cex=1.1)
  
  if(i==1) title(ylab="CH Index", line=2, cex.lab=1.2)
  
  title(xlab = "K", line=2, cex.lab=1.2)
  title(main=paste(pc.ind[i], "PCs Kept"), line=1)
  abline(v=ch.i$k[which.max(ch.i$ch)], col=2)
}
title(main = "CH Index Across Various K in K-Means Clustering", outer=T, line=-1)
dev.off()



#Gaussian Mixture Models

BICs = lapply(PCs.sub, mclustBIC)

pdf("GMM_BIC.pdf")
par(mfrow=c(1,3), pin=c(1.5,3.5))
for(i in 1:3){
  plot(BICs[[i]], xlab="Number of Clusters", main=paste(pc.ind[i], "PCs Kept"), ylab="")
}
title(main="Bayesian Information Criteria Across Number of Clusters for GMM", outer=T, line=-1)
dev.off()


### Hierarchical Clustering
# Function to get within and between cluster sum of squares from hclust
clust.SS = function(data, cluster){
  
  twss = vector()
  tss = vector()
  bss = vector()
  wss = list()
  cluster = as.matrix(cluster)
  
  for(j in seq_len(ncol(cluster))){
    ss = aggregate(data, 
                   by=list(cluster[,j]), 
                   function(x) sum(scale(x, scale=F)**2))
    wss[[j]] = rowSums(ss[,-1])
    twss[j] = sum(ss[,-1])
    tss[j] = sum(scale(data, scale=F)**2)
    bss[j] = tss[j] - twss[j]
  }
  
  ss.all = list(tss=tss, wss=wss, twss=twss, bss=bss)
  
  return(ss.all) 
  
}

#function for computing CH index
ch.index2 = function(data, kmax, twss, bss){
  
  ch = numeric(length=kmax-1)
  n = nrow(data)
  for (k in 2:kmax) {
    to = twss[k-1]
    b = bss[k-1]
    ch[k-1] = (b/(k-1))/(to/(n-k))
  }
  return(list(k=2:kmax,ch=ch))
}

# Get distances for all subsets of the PC space considered
dists = lapply(PCs.sub, dist)

hc.complete = lapply(dists, hclust, method="complete")
hc.single   = lapply(dists, hclust, method="single")
hc.average  = lapply(dists, hclust, method="average")

clusts = list(hc.complete, hc.single, hc.average)

# Function to plot CH and Total WSS for hclust across linkages, pc components kept, and values of K
plot.ch.wss = function(ch.list, wss.list, linkage){
  par(mfrow=c(2,3), mai=c(.4,.5,.4,.2))
  for(i in 1:3){
    data = ch.list[[i]]
    plot(data$k, data$ch, ylab="", xlab="", xaxt="n")
    axis(1, labels=F)
    abline(v = which.max(data$ch)+1, col=2)
    if(i==1) title(ylab="CH Index", line=2, font.lab=2, cex.lab=1.2)
    title(main=paste(pc.ind[i], "Components Kept"), line = .5)
  }
  for(i in 1:3){
    data = wss.list[[i]]
    plot(2:25, data$twss, ylab="", xlab="")
    title(xlab = "K", font.lab=2, line = 2)
    if(i==1) title(ylab="Total WSS", line=2, cex.lab=1.2, font.lab=2)
  }
  title(paste("CH Index and Total WSS For", linkage,  "Linkage Clustering Across PC Spaces"), outer=T, line=-1)
}



methods = c("Complete", "Single", "Average")

for(method in seq_along(clusts)){
  cuts = lapply(clusts[[method]], function(x) cutree(x,k=2:25))
  ss = mapply(clust.SS, cluster=cuts, data=PCs.sub, SIMPLIFY = F)
  ch = lapply(ss, function(x) ch.index2(data=y, kmax=25, twss=x$twss, bss=x$bss))
  pdf(paste0("ch_", methods[method], ".pdf"))
  plot.ch.wss(ch, ss, linkage=methods[method])
}

pdf("dbscan_distplot.pdf")
par(mfrow=c(1,3), mai=c(.75,.75,.4,.3))
for(i in seq_along(PCs.sub)){
  dbscan::kNNdistplot(PCs.sub[[i]], 4)
  title( main=paste(pc.ind[i], "Components Kept"), line=.5)
  title(main = "Distance to Nearest Neighbor Across PC Spaces", outer=T, line=-1)
}
dev.off()

scan.grid.5 = expand.grid(eps = seq(380, 600, 5), 
                          minPts = seq(5, 50, by = 5)
)
scan.grid.20 = expand.grid(eps = seq(800, 1500, 5),
                           minPts= seq(5, 50, 5)
)
scan.grid.40 = expand.grid(eps = seq(1000,1800, 5),
                           minPts= seq(5, 50, 5)
)
scan.grid = list(scan.grid.5, scan.grid.20, scan.grid.40)

#function for computing CH index
ch.index3 = function(data, k, twss, bss){
  
  n = nrow(data)
  ch=(bss/(k-1))/(twss/(n-k))
  
  return(ch)
}

ch.all = list()
for(g in seq_along(scan.grid)){
  
  grid = scan.grid[[g]]
  data = PCs.sub[[g]]
  ch = vector()
  
  for(parm in seq_len(nrow(grid))){
    
    scan = dbscan(data,
                  eps = grid[parm, "eps"], 
                  minPts = grid[parm, "minPts"])
    
    cluster = scan$cluster[scan$cluster != 0]
    ncluster = length(unique(cluster))
    
    if(ncluster <=1){ ch[parm] = NA; next} # CH undefined for k=1
    
    df.scan = data[scan$cluster != 0, ]
    ss = clust.SS(data=df.scan, cluster = cluster)
    ch[parm] = ch.index3(data=df.scan, k=ncluster, twss=ss$twss, bss=ss$bss)
  }
  
  ch.all[[g]] = ch
}

sapply(ch.all, which.max)
scan.opt = list()

for(i in seq_along(ch.all)){
  scan.opt[[i]] = scan.grid[[i]][which.max(ch.all[[i]]),]
}

names(scan.opt) = names(PCs.sub)
print(scan.opt)

scan.5 = dbscan(PCs.sub[["y.pc.5"]], 
                eps=scan.opt[["y.pc.5"]][,"eps"],
                minPts = scan.opt[["y.pc.5"]][,"minPts"]
)

scan.20 = dbscan(PCs.sub[["y.pc.20"]], 
                 eps=scan.opt[["y.pc.20"]][,"eps"],
                 minPts = scan.opt[["y.pc.20"]][,"minPts"]
)

scan.40 = dbscan(PCs.sub[["y.pc.40"]], 
                 eps=scan.opt[["y.pc.40"]][,"eps"],
                 minPts = scan.opt[["y.pc.40"]][,"minPts"]
)

scan.40$cluster

pdf("dbscan_pcspace.pdf")
plot(PCs[,1], PCs[,2], col=scan.40$cluster + 1)
dev.off()
sink()