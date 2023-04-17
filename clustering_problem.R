# Clusering Problem - 2

current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
rm(list=ls())

library(rstudioapi)
library(dplyr)
library(cluster)
library(NbClust)

library(ggplot2)
library("factoextra")
library(mclust)
library(doParallel)
library(dbscan)

# setup parallel backend
ncores = parallel::detectCores()
cl = makeCluster(ncores)
registerDoParallel(cl)

set.seed(1234)

load("cluster_data.RData")
num_rows = nrow(y)
num_cols = ncol(y)

scaled_y = scale(y)

# Multivariate normality check
alpha <- (num_cols-2)/(2*num_cols)
beta <- (num_rows-num_cols-3)/(2*(num_rows-num_cols-1))
a = num_cols/2
b = (num_rows-num_cols-1)/2

install.packages("MVN")
library(MVN)
mvn(y) # performs Shapiro-Wilk test for multivariate normality

# Plot beta and u quantiles
u = mahalanobis(scaled_y, colMeans(scaled_y), cov(scaled_y))
u = u^2
pr = (1:num_rows - alpha)/(num_rows - alpha - beta + 1)
quantiles = qbeta(pr, a, b)
data_frame = data.frame(u=sort(u), quantiles=quantiles)
ggplot(data_frame, aes(x=quantiles, y=u)) +
  geom_line() +
  labs(x="beta quantile", y="u quantile") +
  ggtitle("Checking Multivariate Normality") +
  theme(plot.title = element_text(hjust = 0.5))

# Implementing Feature Selection using PCA
pca <- prcomp(scaled_y, center = TRUE, scale. = TRUE)

# Calculate the cumulative proportion of variance explained by each principal component
variance_explained <- cumsum(pca$sdev^2 / sum(pca$sdev^2))

# Identify the number of principal components required to explain 90% or 99% of the variance
n_components_90 <- which(variance_explained >= 0.9)[1]
n_components_95 <- which(variance_explained >= 0.95)[1]
n_components_99 <- which(variance_explained >= 0.99)[1]

y_pca_90 <- pca$x[,1:n_components_90]
y_pca_95 <- pca$x[,1:n_components_95]
y_pca_99 <- pca$x[,1:n_components_99]

pca_indices = c(n_components_90, n_components_95, n_components_99)
pca_subsets_y = lapply(pca_indices, function(x) pca$x[,1:x])

# kmeans
get_scores_kmeans = function(x,kmax,iter.max=100,nstart=20)
{
  ch = numeric(length=kmax-1)
  silwid = numeric(length=kmax-1)
  n = nrow(x)
  for (k in 2:kmax) {

    # Apply Kmeans
    a = kmeans(x,k,iter.max=iter.max,nstart=nstart)
    
    # Calculate Silhouette Score for each K
    ss <- silhouette(a$cluster, dist(x))
    silwid[k-1] <- mean(ss[, 3])

    # Calculate CH Score for each K
    w = a$tot.withinss
    b = a$betweenss
    ch[k-1] = (b/(k-1))/(w/(n-k))
  }
  return(list(k=2:kmax,sil = silwid, ch=ch))
}

kmeans_scores = lapply(pca_subsets_y, get_scores_kmeans, kmax=20)

# Plotting CH and silhoutte results for Kmeans
score_90 <- kmeans_scores[[1]]
score_95 <- kmeans_scores[[2]]
score_99 <- kmeans_scores[[3]]

# Extract variables for plotting
k_90 <- score_90$k
ch_val_90 <- score_90$ch
sil_score_90 <- score_90$sil

k_95 <- score_95$k
ch_val_95 <- score_95$ch
sil_score_95 <- score_95$sil

k_99 <- score_99$k
ch_val_99 <- score_99$ch
sil_score_99 <- score_99$sil

# Find the maximum values for ch_val and sil_score
max_ch_val_90 <- k_90[which.max(ch_val_90)]
max_sil_score_90 <- k_90[which.max(sil_score_90)]
max_ch_val_95 <- k_95[which.max(ch_val_95)]
max_sil_score_95 <- k_95[which.max(sil_score_95)]
max_ch_val_99 <- k_99[which.max(ch_val_99)]
max_sil_score_99 <- k_99[which.max(sil_score_99)]

# Create the first plot for ch_val
plot(k_90, ch_val_90, type = "o", pch = 16, col = "blue", 
     ylim = range(ch_val_90, ch_val_99), xlab = "k values", ylab = "ch_val", 
     main = "ch_val vs k values")
lines(k_95, ch_val_95, type = "o", pch = 16, col = "black")
lines(k_99, ch_val_99, type = "o", pch = 16, col = "red")
points(max_ch_val_90, ch_val_90[which.max(ch_val_90)], pch = 16, col = "green")
points(max_ch_val_95, ch_val_95[which.max(ch_val_95)], pch = 16, col = "green")
points(max_ch_val_99, ch_val_99[which.max(ch_val_99)], pch = 16, col = "green")
abline(v = max_ch_val_90, col = "black", lty = 2)
abline(v = max_ch_val_95, col = "black", lty = 2)
abline(v = max_ch_val_99, col = "black", lty = 2)
legend("topright", legend = c("90", "95", "99"), col = c("blue","black", "red"), pch = c(16, 16, 16))

# Create the second plot for sil_score
plot(k_90, sil_score_90, type = "o", pch = 16, col = "blue", 
     ylim = range(sil_score_90, sil_score_99), xlab = "k values", ylab = "sil_score",
     main = "sil_score vs k values")
lines(k_99, sil_score_95, type = "o", pch = 16, col = "red")
lines(k_99, sil_score_99, type = "o", pch = 16, col = "red")
points(max_sil_score_90, sil_score_90[which.max(sil_score_90)], pch = 16, col = "green")
points(max_sil_score_95, sil_score_95[which.max(sil_score_95)], pch = 16, col = "green")
points(max_sil_score_99, sil_score_99[which.max(sil_score_99)], pch = 16, col = "green")
abline(v = max_sil_score_90, col = "black", lty = 2)
abline(v = max_sil_score_95, col = "black", lty = 2)
abline(v = max_sil_score_99, col = "black", lty = 2)
legend("topright", legend = c("90", "95", "99"), col = c("blue","black", "red"), pch = c(16, 16, 16))
                                                                                           

# NB Clustering
nbc_90 <- NbClust(y_pca_90, distance = "euclidean", min.nc = 2, 
              max.nc = 6, method = "complete", index = "all")
nbc_99 <- NbClust(y_pca_99, distance = "euclidean", min.nc = 2, 
                  max.nc = 6, method = "complete", index = "all")

stopCluster(cl)

fviz_nbclust <- function (x, FUNcluster = NULL, method = c("silhouette", "wss", 
                                                           "gap_stat"), diss = NULL, k.max = 10, nboot = 100, verbose = interactive(), 
                          barfill = "steelblue", barcolor = "steelblue", linecolor = "steelblue", 
                          print.summary = TRUE, ...) 
{
  set.seed(123)
  if (k.max < 2) 
    stop("k.max must bet > = 2")
  method = match.arg(method)
  if (!inherits(x, c("data.frame", "matrix")) & !("Best.nc" %in% 
                                                  names(x))) 
    stop("x should be an object of class matrix/data.frame or ", 
         "an object created by the function NbClust() [NbClust package].")
  if (inherits(x, "list") & "Best.nc" %in% names(x)) {
    best_nc <- x$Best.nc
    if (any(class(best_nc) == "numeric") ) 
      print(best_nc)
    else if (any(class(best_nc) == "matrix") )
      .viz_NbClust(x, print.summary, barfill, barcolor)
  }
  else if (is.null(FUNcluster)) 
    stop("The argument FUNcluster is required. ", "Possible values are kmeans, pam, hcut, clara, ...")
  else if (!is.function(FUNcluster)) {
    stop("The argument FUNcluster should be a function. ", 
         "Check if you're not overriding the specified function name somewhere.")
  }
  else if (method %in% c("silhouette", "wss")) {
    if (is.data.frame(x)) 
      x <- as.matrix(x)
    if (is.null(diss)) 
      diss <- stats::dist(x)
    v <- rep(0, k.max)
    if (method == "silhouette") {
      for (i in 2:k.max) {
        clust <- FUNcluster(x, i, ...)
        v[i] <- .get_ave_sil_width(diss, clust$cluster)
      }
    }
    else if (method == "wss") {
      for (i in 1:k.max) {
        clust <- FUNcluster(x, i, ...)
        v[i] <- .get_withinSS(diss, clust$cluster)
      }
    }
    df <- data.frame(clusters = as.factor(1:k.max), y = v, 
                     stringsAsFactors = TRUE)
    ylab <- "Total Within Sum of Square"
    if (method == "silhouette") 
      ylab <- "Average silhouette width"
    p <- ggpubr::ggline(df, x = "clusters", y = "y", group = 1, 
                        color = linecolor, ylab = ylab, xlab = "Number of clusters k", 
                        main = "Optimal number of clusters")
    if (method == "silhouette") 
      p <- p + geom_vline(xintercept = which.max(v), linetype = 2, 
                          color = linecolor)
    return(p)
  }
  else if (method == "gap_stat") {
    extra_args <- list(...)
    gap_stat <- cluster::clusGap(x, FUNcluster, K.max = k.max, 
                                 B = nboot, verbose = verbose, ...)
    if (!is.null(extra_args$maxSE)) 
      maxSE <- extra_args$maxSE
    else maxSE <- list(method = "firstSEmax", SE.factor = 1)
    p <- fviz_gap_stat(gap_stat, linecolor = linecolor, 
                       maxSE = maxSE)
    return(p)
  }
}

.viz_NbClust <- function (x, print.summary = TRUE, barfill = "steelblue", 
                          barcolor = "steelblue") 
{
  best_nc <- x$Best.nc
  if (any(class(best_nc) == "numeric") )
    print(best_nc)
  else if (any(class(best_nc) == "matrix") ) {
    best_nc <- as.data.frame(t(best_nc), stringsAsFactors = TRUE)
    best_nc$Number_clusters <- as.factor(best_nc$Number_clusters)
    if (print.summary) {
      ss <- summary(best_nc$Number_clusters)
      cat("Among all indices: \n===================\n")
      for (i in 1:length(ss)) {
        cat("*", ss[i], "proposed ", names(ss)[i], 
            "as the best number of clusters\n")
      }
      cat("\nConclusion\n=========================\n")
      cat("* According to the majority rule, the best number of clusters is ", 
          names(which.max(ss)), ".\n\n")
    }
    df <- data.frame(Number_clusters = names(ss), freq = ss, 
                     stringsAsFactors = TRUE)
    p <- ggpubr::ggbarplot(df, x = "Number_clusters", 
                           y = "freq", fill = barfill, color = barcolor) + 
      labs(x = "Number of clusters k", y = "Frequency among all indices", 
           title = paste0("Optimal number of clusters - k = ", 
                          names(which.max(ss))))
    return(p)
  }
}

fviz_nbclust(nbc_90)
fviz_nbclust(nbc_99)


# K is finalized to 3, Now we perform Different Clustering Methods

### Kmeans
set.seed(123)
km.res <- kmeans(y_pca_99, 3, nstart = 25)
# print(km.res)

fviz_cluster(km.res, data = y_pca_99,
             palette = c("#2E9FDF", "#00AFBB", "#E7B800"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)

### Gaussian mixture model (GMM)
library(mclust)
gmm = Mclust(pca_subsets_y[[1]], G = 10)
plot(pca_subsets_y[[1]],col=gmm$classification,cex=2,pch=1,lwd=2)


### Hierarchical Clustering
# Function to get within and between cluster sum of squares from hclust
get_sum_of_squares = function(data, cluster){
  wss = list()
  twss = vector()
  tss = vector()
  bss = vector()
  cluster = as.matrix(cluster)
  for(index in seq_len(ncol(cluster))){
    ss = aggregate(data, 
                   by=list(cluster[,index]), 
                   function(x) sum(scale(x, scale=F)**2))
    wss[[index]] = rowSums(ss[,-1])
    twss[index] = sum(ss[,-1])
    tss[index] = sum(scale(data, scale=F)**2)
    bss[index] = tss[index] - twss[index]
  }
  ss.all = list(wss=wss, twss=twss, tss=tss, bss=bss)
  return(ss.all)
}

#function for computing CH index
get_scores_for_hierarchical = function(data, kmax, twss, bss){
  
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
dists = lapply(pca_subsets_y, dist)

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
    title(main=paste(pca_indices[i], "Components Kept"), line = .5)
  }
  for(i in 1:3){
    data = wss.list[[i]]
    plot(2:25, data$twss, ylab="", xlab="")
    title(xlab = "K", font.lab=2, line = 2)
    if(i==1) title(ylab="Total WSS", line=2, cex.lab=1.2, font.lab=2)
  }
  title(paste("CH Index and Total WSS For", linkage,
              "Linkage Clustering Across PC Spaces"), 
        outer=T, line=-1)
}

methods = c("Complete", "Single", "Average")

for(method in seq_along(clusts)){
  print(method)
  cuts = lapply(clusts[[method]], function(x) cutree(x,k=2:20))
  ss = mapply(get_sum_of_squares, cluster=cuts, data=pca_subsets_y, SIMPLIFY = F)
  ch = lapply(ss, function(x) get_scores_for_hierarchical(data=y, kmax=20, 
                                                      twss=x$twss, bss=x$bss))
  pdf(paste0("ch_", methods[method], ".pdf"))
  plot.ch.wss(ch, ss, linkage=methods[method])
  dev.off()
}

### DBSCAN
kNNdistplot(pca_subsets_y[[3]], k=4)
abline(h=35, col="red")
set.seed(1234)
db = hdbscan(pca_subsets_y[[3]], minPts = 3)

kNNdistplot(pca_subsets_y[[3]], k=4)
abline(h=40, col="green")
set.seed(1234)
db = hdbscan(pca_subsets_y[[3]], minPts = 3)


### OPTICS


