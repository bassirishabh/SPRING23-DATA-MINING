# CLusering Problem - 2

# Clean up
library(rstudioapi)
library(dplyr)
library(cluster)
library(NbClust)
library(MVN)
library(ggplot2)
library("factoextra")
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
rm(list=ls())

set.seed(1234)

load("cluster_data.RData")
n = nrow(y)
p = ncol(y)

y.std = scale(y)

# Multivariate normality check
alp <- (p-2)/(2*p)
bet <- (n-p-3)/(2*(n-p-1))
a   <- p/2
b   <- (n-p-1)/2

mvn(y) # performs Shapiro-Wilk test for multivariate normality

# Plot beta and u quantiles
y.std <- scale(y)
S <- cov(y.std)
u <- mahalanobis(y.std, colMeans(y.std), S)
u <- u^2
pr <- (1:n - alp)/(n - alp - bet + 1)
quantiles <- qbeta(pr, a, b)
df <- data.frame(u=sort(u), quantiles=quantiles)
ggplot(df, aes(x=quantiles, y=u)) +
  geom_line() +
  labs(x="beta quantile", y="u quantile") +
  ggtitle("Checking Multivariate Normality") +
  theme(plot.title = element_text(hjust = 0.5))

pca <- prcomp(y.std, center = TRUE, scale. = TRUE)

# Calculate the cumulative proportion of variance explained by each principal component
variance_explained <- cumsum(pca$sdev^2 / sum(pca$sdev^2))

# Identify the number of principal components required to explain 90% or 99% of the variance
n_components_90 <- which(variance_explained >= 0.9)[1]
n_components_99 <- which(variance_explained >= 0.99)[1]

#kmeans # use Calinski and Harabasz (1974) ch index
ch.index = function(x,kmax,iter.max=100,nstart=25)
{
  ch = numeric(length=kmax-1)
  silwid = numeric(length=kmax-1)
  n = nrow(x)
  for (k in 2:kmax) {
    a = kmeans(x,k,iter.max=iter.max,nstart=nstart)
    w = a$tot.withinss
    b = a$betweenss
    ch[k-1] = (b/(k-1))/(w/(n-k))
    
    ss <- silhouette(a$cluster, dist(x))
    silwid[k-1] <- mean(ss[, 3])
  }
  return(list(k=2:kmax,ch=ch, sil = silwid))
}

ch = lapply(list(pca$x[,1:n_components_90], pca$x[,1:n_components_99]), ch.index, kmax=25)

# Plotting CH and silhoutte results
ch_90 <- ch[[1]]
ch_99 <- ch[[2]]

# Extract variables for plotting
k_90 <- ch_90$k
ch_val_90 <- ch_90$ch
sil_score_90 <- ch_90$sil

k_99 <- ch_99$k
ch_val_99 <- ch_99$ch
sil_score_99 <- ch_99$sil

# Find the maximum values for ch_val and sil_score
max_ch_val_90 <- k_90[which.max(ch_val_90)]
max_sil_score_90 <- k_90[which.max(sil_score_90)]
max_ch_val_99 <- k_99[which.max(ch_val_99)]
max_sil_score_99 <- k_99[which.max(sil_score_99)]

# Create the first plot for ch_val
plot(k_90, ch_val_90, type = "o", pch = 16, col = "blue", 
     ylim = range(ch_val_90, ch_val_99), xlab = "k values", ylab = "ch_val", 
     main = "ch_val vs k values")
lines(k_99, ch_val_99, type = "o", pch = 16, col = "red")
points(max_ch_val_90, ch_val_90[which.max(ch_val_90)], pch = 16, col = "green")
points(max_ch_val_99, ch_val_99[which.max(ch_val_99)], pch = 16, col = "green")
abline(v = max_ch_val_90, col = "black", lty = 2)
abline(v = max_ch_val_99, col = "black", lty = 2)
legend("topright", legend = c("90", "99"), col = c("blue", "red"), pch = c(16, 16))

# Create the second plot for sil_score
plot(k_90, sil_score_90, type = "o", pch = 16, col = "blue", 
     ylim = range(sil_score_90, sil_score_99), xlab = "k values", ylab = "sil_score",
     main = "sil_score vs k values")
lines(k_99, sil_score_99, type = "o", pch = 16, col = "red")
points(max_sil_score_90, sil_score_90[which.max(sil_score_90)], pch = 16, col = "green")
points(max_sil_score_99, sil_score_99[which.max(sil_score_99)], pch = 16, col = "green")
abline(v = max_sil_score_90, col = "black", lty = 2)
abline(v = max_sil_score_99, col = "black", lty = 2)
legend("topright", legend = c("90", "99"), col = c("blue", "red"), pch = c(16, 16))
                                                                                           

# NB Clustering
nbc_90 <- NbClust(pca$x[,1:n_components_90], distance = "euclidean", min.nc = 2, 
              max.nc = 6, method = "complete", index = "all")
nbc_99 <- NbClust(pca$x[,1:n_components_99], distance = "euclidean", min.nc = 2, 
                  max.nc = 6, method = "complete", index = "all")

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

fviz_nbclust(nc_90)
fviz_nbclust(nc_99)


# K is finalized, Now we perform Different Clustering Methods

### Kmeans

### TSNE

### GMM

### Hierarchical Clustering

### DBSCAN

