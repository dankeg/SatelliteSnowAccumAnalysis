# Bayesian Spatial Segmentation - Hidden Markov Random Field

# Load Libraries ----
library(rstudioapi)
library(png)
library(fields)

# Import Data ----
folder_path <- rstudioapi::selectDirectory(caption = "Select Folder")
folder_name = sub(".*/", "", folder_path)

image_path = file.path(folder_path, "image.png")
logits_pos_path = file.path(folder_path, "class1.csv")
logits_neg_path = file.path(folder_path, "class0.csv")


# Plot Raw Image ----
image_png = readPNG(image_path)

par(mar=c(3,3,3,3)) # bottom, left, top, right

plot(NA,
     xlim=c(0,256),
     ylim=c(256,0),
     type="n",
     xaxs="i", yaxs="i",
     asp=1,
     axes=FALSE,
     xlab="",
     ylab="",
     main=folder_name)

axis(1, at=c(0,256), pos=256)
axis(2, at=c(0,256), pos=0)

rasterImage(image_png, 0, 256, 256, 0)


# Merge Logits to Probabilities ----
logits_pos <- read.csv(logits_pos_path, header = FALSE)
logits_neg <- read.csv(logits_neg_path, header = FALSE)

# Sigmoid of Logits (Binary Softmax)
prob_map = 1 / (1 + exp(-(logits_pos + logits_neg)))
prob_map = t(prob_map)  # To match png orientation


# Probability Heat Map ----

# Need to transpose to match png orientation
plot(NA,
     xlim=c(0,256),
     ylim=c(256,0),
     type="n",
     xaxs="i", yaxs="i",
     asp=1,
     axes=FALSE,
     xlab="",
     ylab="",
     main="Snow Probability Heatmap")

axis(1, at=c(0,256), pos=256)
axis(2, at=c(0,256), pos=0)

image.plot(x=0:256,
           y=0:256,
           z=prob_map,
           asp=1,
           xlab="",
           ylab="",
           axes=FALSE,
           add=TRUE)


# Model Definitions ----

# Bernoulli Log Likelihood
likelihood_log = function(y, p) {
  dbinom(y, size=1, prob=p, log=TRUE)
}

# Spatial Log Prior
prior_log = function(labels, i, j, candidate_label, beta) {
  n_pos = 0
  n_neg = 0
  
  # Top (North) Neighbor
  if (j > 1) {
    n_pos = n_pos + (labels[i, j-1] == 1)
    n_neg = n_neg + (labels[i, j-1] == 0)
  }
  # Bottom (South) Neighbor
  if (j < 256) {
    n_pos = n_pos + (labels[i, j+1] == 1)
    n_neg = n_neg + (labels[i, j+1] == 0)
  }
  # Right (East) Neighbor
  if (i > 1) {
    n_pos = n_pos + (labels[i-1, j] == 1)
    n_neg = n_neg + (labels[i-1, j] == 0)
  }
  # Left (West) Neighbor
  if (i < 256) {
    n_pos = n_pos + (labels[i+1, j] == 1)
    n_neg = n_neg + (labels[i+1, j] == 0)
  }
  
  # Result
  if (candidate_label == 1) {
    return(n_pos * beta)
  } else {
    return(n_neg * beta)
  }
}

# Gibbs Sampler
gibbs_sampling <- function(prob_map, n, beta=1.0) {
  labels = matrix(ifelse(prob_map > 0.5, 1, 0), 256, 256)
  samples = array(0, dim = c(256, 256, n))
  
  for (sample in 1:n) {
    
    for (i in 1:256) {
      for (j in 1:256) {
        
        # likelihood
        ll_pos = likelihood_log(1, prob_map[i, j])
        ll_neg = likelihood_log(0, prob_map[i, j])
        
        # prior
        pr_pos = prior_log(labels, i, j, 1, beta)
        pr_neg = prior_log(labels, i, j, 0, beta)
        
        # posterior
        post_log_pos = ll_pos + pr_pos
        post_log_neg = ll_neg + pr_neg
        
        post_log_max = max(post_log_pos, post_log_neg)
        
        # update
        p = exp(post_log_pos - post_log_max) / (exp(post_log_pos - post_log_max) + exp(post_log_neg - post_log_max))
        labels[i, j] = rbinom(1, 1, p)
        
      }
    }
    
    samples[, , sample] = labels
    cat("Iteration:", sample, "\n")
  }
  
  return(samples)
}

# Run Model ----
samples = gibbs_sampling(prob_map, n=500, beta=1.0)

posterior_prob = apply(samples, c(1,2), mean)
posterior_uncertainty = posterior_prob * (1 - posterior_prob)

snow_map = ifelse(posterior_prob > 0.5, 1, 0)

# Plot Results ----
plot(NA,
     xlim=c(0,256),
     ylim=c(256,0),
     type="n",
     xaxs="i", yaxs="i",
     asp=1,
     axes=FALSE,
     xlab="",
     ylab="",
     main="Posterior Snow Segmentation")

axis(1, at=c(0,256), pos=256)
axis(2, at=c(0,256), pos=0)

image(x=0:256,
           y=0:256,
           z=snow_map,
           asp=1,
           xlab="",
           ylab="",
           col = c("blue", "white"),
           axes=FALSE,
           add=TRUE)

legend(
  "topright",
  legend = c("Snow", "No Snow"),
  fill = c("white", "blue"),
  bty = "n"
)


# Plot Uncertainty
plot(NA,
     xlim=c(0,256),
     ylim=c(256,0),
     type="n",
     xaxs="i", yaxs="i",
     asp=1,
     axes=FALSE,
     xlab="",
     ylab="",
     main="Posterior Snow Segmentation Uncertainty")

axis(1, at=c(0,256), pos=256)
axis(2, at=c(0,256), pos=0)

image.plot(x=0:256,
           y=0:256,
           z=posterior_uncertainty,
           asp=1,
           xlab="",
           ylab="",
           axes=FALSE,
           add=TRUE)
