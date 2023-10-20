## Installing simcausal
# devtools::install_github('osofr/simcausal', build_vignettes = FALSE)

#Load libraries
library(simcausal)
library(infotheo)


#Create a df that will help me keep track of the Ground truth for each model
gt = data.frame(row.names = c('C1', 'C2', 'C3', 'Noise'))

set.seed(3450)
{
#2 vars mutual information, 1 hidden confounder
for (i in c(100, 1000, 5000, 10000, 15000, 20000)) {
  D <- DAG.empty() +
    node("HC1", distr="rbern", prob = 0.9) +
    node("C1", distr="rbern", prob = HC1*0.9) +
    node("C2", distr="rbern", prob = HC1*0.8) +
    node("C3", distr="rbern", prob = 0.76) +
    node("Noise", distr="rbern", prob = 0.5) +
    node("Target", distr="rbern", prob = ((C1 + C2 + C3)/3) > 0.5 )
  D <- set.DAG(D)
  dat <- sim(D,n=i)
  # plotDAG(D)
  print(summary(dat))
  print(mutinformation(dat[c(2,3,4,5,6)]))
  gt[paste0('2_vars_corr_1HC_n',i,'.csv')] = as.data.frame(cor(dat, method = 'pearson'))$Target[2:5]
  write.csv(dat[-1], paste0('2_vars_corr_1HC_n',i,'.csv'), row.names = FALSE)
}


#3 vars mutual information, 1 hidden confounder
for (i in c(100, 1000, 5000, 10000, 15000, 20000)) {
  D <- DAG.empty() +
    node("HC1", distr="rbern", prob = 0.9) +
    node("C1", distr="rbern", prob = HC1*0.9) +
    node("C2", distr="rbern", prob = HC1*0.8) +
    node("C3", distr="rbern", prob = HC1*0.76) +
    node("Noise", distr="rbern", prob = 0.5) +
    node("Target", distr="rbern", prob = ((C1 + C2 + C3)/3) > 0.5 )
  D <- set.DAG(D)
  dat <- sim(D,n=i)
  # plotDAG(D)
  print(summary(dat))
  print(mutinformation(dat[c(2,3,4,5,6)]))
  gt[paste0('3_vars_corr_1HC_n',i,'.csv')] = as.data.frame(cor(dat, method = 'pearson'))$Target[2:5]
  write.csv(dat[-1], paste0('3_vars_corr_1HC_n',i,'.csv'), row.names = FALSE)
}


#2 vars mutual information, 2 hidden confounders
for (i in c(100, 1000, 5000, 10000, 15000, 20000)) {
  D <- DAG.empty() +
    node("HC1", distr="rbern", prob = 0.85) +
    node("HC2", distr = "rbern", prob = 0.9)+
    node("C1", distr="rbern", prob = HC1*HC2*0.9) +
    node("C2", distr="rbern", prob = HC1*HC2*0.8) +
    node("C3", distr="rbern", prob = 0.9) +
    node("Noise", distr="rbern", prob = 0.5) +
    node("Target", distr="rbern", prob = ((C1 + C2 + C3)/3) > 0.5 )
  D <- set.DAG(D)
  dat <- sim(D,n=i)
  # plotDAG(D)
  print(summary(dat))
  print(mutinformation(dat[c(2,3,4,5,6)]))
  gt[paste0('2_vars_corr_2HC_n',i,'.csv')] = as.data.frame(cor(dat, method = 'pearson'))$Target[2:5]
  write.csv(dat[-1], paste0('2_vars_corr_2HC_n',i,'.csv'), row.names = FALSE)
}
  


#3 vars mutual information, 2 hidden confounders, version B
#2 vars mutual information, 2 hidden confounders
for (i in c(100, 1000, 5000, 10000, 15000, 20000)) {
  D <- DAG.empty() +
    node("HC1", distr="rbern", prob = 0.85) +
    node("HC2", distr = "rbern", prob = 0.9)+
    node("C1", distr="rbern", prob = HC1*0.9) +
    node("C2", distr="rbern", prob = HC1*HC2*0.8) +
    node("C3", distr="rbern", prob = HC2*0.9) +
    node("Noise", distr="rbern", prob = 0.5) +
    node("Target", distr="rbern", prob = ((C1 + C2 + C3)/3) > 0.5 )
  D <- set.DAG(D)
  dat <- sim(D,n=i)
  # plotDAG(D)
  print(summary(dat))
  print(mutinformation(dat[c(2,3,4,5,6)]))
  gt[paste0('3_vars_corr_2HC_n',i,'B.csv')] = as.data.frame(cor(dat, method = 'pearson'))$Target[2:5]
  write.csv(dat[-1], paste0('3_vars_corr_2HC_n',i,'B.csv'), row.names = FALSE)
}



#3 vars mutual information, 2 hidden confounders
for (i in c(100, 1000, 5000, 10000, 20000, 30000)) {
  D <- DAG.empty() +
    node("HC1", distr="rbern", prob = 0.85) +
    node("HC2", distr = "rbern", prob = 0.95)+
    node("C1", distr="rbern", prob = HC1*HC2*0.85) +
    node("C2", distr="rbern", prob = HC1*HC2*0.80) +
    node("C3", distr="rbern", prob = HC1*HC2*0.90) +
    node("Noise", distr="rbern", prob = 0.5) +
    node("Target", distr="rbern", prob = ((C1 + C2 + C3)/3) > 0.5 )
  D <- set.DAG(D)
  dat <- sim(D,n=i)
  # plotDAG(D)
  print(summary(dat))
  print(mutinformation(dat[c(2,3,4,5,6)]))
  gt[paste0('3_vars_corr_2HC_n',i,'.csv')] = as.data.frame(cor(dat, method = 'pearson'))$Target[2:5]
  write.csv(dat[-1], paste0('3_vars_corr_2HC_n',i,'.csv'), row.names = FALSE)
}

#### Data for Experiment: Vary Sample Size ####
set.seed(3450)

#No mutual information
for (i in c(100, 1000, 5000, 10000, 15000, 20000, 25000, 30000)) {
  D <- DAG.empty() +
    node("C1", distr="rbern", prob = 0.9) +
    node("C2", distr="rbern", prob = 0.8) +
    node("C3", distr="rbern", prob = 0.7) +
    node("Noise", distr="rbern", prob = 0.5) +
    node("Target", distr="rbern", prob =  ((C1 + C2 + C3)/3) > 0.5 )
  D <- set.DAG(D)
  dat <- sim(D,n=i)
  # plotDAG(D)
  print(summary(dat))
  print(mutinformation(dat[c(2,3,4,5,6)]))
  gt[paste0('0_vars_corr_0HC_n',i,'.csv')] = as.data.frame(cor(dat, method = 'pearson'))$Target[2:5]
  write.csv(dat[-1], paste0('0_vars_corr_0HC_n',i,'.csv'), row.names = FALSE)
}

}

## Save the ground truth
write.csv(gt, 'GroundTruth.csv', row.names = TRUE)

