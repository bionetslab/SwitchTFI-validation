
library(switchde)

verbosity <- verbosity
if (verbosity >= 1){
    print('### Loading data into R ...')
}
data <- data
genes <- genes
pt <- pseudotime
zinf <- zero_inflated

rownames(data) <- genes

verbosity <- verbosity
if (verbosity >= 1){
    print('### Calculating SwitchDE p-values ...')
}

sde <- switchde(data, pt, zero_inflated = zinf)

# /data/bionets/ac07izid/miniconda3/envs/dtfi/lib/R/library/switchde
