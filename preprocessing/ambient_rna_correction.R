
# Install SoupX package if not already installed
if (!requireNamespace("SoupX", quietly = TRUE)) {
  install.packages("SoupX")
}

# Load the SoupX library
library(SoupX)

# Load data from Python
data <- data
data_tod <- data_tod
genes <- genes
cells <- cells
soupx_groups <- soupx_groups

# Specify row and column names of data
rownames(data) <- genes
colnames(data) <- cells

# Ensure correct sparse format for table of counts and table of droplets
data <- as(data, "sparseMatrix")
data_tod <- as(data_tod, "sparseMatrix")

# Generate SoupChannel Object for SoupX
sc <- SoupX::SoupChannel(data_tod, data, calcSoupProfile = FALSE)

# Add extra meta data to the SoupChannel object
soupProf <- data.frame(row.names = rownames(data), est = rowSums(data)/sum(data), counts = rowSums(data))
sc <- SoupX::setSoupProfile(sc, soupProf)

# Set cluster information in SoupChannel
sc <- SoupX::setClusters(sc, soupx_groups)

# Estimate contamination fraction
sc  <- SoupX::autoEstCont(sc, doPlot=FALSE)

# Infer corrected table of counts and round to integer
out <- SoupX::adjustCounts(sc, roundToInt = TRUE)

