
# Load the scry library
library(scry)

# Load data from Python
adata <- adata

# Call feature selection method
sce <- devianceFeatureSelection(adata, assay="X")
