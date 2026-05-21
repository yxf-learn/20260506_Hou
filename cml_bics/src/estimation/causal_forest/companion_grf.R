suppressPackageStartupMessages({
  library(grf)
  library(arrow)
})
args <- commandArgs(trailingOnly = TRUE)
panel <- read_parquet(args[1])
X <- as.matrix(panel[, c("age_at_grad","cum_gpa","first_gen","rural_origin",
                         "low_income_q","internship_months","high_demand_field")])
cf <- causal_forest(X = X, Y = panel$y_stability_12m, W = panel$enrolled_d,
                    num.trees = 2000, honesty = TRUE)
preds <- predict(cf, estimate.variance = TRUE)
write_parquet(as.data.frame(preds), args[2])
