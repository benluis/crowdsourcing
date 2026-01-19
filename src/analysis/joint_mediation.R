# 1. Load Required Libraries
if (!require(mediation)) install.packages("mediation")
if (!require(dplyr)) install.packages("dplyr")

library(mediation)
library(dplyr)

# Helper function for significance stars
get_stars <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.001) return("***")
  if (p < 0.01) return("**")
  if (p < 0.05) return("*")
  if (p < 0.1) return(".")
  return("")
}

# 2. Load and Prepare Data
data <- read.csv("C:/Users/Ben/Documents/GitHub/crowdsourcing/analysis/mediation_analysis_data.csv")

# Ensure categorical variables are treated as factors
data$category_unified <- as.factor(data$category_unified)
data$month <- as.factor(data$month)
data$year <- as.factor(data$year)

# Split Data for Group Comparison
data_pre <- subset(data, PostGPT == 0)
data_post <- subset(data, PostGPT == 1)

# =========================================================================
# PART 1: JOINT REGRESSION (INTERACTION MODEL)
# =========================================================================
cat("\n\n=======================================================\n")
cat(" PART 1: JOINT REGRESSION (INTERACTION MODEL)\n")
cat("=======================================================\n")

# We interact text_quality with PostGPT to see if the slope shifts
joint_model <- lm(log_pledged_amount ~ 
                    text_quality * PostGPT + 
                    preparation_time + 
                    log_goal + word_count + category_unified + month + year, 
                  data = data)

# Extract key interaction coefficient
joint_summ <- summary(joint_model)
inter_coef <- coef(joint_model)["text_quality:PostGPT"]
inter_pval <- joint_summ$coefficients["text_quality:PostGPT", "Pr(>|t|)"]

cat("\n--- Key Result: Did the Slope Change? ---\n")
cat(sprintf("Interaction Term:  %.6f %s\n", inter_coef, get_stars(inter_pval)))
cat(sprintf("P-value:           %.4g\n", inter_pval))

if(inter_pval < 0.05) {
  cat("CONCLUSION: SIGNIFICANT. The market value of quality CHANGED after ChatGPT.\n")
} else {
  cat("CONCLUSION: NOT SIGNIFICANT. No change detected.\n")
}

cat("\n--- Full Joint Model Coefficients (Top 10) ---\n")
print(head(coef(joint_summ), 10))

# =========================================================================
# PART 2: CAUSAL MEDIATION COMPARISON
# =========================================================================
cat("\n\n=======================================================\n")
cat(" PART 2: CAUSAL MEDIATION COMPARISON\n")
cat("=======================================================\n")

# Formulas
f_med <- text_quality ~ preparation_time + log_goal + word_count + category_unified + month + year
f_out <- log_pledged_amount ~ text_quality + preparation_time + log_goal + word_count + category_unified + month + year

cat("Running Pre-GPT Mediation (1000 sims)...\n")
med.fit.pre <- lm(f_med, data = data_pre)
out.fit.pre <- lm(f_out, data = data_pre)
set.seed(2025)
med.out.pre <- mediate(med.fit.pre, out.fit.pre, treat = "preparation_time", mediator = "text_quality", boot = TRUE, sims = 1000)

cat("Running Post-GPT Mediation (1000 sims)...\n")
med.fit.post <- lm(f_med, data = data_post)
out.fit.post <- lm(f_out, data = data_post)
set.seed(2025)
med.out.post <- mediate(med.fit.post, out.fit.post, treat = "preparation_time", mediator = "text_quality", boot = TRUE, sims = 1000)

# Compare ACME (Signal)
acme_pre <- med.out.pre$d0
acme_post <- med.out.post$d0
acme_pre_p <- med.out.pre$d0.p
acme_post_p <- med.out.post$d0.p

cat("\n--- Signal Strength Comparison (ACME) ---\n")
cat(sprintf("Pre-GPT Signal:  %.8f %s\n", acme_pre, get_stars(acme_pre_p)))
cat(sprintf("Post-GPT Signal: %.8f %s\n", acme_post, get_stars(acme_post_p)))
cat(sprintf("Difference:      %.8f\n", acme_post - acme_pre))

# =========================================================================
# PART 3: PATH DECONSTRUCTION (WITH SIGNIFICANCE)
# =========================================================================
cat("\n\n=======================================================\n")
cat(" PART 3: PATH DECONSTRUCTION (WHY DID IT BREAK?)\n")
cat("=======================================================\n")

# Extract Coefficients and P-values
a_pre  <- coef(med.fit.pre)["preparation_time"]
a_pre_p <- summary(med.fit.pre)$coefficients["preparation_time", "Pr(>|t|)"]
a_post <- coef(med.fit.post)["preparation_time"]
a_post_p <- summary(med.fit.post)$coefficients["preparation_time", "Pr(>|t|)"]

b_pre  <- coef(out.fit.pre)["text_quality"]
b_pre_p <- summary(out.fit.pre)$coefficients["text_quality", "Pr(>|t|)"]
b_post <- coef(out.fit.post)["text_quality"]
b_post_p <- summary(out.fit.post)$coefficients["text_quality", "Pr(>|t|)"]

cat("\n[Path A: Effort -> Quality]\n")
cat(sprintf("Pre-GPT Slope:   %.6f %s\n", a_pre, get_stars(a_pre_p)))
cat(sprintf("Post-GPT Slope:  %.6f %s\n", a_post, get_stars(a_post_p)))

# T-Test for Difference in Path A
# Simple Z-test approximation: (b1 - b2) / sqrt(SE1^2 + SE2^2)
se_a_pre <- summary(med.fit.pre)$coefficients["preparation_time", "Std. Error"]
se_a_post <- summary(med.fit.post)$coefficients["preparation_time", "Std. Error"]
z_a <- (a_post - a_pre) / sqrt(se_a_pre^2 + se_a_post^2)
p_diff_a <- 2 * (1 - pnorm(abs(z_a)))

cat(sprintf("Difference:      %.6f (p = %.4g %s)\n", (a_post - a_pre), p_diff_a, get_stars(p_diff_a)))


cat("\n[Path B: Quality -> Funding]\n")
cat(sprintf("Pre-GPT Slope:   %.6f %s\n", b_pre, get_stars(b_pre_p)))
cat(sprintf("Post-GPT Slope:  %.6f %s\n", b_post, get_stars(b_post_p)))

# T-Test for Difference in Path B
se_b_pre <- summary(out.fit.pre)$coefficients["text_quality", "Std. Error"]
se_b_post <- summary(out.fit.post)$coefficients["text_quality", "Std. Error"]
z_b <- (b_post - b_pre) / sqrt(se_b_pre^2 + se_b_post^2)
p_diff_b <- 2 * (1 - pnorm(abs(z_b)))

cat(sprintf("Difference:      %.6f (p = %.4g %s)\n", (b_post - b_pre), p_diff_b, get_stars(p_diff_b)))

cat("\n--- Final Interpretation ---\n")
if (b_post < 0 && b_post_p < 0.05) {
  cat("CRITICAL FINDING: The market significantly PENALIZES quality Post-GPT.\n")
} else if (b_post < b_pre && p_diff_b < 0.05) {
  cat("FINDING: The reward for quality has significantly DROPPED.\n")
} else {
  cat("FINDING: No significant change in how the market values quality.\n")
}
