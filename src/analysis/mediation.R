
# 1. Load Required Libraries
if (!require(mediation)) install.packages("mediation")
if (!require(dplyr)) install.packages("dplyr")

library(mediation)
library(dplyr)

# 2. Load and Prepare Data
data <- read.csv("C:/Users/Ben/Documents/GitHub/crowdsourcing/analysis/mediation_analysis_data.csv")

# Ensure categorical variables are treated as factors
data$category_unified <- as.factor(data$category_unified)
data$month <- as.factor(data$month)
data$year <- as.factor(data$year)

# Split data
data_pre <- subset(data, PostGPT == 0)
data_post <- subset(data, PostGPT == 1)

# 3. Define the Mediation Function
run_full_mediation <- function(df, period_label, mediator_var = "text_quality") {
  cat("\n\n")
  cat("ANALYSIS PERIOD:", period_label, "| MEDIATOR:", mediator_var, "\n")
  cat("-------------------------------------------------------\n")
  
  # Construct formulas dynamically
  f_med <- as.formula(paste(mediator_var, "~ preparation_time + log_goal + word_count + category_unified + month + year"))
  f_out <- as.formula(paste("log_pledged_amount ~", mediator_var, "+ preparation_time + log_goal + word_count + category_unified + month + year"))
  
  # Phase 1: Fit Statistical Models
  med.fit <- lm(f_med, data = df)
  out.fit <- lm(f_out, data = df)

  # Fix scope for mediate function
  med.fit$call$formula <- f_med
  out.fit$call$formula <- f_out
  
  # Extract values
  a_path <- coef(med.fit)["preparation_time"]
  b_path <- coef(out.fit)[mediator_var]
  
  a_pval <- summary(med.fit)$coefficients["preparation_time", "Pr(>|t|)"]
  b_pval <- summary(out.fit)$coefficients[mediator_var, "Pr(>|t|)"]
  
  # --- Statistical Report ---
  cat("Step 1: Path Analysis\n")
  
  # Path A Report
  cat("Path A (Prep Time ->", mediator_var, "):\n")
  cat("   Coefficient:", round(a_path, 6), "\n")
  cat("   p-value:    ", format.pval(a_pval, digits=3), "\n")
  
  cat("\n")
  
  # Path B Report
  cat("Path B (", mediator_var, "-> Funding):\n")
  cat("   Coefficient:", round(b_path, 6), "\n")
  cat("   p-value:    ", format.pval(b_pval, digits=3), "\n")

  cat("\nStep 2: Causal Mediation Estimation (5,000 sims)\n")
  
  set.seed(2025)
  med.out <- mediate(med.fit, out.fit, 
                     treat = "preparation_time", 
                     mediator = mediator_var, 
                     boot = TRUE, sims = 5000)
  
  return(med.out)
}

# 4. Execute Analysis

cat("\nPRIMARY ANALYSIS: TEXT QUALITY\n")
results_pre <- run_full_mediation(data_pre, "Pre-ChatGPT", mediator_var = "text_quality")
print(summary(results_pre))

results_post <- run_full_mediation(data_post, "Post-ChatGPT", mediator_var = "text_quality")
print(summary(results_post))

cat("\nREFINEMENT: AI SCORE\n")
if ("ai_score" %in% names(data)) {
  results_pre_ai <- run_full_mediation(data_pre, "Pre-ChatGPT", mediator_var = "ai_score")
  print(summary(results_pre_ai))
  
  results_post_ai <- run_full_mediation(data_post, "Post-ChatGPT", mediator_var = "ai_score")
  print(summary(results_post_ai))
} else {
  cat("Warning: 'ai_score' column not found.\n")
}