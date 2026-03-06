# -------- PACKAGES --------
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(stringr)) install.packages("stringr")
if (!require(readxl)) install.packages("readxl")
if (!require(furrr)) install.packages("furrr")
if (!require(progressr)) install.packages("progressr")

library(tidyverse)
library(stringr)
library(readxl)
library(furrr)
library(progressr)

# Parallel + progress setup
# Uses available cores minus 1 to keep system responsive
future::plan(multisession, workers = max(1, parallel::detectCores() - 1))
handlers(global = TRUE)

# -------- 1) Load & prep Concreteness dictionary --------
# Path updated to root folder as requested
dict_path <- "C:/Users/Ben/Documents/GitHub/crowdsourcing/data/dictionaries/Paetzold_2016.xlsx"

cat("Loading Dictionary from:", dict_path, "...\n")

if (!file.exists(dict_path)) {
  stop(paste("Dictionary file not found at:", dict_path))
}

concreteness <- read_xlsx(dict_path, sheet = "Bootstrapped_Psycholinguistic_F") %>%
  select(Word, Concreteness) %>%
  mutate(Word = str_to_lower(Word))

# -------- 2) Load Your Data --------
data_path <- "C:/Users/Ben/Documents/GitHub/crowdsourcing/data/processed/final_with_sentence_ai_scores_20260126_123413.csv"
cat("Loading Data from:", data_path, "...\n")

df <- read.csv(data_path)

# Ensure story_content is character type and handle NAs
df$story_content <- as.character(df$story_content)

# -------- 3) Tokenizer --------
tokenize_alpha <- function(text) {
  if (is.na(text) || !nzchar(text)) return(character(0))
  # Extract only alphabetic words
  str_extract_all(str_to_lower(text), "\\b[a-z]+\\b")[[1]]
}

# -------- 4) Concreteness measures --------

# Mean concreteness among all matched words
compute_conc_mean_all <- function(text) {
  words <- tokenize_alpha(text)
  if (length(words) == 0) return(NA_real_)
  
  # Fast join with dictionary
  merged <- tibble(Word = words) %>% inner_join(concreteness, by = "Word")
  
  if (nrow(merged) == 0) return(NA_real_)
  mean(merged$Concreteness, na.rm = TRUE)
}

# -------- 5) Execute on Dataset --------
cat("Calculating concreteness for", nrow(df), "rows using 'story_content'...\n")

# Run in parallel with progress bar
with_progress({
  p <- progressor(steps = nrow(df))
  
  df$concreteness_score <- future_map_dbl(df$story_content, function(x) {
    p()
    compute_conc_mean_all(x)
  }, .options = furrr_options(seed = TRUE))
})

# -------- 6) Save Result --------
output_path <- "analysis/mediation_data_with_concreteness.csv"
write.csv(df, output_path, row.names = FALSE)

cat("\nDone! Saved updated dataset to:", output_path, "\n")
cat("Summary of Concreteness Scores:\n")
print(summary(df$concreteness_score))