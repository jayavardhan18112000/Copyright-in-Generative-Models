# Load necessary libraries
library(readxl)
library(ggplot2)
library(dplyr)

# Read the Excel file
data <- read_excel("C:/Jayavardhan/5602/text_analysis4.xlsx")

# Visualize the distribution of perplexity for original and AI-generated texts
ggplot(data, aes(x = Perplexity, fill = Type)) +
  geom_histogram(position = "dodge", binwidth = 1000) +
  theme_minimal() +
  labs(title = "Perplexity Distribution", x = "Perplexity", y = "Count")

# Compare the average perplexity between original and AI-generated texts
data %>%
  group_by(Type) %>%
  summarise(Average_Perplexity = mean(Perplexity, na.rm = TRUE),
            SD_Perplexity = sd(Perplexity, na.rm = TRUE))




# Visualize the relationship between Burstiness and Lexical Diversity
ggplot(data, aes(x = Burstiness, y = Lexical_Diversity, color = Type)) +
  geom_point() +
  theme_minimal() +
  labs(title = "Burstiness vs Lexical Diversity", x = "Burstiness", y = "Lexical Diversity")

# Calculate correlation between Burstiness and Lexical Diversity for each type
data %>%
  group_by(Type) %>%
  summarise(Correlation = cor(Burstiness, Lexical_Diversity, use = "complete.obs"))




ggplot(data, aes(x = Type, y = Sentence_Complexity, fill = Type)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Sentence Complexity Comparison", x = "Text Type", y = "Sentence Complexity")

# Statistical comparison using t-test
original_complexity <- data %>%
  filter(Type == "Original") %>%
  pull(Sentence_Complexity)

generated_complexity <- data %>%
  filter(Type == "AI Generated") %>%
  pull(Sentence_Complexity)

t.test(original_complexity, generated_complexity)





ai_data <- data %>%
  filter(Type == "AI Generated")

# Analyze the distribution of similarity scores for AI-generated texts
summary(ai_data$Similarity_Score)
hist(ai_data$Similarity_Score, main = "Distribution of Similarity Scores", xlab = "Similarity Score", breaks = 10)

# Establishing thresholds
# Assuming you want to identify a specific percentile as the threshold
# For example, setting the 95th percentile as the threshold for AI-generated texts
threshold <- quantile(ai_data$Similarity_Score, 0.95)
print(paste("Similarity threshold for AI-Generated Texts: ", threshold))




library(corrplot)
metrics_data <- data %>%
  select(Perplexity, Burstiness, Lexical_Diversity, Sentence_Complexity)

# Calculate the correlation matrix
cor_matrix <- cor(metrics_data, use = "complete.obs")

# Visualize the correlation matrix
corrplot(cor_matrix, method = "circle", type = "upper", tl.col = "black", tl.srt = 45)
