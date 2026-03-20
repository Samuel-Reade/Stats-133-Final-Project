## ----setup, include=FALSE--------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## --------------------------------------------------------------------------------------------------

library(tidyverse)
library(tidytext)
library(textdata)
library(stopwords)
library(stringr)
library(janitor)
library(wordcloud)
library(RColorBrewer)
library(tidylo)
library(ggwordcloud)
library(topicmodels)
library(broom)

library(caret)
library(nnet)         
library(randomForest) 



## --------------------------------------------------------------------------------------------------

posts <- read_csv("Combined Data.csv") 


## --------------------------------------------------------------------------------------------------
head(posts)


## --------------------------------------------------------------------------------------------------

posts <- posts %>%
  rename(
    post_id = "...1",
    text = "statement",
    label = "status" 
  ) %>%
  mutate(label = as.factor(label))

glimpse(posts)



## --------------------------------------------------------------------------------------------------

# Counting posts by label: 
posts %>%
  count(label, sort = TRUE)

# Plotting: 
posts %>%
  count(label) %>%
  ggplot(aes(x = reorder(label, n), y = n)) +
  geom_col(fill = "orange") +
  coord_flip() +
  labs(
    title = "Distribution of Mental Health Categories",
    x = "Category",
    y = "Number of Posts"
  )



## --------------------------------------------------------------------------------------------------

data(stop_words)

# Cleaning at the post level: 
posts_clean <- posts %>%
  mutate(
    text = str_to_lower(text),
    text = str_replace_all(text, "’", "'"),                 # normalizing apostrophes
    text = str_replace_all(text, "http\\S+|www\\S+", " "),  # removing URLs
    text = str_replace_all(text, "@\\w+", " "),             # removing @
    text = str_replace_all(text, "#\\w+", " "),             # removing hashtags
    text = str_replace_all(text, "[^a-z\\s']", " "),        # keeping letters/spaces/apostrophes
    text = str_squish(text),
    word_count = str_count(text, "\\S+")
  ) %>%
  filter(!is.na(text), text != "", word_count >= 5) 

# Token cleaning: 
tokens <- posts_clean %>%
  select(post_id, label, text) %>%
  unnest_tokens(word, text, token = "words") %>%
  mutate(word = str_replace_all(word, "^'+|'+$", "")) %>%  # stripping stray quotes
  filter(str_detect(word, "^[a-z]+$")) %>%                 # only alphabetic tokens 
  filter(nchar(word) >= 3) %>%                             # dropping small tokens (ve, ll, don, etc.)
  anti_join(stop_words, by = "word") %>%                   # removing stopwords 
  filter(!word %in% c(                                     
    "im","ive","id","ill","dont","didnt","cant","wont","isnt","wasnt","arent","werent",
    "youre","theyre","ive","dont","didn","wasn","isn","aren","weren","couldn","wouldn","shouldn",
    "amp","rt","http","gon","wan","url","quot","amp"
  ))

# Creating bigrams
bigrams <- posts_clean %>%
  select(post_id, label, text) %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2)

# Separating and cleaning bigrams
bigrams_clean <- bigrams %>%
  separate(bigram, into = c("word1", "word2"), sep = " ") %>%
  filter(!is.na(word1), !is.na(word2)) %>%
  filter(str_detect(word1, "^[a-z]+$"),
         str_detect(word2, "^[a-z]+$")) %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word) %>%
  filter(nchar(word1) >= 3, nchar(word2) >= 3) %>%
  unite(bigram, word1, word2, sep = " ")

top_bigrams <- bigrams_clean %>%
  count(bigram, sort = TRUE) %>%
  slice_max(n, n = 200) %>%
  pull(bigram)

bigram_dtm <- bigrams_clean %>%
  filter(bigram %in% top_bigrams) %>%
  count(post_id, bigram) %>%
  bind_tf_idf(bigram, post_id, n) %>%
  pivot_wider(
    names_from = bigram,
    values_from = tf_idf,
    values_fill = 0
  )



## --------------------------------------------------------------------------------------------------

# Creating TF-IDF: 
tfidf <- tokens %>%
  count(label, word, sort = TRUE) %>%
  bind_tf_idf(word, label, n) %>%
  arrange(desc(tf_idf))

# Plotting: 
tfidf %>%
  group_by(label) %>%
  slice_max(tf_idf, n = 10) %>%
  ungroup() %>%
  ggplot(aes(tf_idf, reorder_within(word, tf_idf, label))) +
  geom_col(fill = "orange") +
  facet_wrap(~label, scales = "free") +
  scale_y_reordered() +
  labs(
    title = "Top TF-IDF Words by Mental Health Category",
    x = "TF-IDF",
    y = ""
  )



## --------------------------------------------------------------------------------------------------

# Sentiment Lexicons
afinn <- get_sentiments("afinn")
bing <- get_sentiments("bing") %>% distinct(word, sentiment)   
nrc <- get_sentiments("nrc")  %>% distinct(word, sentiment)   

# AFINN: 
sent_afinn <- tokens %>%
  inner_join(afinn, by = "word") %>%
  group_by(post_id, label) %>%
  summarise(afinn_score = sum(value), .groups = "drop")

# Bing: 
sent_bing <- tokens %>%
  inner_join(bing, by = "word", relationship = "many-to-many") %>%
  count(post_id, label, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
  mutate(bing_net = positive - negative)

# NRC: 
sent_nrc <- tokens %>%
  inner_join(nrc, by = "word", relationship = "many-to-many") %>%
  count(post_id, label, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0)



## --------------------------------------------------------------------------------------------------

# Creating table: 
features <- posts_clean %>%   
  select(post_id, label, word_count) %>%
  left_join(sent_afinn, by = c("post_id","label")) %>%
  left_join(sent_bing,  by = c("post_id","label")) %>%
  left_join(sent_nrc,   by = c("post_id","label")) %>%
  mutate(across(where(is.numeric), ~ replace_na(.x, 0)))

features <- features %>%
  mutate(log_afinn = sign(afinn_score) * log1p(abs(afinn_score)))

glimpse(features)



## --------------------------------------------------------------------------------------------------

# Signed Log of AFINN: 
features <- features %>%
  mutate(log_afinn = sign(afinn_score) * log1p(abs(afinn_score)))

# AFINN Sentiment Distribution by label: 
ggplot(features, aes(x = label, y = log_afinn, fill = label)) +
  geom_boxplot(show.legend = FALSE) +
  coord_flip() +
  labs(
    title = "Log-Transformed AFINN Sentiment by Mental Health Category",
    x = "",
    y = "Log AFINN score"
  ) +
  theme_minimal()

# Average AFINN Sentiment by label: 
features %>%
  group_by(label) %>%
  summarise(mean_afinn = mean(afinn_score), .groups = "drop") %>%
  ggplot(aes(x = reorder(label, mean_afinn), y = mean_afinn)) +
  geom_col(fill = "orange") +
  coord_flip() +
  labs(
    title = "Average AFINN Sentiment by Category",
    x = "",
    y = "Mean AFINN score"
  )

# Average Bing Sentiment by label: 
features %>%
  group_by(label) %>%
  summarise(mean_bing_net = mean(bing_net), .groups = "drop") %>%
  ggplot(aes(x = reorder(label, mean_bing_net), y = mean_bing_net)) +
  geom_col(fill = "orange") +
  coord_flip() +
  labs(
    title = "Average Bing Net Sentiment (Positive - Negative) by Category",
    x = "",
    y = "Mean net sentiment"
  )

# NRC emotions by label:
features %>%
  group_by(label) %>%
  summarise(
    anger = mean(anger),
    fear = mean(fear),
    joy = mean(joy),
    sadness = mean(sadness),
    .groups = "drop"
  ) %>%
  pivot_longer(-label, names_to = "emotion", values_to = "mean_count") %>%
  ggplot(aes(x = reorder(label, mean_count), y = mean_count)) +
  geom_col(fill = "orange") +
  facet_wrap(~ emotion, scales = "free") +
  coord_flip() +
  labs(
    title = "Average NRC Emotion Counts by Category",
    x = "",
    y = "Mean count"
  )

# Sentiment distribution: 
ggplot(features, aes(x = afinn_score)) +
  geom_histogram(bins = 40, fill = "orange") +
  facet_wrap(~label) +
  labs(
    title = "Distribution of Sentiment Scores by Category"
  )

# Emotion heatmap: 
emotion_means <- features %>%
  group_by(label) %>%
  summarise(
    anger = mean(anger),
    fear = mean(fear),
    sadness = mean(sadness),
    joy = mean(joy),
    trust = mean(trust)
  )

emotion_means %>%
  pivot_longer(-label, names_to = "emotion", values_to = "value") %>%
  ggplot(aes(x = emotion, y = label, fill = value)) +
  geom_tile() +
  scale_fill_viridis_c() +
  labs(
    title = "Emotion Intensity by Mental Health Category"
  )

# Heat-map with status and AFINN score: 
features %>%
  mutate(afinn_bin = cut(
    afinn_score,
    breaks = c(-Inf, -10, -5, -2, 0, 2, 5, 10, Inf),
    labels = c("<= -10", "-10 to -5", "-5 to -2", "-2 to 0",
               "0 to 2", "2 to 5", "5 to 10", ">= 10")
  )) %>%
  count(label, afinn_bin) %>%
  ggplot(aes(x = afinn_bin, y = label, fill = n)) +
  geom_tile(color = "white") +
  scale_fill_viridis_c() +
  labs(
    title = "Heatmap of Mental Health Category by AFINN Score Range",
    x = "AFINN Score Range",
    y = "Mental Health Category",
    fill = "Count"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



# Heat-map with status and AFINN score: 
features %>%
  mutate(affin_bin = cut(
    log_afinn,
    breaks = c(-Inf, -10, -5, -2, 0, 2, 5, 10, Inf),
    labels = c("<= -10", "-10 to -5", "-5 to -2", "-2 to 0",
               "0 to 2", "2 to 5", "5 to 10", ">= 10")
  )) %>%
  count(label, affin_bin) %>%
  ggplot(aes(x = affin_bin, y = label, fill = n)) +
  geom_tile(color = "white") +
  scale_fill_viridis_c() +
  labs(
    title = "Heatmap of Mental Health Category by AFINN Score Range",
    x = "AFINN Score Range",
    y = "Mental Health Category",
    fill = "Count"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



## --------------------------------------------------------------------------------------------------

par(mfrow = c(2,3))
tokens %>%
  count(label, word, sort = TRUE) %>%
  group_split(label) %>%
  lapply(function(x) {
  wordcloud(x$word, x$n, max.words = 80)
  title(main = unique(x$label), line = -3)
  })



## --------------------------------------------------------------------------------------------------

# Creating DTM for LDA: 
dtm_lda <- tokens %>%
  count(post_id, word) %>%
  cast_dtm(post_id, word, n)

# Fitting LDA: 
set.seed(5)

lda_model <- LDA(
  dtm_lda,
  k = 5,
  control = list(seed = 5)
)

# Inspecting Topics: 
tidy(lda_model, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  arrange(topic, -beta)

# Probs per Post: 
topic_gamma <- tidy(lda_model, matrix = "gamma")

# Converting to wide: 
topic_features <- topic_gamma %>%
  pivot_wider(
    names_from = topic,
    values_from = gamma,
    names_prefix = "topic_"
  )



## --------------------------------------------------------------------------------------------------

# Getting Top Words: 
top_words <- tokens %>% 
  count(word, sort = TRUE) %>%
  slice_max(n, n = 500) %>%
  pull(word)

tokens_model <- tokens %>%
  filter(word %in% top_words)

# Creating DTM: 
dtm <- tokens_model %>%
  count(post_id, word) %>%
  bind_tf_idf(word, post_id, n) %>%
  select(post_id, word, tf_idf) %>%
  pivot_wider(
    names_from = word,
    values_from = tf_idf,
    values_fill = 0
  )

# Merging with sentiment features: 
features_full <- features %>%
  left_join(dtm, by = "post_id") %>%
  left_join(bigram_dtm, by = "post_id") %>%
  left_join(topic_features, by = c("post_id" = "document"))

topic_features <- topic_features %>%
  mutate(document = as.numeric(document))

# Removing NAs: 
features_full <- features_full %>%
  mutate(across(where(is.numeric), ~ replace_na(.x, 0))) %>%
  mutate(label = as.factor(label))

# Sanity Check: 
sum(is.na(features_full))



## --------------------------------------------------------------------------------------------------

set.seed(100) 
idx <- caret::createDataPartition(features$label, p = 0.8, list = FALSE)
train_lr <- features[idx, ] %>% select(-post_id)
test_lr  <- features[-idx, ] %>% select(-post_id)

m_multinom <- nnet::multinom(label ~ ., data = train_lr, trace = FALSE)
pred_multinom <- predict(m_multinom, newdata = test_lr)
caret::confusionMatrix(pred_multinom, test_lr$label)



## --------------------------------------------------------------------------------------------------
# Logistic Regression confusion matrix object
cm_lr <- caret::confusionMatrix(pred_multinom, test_lr$label)

# Convert to data frame for ggplot
cm_lr_df <- as.data.frame(cm_lr$table)

# Heatmap
ggplot(cm_lr_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 3) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = "Logistic Regression Confusion Matrix",
    x = "Actual Label",
    y = "Predicted Label",
    fill = "Count"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


## --------------------------------------------------------------------------------------------------

set.seed(101)

# Sanitizing column names 
names(features_full) <- make.names(names(features_full), unique = TRUE)

# Splitting: 
idx <- caret::createDataPartition(features_full$label, p = 0.8, list = FALSE)
train_rf <- features_full[idx, ] %>% dplyr::select(-post_id)
test_rf  <- features_full[-idx, ] %>% dplyr::select(-post_id)

# Fitting and predicting: 
rf_model <- randomForest::randomForest(label ~ ., data = train_rf, ntree = 300) 
rf_pred  <- predict(rf_model, newdata = test_rf)

caret::confusionMatrix(rf_pred, test_rf$label) 



## --------------------------------------------------------------------------------------------------

cm <- caret::confusionMatrix(rf_pred, test_rf$label)

conf_table <- as.data.frame.matrix(cm$table)

knitr::kable(conf_table,
             caption = "Table 4: Confusion Matrix")

metrics_table <- data.frame(
  Metric = c("Accuracy", "95% CI (Lower)", "95% CI (Upper)", "Kappa"),
  Value = c(
    cm$overall["Accuracy"],
    cm$overall["AccuracyLower"],
    cm$overall["AccuracyUpper"],
    cm$overall["Kappa"]
  )
)

knitr::kable(metrics_table,
             caption = "Table 5: Model Performance Metrics")


class_table <- data.frame(
  Class = rownames(cm$byClass),
  Sensitivity = cm$byClass[, "Sensitivity"],
  Specificity = cm$byClass[, "Specificity"]
)

knitr::kable(class_table,
             caption = "Table 6: Sensitivity and Specificity by Class")



## --------------------------------------------------------------------------------------------------

cm_rf <- caret::confusionMatrix(rf_pred, test_rf$label)

cm_rf_df <- as.data.frame(cm_rf$table)

ggplot(cm_rf_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 3) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = "Random Forest Confusion Matrix",
    x = "Actual Label",
    y = "Predicted Label",
    fill = "Count"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


