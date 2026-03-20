# Investigating Social Media Posts Across Mental Health Categories

## Overview
This project investigates whether linguistic and sentiment-based features extracted from social media posts can be used to classify mental health categories. Using over 53,000 labeled posts, text mining techniques and machine learning models are applied to identify patterns associated with different mental health conditions.

---

## Research Question
Can linguistic and sentiment features extracted from social media posts be used to distinguish and predict different mental health categories?

---

## Dataset
- Source: Kaggle – Sentiment Analysis for Mental Health  
- Size: 53,043 posts  

### Labels
- Anxiety  
- Bipolar  
- Depression  
- Normal  
- Personality Disorder  
- Stress  
- Suicidal  

### Notes
- Posts are anonymous  
- No demographic data available  
- Filtered for English, non-missing, meaningful text  

---

## Methodology

### Text Preprocessing
- Lowercasing text  
- Removing punctuation, stopwords, URLs, hashtags, and mentions  
- Tokenization using `tidytext::unnest_tokens()`  

### Feature Engineering

#### Structural Features
- Word count  

#### Text Features
- Unigram TF-IDF  
- Bigram TF-IDF  

#### Sentiment Features
- AFINN (continuous polarity score)  
- Log-transformed AFINN  
- Bing (positive/negative classification)  
- NRC (emotion categories such as anger, fear, joy, sadness)  

#### Topic Modeling
- Latent Dirichlet Allocation (LDA)  
- Topic probabilities used as model features  

---

## Models

### Multinomial Logistic Regression
- Baseline linear model  
- Captures direct relationships between features and labels  

### Random Forest
- 300 trees  
- Captures nonlinear relationships and feature interactions  

---

## Evaluation Metrics
- Accuracy  
- Confusion Matrix  
- Sensitivity (Recall)  
- Specificity  
- Cohen’s Kappa  

---

## Results

| Model                 | Accuracy | Kappa |
|----------------------|----------|-------|
| Logistic Regression  | 52.7%    | 0.349 |
| Random Forest        | 78.1%    | 0.713 |

### Key Findings
- Random Forest significantly outperformed Logistic Regression  
- Strong performance on Depression, Normal, and Anxiety categories  
- Performance improvements driven by bigrams, topic modeling, and combined sentiment features  

---

## Insights
- Mental health categories exhibit distinct linguistic and sentiment patterns  
- Depression, Anxiety, and Suicidal posts show more negative sentiment and higher emotional intensity  
- Normal posts are more neutral in tone  
- Overlap exists between closely related categories, particularly depression and suicidal  

---

## Limitations
- No demographic information  
- Noisy and informal social media text  
- Labels are not clinically verified  
- Class imbalance across categories  
- Lexicon methods may miss context such as sarcasm  

---

## Future Improvements
- Incorporate transformer-based models such as BERT  
- Address class imbalance  
- Include additional metadata  
- Explore temporal trends in mental health expression  

---

## Impact
This project demonstrates the potential of using text mining and machine learning to identify mental health signals in real time, supporting early detection, awareness, and intervention strategies.

---

## Tech Stack
- Language: R  

### Libraries
- tidyverse  
- tidytext  
- ggplot2  
- randomForest  
- nnet  
- topicmodels  
- caret  

---

## Project Structure

'''
├── data/
├── figures/ 
├── final_project.R 
├── final_project.Rmd
├── final_project.pdf 
└── README.md
'''
