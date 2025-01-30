# Project NLP | Automated Customers Reviews

## Executive Summary
This business case outlines the development of an NLP model to automate the processing of customer feedback for a retail company.

The goal is to evaluate how a traditional ML solutions (NaiveBayes, SVM, RandomForest, etc) compares against a Deep Learning solution (e.g, a Transformer from HuggingFace) when trying to analyse a user review, in terms of its score (positive, negative or neutral).

Project goals
- The ML/AI system should be able to run classification of customers' reviews (the textual content of the reviews) into positive, neutral, or negative.
- You should be able to compare which solution yeilds better results:
  - One that reads the text with a Language Model and classifies into "Positive", "Negative" or "Neutral"
  - One that transforms reviews into tabular data and classifies them using traditional Machine Learning techniques
