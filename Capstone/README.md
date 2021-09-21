# Capstone Project: Customer Segmentation Report for Arvato Financial Services

## Overview of the project

For the final projetct in the <a href='real messages that were sent during disaster events'> Udacity Data Science Nano Degree </a>, several project ideas were presented. I chose to tackle on the Bertelsmann/Arvato Project.

The ideia behind this project is to deal with a real-life problem that the Arvato's Data Science group has to handle: the search for new customers. 

The main ideia of this project is to find similarities between people who are currently customers and people who are not. Then, use this information to find groups of potential new customers (people who are not currently customers but have high similarities with people who are).

This project was divided in 3 parts:
- Part One: Data Analysis and Data Cleaning
- Part Two: Customer Segmentation
- Part Three: Supervised Learning Model

This project is mainly divided in two goals:
- The first goal of this project is to analyze demographics data for customers of a mail-order sales company in Germany, comparing it against demographics information for the general population. The purpose is to find simillar characteristics in both groups, signaling good candidates, among the general population, for a marketing campaign.

- The second goal is to develop a machine learning model that can classify new samples as good or bad candidates for a marketing campaingn using the demographic information from each individual.

## Data Information
For this project, four datasets were made available with the following characteristics: 

- azdias: Demographics data for the general population of Germany. 891211 persons (rows) x 366 features (columns).
- customers: Demographics data for customers of a mail-order company. 191652 persons (rows) x 369 features (columns).
- mailout_train: Demographics data for individuals who were targets of a marketing campaign. 42982 persons (rows) x 367 (columns).
- mailout_test: Demographics data for individuals who were targets of a marketing campaign. 42833 persons (rows) x 366 (columns).

Two excel files containing the information about the data were also used.
- DIAS Information Levels - Attributes 2017: Information about the features present in the datasets.
- DIAS Attributes - Values 2017: Information about the values and what they represent in each feature of the dataset.

**Since the information contained in the files are sensitive, the data will not be avaible in this repository.**

## Customer Segmentation
In this part of the project, the two datasets (azdias and customers) were compared to find similarities between then. This was made to find wich characteristics the two populations (customers and possible future customers) shared in common. These similarities can be used to narrow down the search for new customers, optimizing the results and reducing marketing costs.

## Supervised Learning Model
The next part of the project involved building a Supervisioned Machine Learning model to predict if a person was a good candidate of becoming a new customer.
This was made using the mailout_train dataset that had a column signaling if the person responded to the marketing campaign or not.

This part was divided into 4 steps:
- Analysis of the dataset: Training and testing using all features and using only the selected features found in the Customer Clustering part. A Logistic Regression Classifier will be used to determin wich one is the best. The Stratified K-Fold Cross Validation algorithm will also be used to try dealing with the unbalanced data.

- Sampling redistribuition technique: Resampling techniques (Random Undersampling and Random Oversampling) will be used to deal with the unbalanced data. The same classifier will be trained and tested again with the resampled data and the results will be compared with the ones previosly obtained.

- Definition of the best model: Several Machine Learning Models will be used to find the one that delivers the best result.

- Hyperparameter Tuning: The hyperparameters of the best model will be tuned to find the combination that optimizes the results.

## Libraries used:
- <a href = 'https://pandas.pydata.org'>pandas</a> == 1.2.3
- <a href = 'https://scikit-learn.org/stable/'>scikit-learn</a> == 0.23.2
- <a href = 'https://docs.python.org/3/library/pickle.html'>pickle</a>
- <a href = 'https://numpy.org'>numpy</a> == 1.21.2
- <a href = 'https://docs.python.org/3/library/collections.html'>collections</a>
- <a href = 'https://matplotlib.org'>matplotlib</a> == 3.3.4
- <a href = 'https://seaborn.pydata.org'>seaborn</a> == 0.11.1
- <a href = 'https://scikit-optimize.github.io/stable/'>skopt</a> == 0.8.1
- <a href = 'https://imbalanced-learn.org/stable/'>imblearn</a> == 0.8.0

## Authors
<a href = 'https://github.com/alerlemos'>Alexandre Rosseto Lemos</a>

## Medium Post
<a href = 'https://alexandrerossetolemos.medium.com/customer-segmentation-project-116c47d7a4df'>Customer Segmentation Project</a>

## Acknowledgements
* [Udacity](https://www.udacity.com/) for providing this amazing Data Science Nanodegree Program.
* [Bertelsmann/Arvato](https://www.bertelsmann.com/divisions/arvato/) for providing the relevant datasets for this project.
