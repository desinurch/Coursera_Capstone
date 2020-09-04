# Car Crash Severity for Autonomous Driving Capstone Project
_Capstone Project for Data Science Course_

## Overview

Detection of car crash severity with the background:
* Growth interest in AI systems, especially __Autonomous Driving__
* Prediction by governments to automate driving systems (BCG, 2015)
* Safety is still intricate issue, therefore:
	* Observe variables that took part in car crash severity from real-life data
	* Comparison of best machine learning methods to classify severity
* Model is also attractive to __build an alert system__ for city infrastructure workers (paramedics, police, firefighter, etc)

## Data
- Seattle GeoData. The data is an open data from the Seattle government, collected from 2004-2020. ([Collisions_OD](https://data-seattlecitygis.opendata.arcgis.com/datasets/5b5c745e0f1f48e7a53acec63a0022ab_0?geometry=-124.788%2C47.371%2C-119.732%2C48.018)). 
- Data specifications: 40 attributes, 221,144 collection of accidents. Severity indicator:
	* 3: Fatality — High Probability
	* 2b: Serious Injury — Mild Probability
	* 2: Injury — Low Probability
	* 1: Property Damage — Very Low Probability
	* 0: Unknown — Little to No Probability

## Requirements
- Language : Python

Libraries:
- matplotlib :3.2.1
- pandas : 1.0.5
- scikit-learn : 0.22.2
- numpy : 1.18.5
- jupyter notebook : 5.2.2

## Description

### Preprocessing Steps
* Data cleaning
* Fill NaN values   
* Variable correlation

### Model Architecture

Three models are evaluated:
* K-Nearest Neighbor
* Decision Tree
* Logistic Regression
All models are previously searched in the space of [1,20] for k, [1,15] for depth and [0.001,0.01,0.1,1,10,100] for regression in logistic regression.

![Capstone Pipeline](/img/capstone-pipeline.png)

## Evaluation

Evaluation is done using 5 metrics: Jaccard-index, F1-score, LogLoss, Precision, Recall
