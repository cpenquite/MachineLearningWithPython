# Understanding Machine Learning with Python3

course url: [https://app.pluralsight.com/library/courses/python-understanding-machine-learning](url)

what is machine learning?
- building a model from example inputs to make data-driven predictions vs following strictly static program instructions
- machine learning logic - gather data we need and modify format to be passed into algorithm

types of machine learning
- supervised and unsupervised
- supervised: data has feature values and the value we want to algorithm to predict
    - value prediction
    - needs training data and value being predicted
    - trained model predicts value in new data
- unsupervised: clusters of like data and identifies groups of data with similar traits
    - identify clusters of like data
    - data does not contain cluster membership
    - model provides access to data by cluster

python libraries for machine learning
- numpy - scientific computing
- pandas - data frames
- matplotlib - 2D plotting
- scikit-learn - algorithms, preprocessing, performance evaluation, …

machine learning workflow
- an orchestrated and repeatable pattern which systematically transforms and processes information to create prediction solutions
- steps:
    - asking the right question
    - preparing data to answer the question
    - selecting the algorithm
    - training the model
    - testing the model’s accuracy
        - may need to go back and modify steps 2-4

asking the right question
- need statement to direct and validate work
- define end goal, starting point, and how to achieve goal
- solution statement goals
    - define scope (including data sources)
    - define target performance
    - define context for usage
    - define how solution will be created
- example
    - “predict if a person will develop diabetes” ->
    - “using pima indian diabetes data, predict which people will develop diabetes” ->
        - binary result (true or false)
        - genetic difference are a factor
        - 70% accuracy is common target
    - “using pima indian diabetes data, predict with 70% or greater accuracy, which people will develop diabetes” ->
        - disease prediction
        - medical research practices
        - unknown variations between people
        - likelihood is used
    - “using pima indian diabetes data, predict with 70% or greater accuracy, which people are likely to develop diabetes” ->
        - process pima indian data
        - transform data as required
    - “Use the Machine Learning Workflow to process and transform Pima Indian data to create a prediction model.  This model must predict which people are likely to develop diabetes with 70% or greater accuracy”

preparing your data
- Tidy data: Tidy datasets are easy to manipulate, model and visualize, and have a specific structure:
    - each variable is a column
    - each observation is a row
    - each type of observational unit is a table
- getting data
    - google, government databases, professional or company data sources, your company, your department, all of the above
- prima indian diabetes data
    - originally from UCI Machine Learning Repository - pima-data.csv
    - female patients at least 21 years old
    - 768 patient observation rows
    - 10 columns
        - 9 feature columns
            - number of pregnancies, blood pressure, glucose …
        - 1 class column
            - diabetes - true or false

selecting your algorithm
- scikit-learn has algorithms
- prediction model = supervised machine learning
- result type = regression (continuous values) or classification (discrete values)
- naive bayes algorithm*
    - based on likelihood and probability
    - bayes theorem
    - every feature has the same weight
    - requires small amount of data to train
- logistic regression algorithm
    - binary result
    - relationship between features are weighted
- decision tree
    - binary tree structure
    - node contains decision
    - requires enough data to determine nodes and splits

training the model
- letting specific data teach a ML algorithm to create a specific prediction model
- prepared data -> 70% training and 30% testing
- python training tip
    - scikit-learn has training functions

testing model’s accuracy
- improve performance
    - adjust current algorithm
    - get more data or improve data
    - improve training
    - switch algorithms
- random forest
    - ensemble algorithm
    - fits multiple trees with subsets of data
    - control overfitting
- fix overfitting:
    - regularization hyperparameter
    - cross validation
    - bias-variance trade-off
- cross validation (K-fold cross validation)
    - split training data into n size and use one fold as the validation data and repeat for each fold
- scikit-learn includes “CV” algorithms

Summary
- UCI ML Repo
    - [archive.ics.uci.edu/ml](url)
