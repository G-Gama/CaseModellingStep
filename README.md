# CaseModellingStep

A case where I show how I would approach the Modelling Step of a Binary Classifier. In this case, I had to develop a model to prevent fraudulent transactions to occur, which can be modeled as a binary classifier. Here I summarized the modelling step in a single notebook, which condenses the Feature Engineering, Variable Selection, Model Training, Model Evaluation, Feature Importance and Model Finalization steps. In the notebook I try to explain why I took some decisions (Sampling the test with OOS instead of OOT, using custom transformers, sampling the training set *et cétera*) and what could be done differently, so pay attention on the markdowns I wrote there! 

The files you should find here:

- requirements.txt: I install these directly on the notebook with pip install, but they should help you to reproduce the code I'm running here. It just installs some libs that are not found on default conda environments.
- FraudModel.ipynb: The notebook where the model is developed.
- scripts/
    - helper.py: Some custom transformers which will help me with the feature engineering step.
- model/
    - best_model.pickle: the best model I could achieve using this dataset.
- files/
    - labeled_transactions.csv: the dataset I'm using for this case.

Feel free to give me any sugestions and feedbacks!
