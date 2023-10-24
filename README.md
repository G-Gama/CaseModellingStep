# CaseModellingStep

A case where I show how I would approach the Modelling Step of a Binary Classifier.

The files you should find here:

- requirements.txt: I install these directly on the notebook with pip install, but they should help you to reproduce the code I'm running here. They are based on a sagemaker environment with a few additions like optuna, shap, boruta etc.
- FraudModel.ipynb: The development of the model.
- scripts/
    - helper.py: Some custom transformers which will help me with the feature engineering step.
- model/
    - best_model.pickle: the best model I could achieve using this dataset.
- files/
    - labeled_transactions.csv: the dataset I'm using for this case.

Feel free to give me any sugestions and feedbacks!