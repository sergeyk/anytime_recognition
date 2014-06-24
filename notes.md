## Notes

- sklearn has package for estimating covariance: http://scikit-learn.org/dev/modules/covariance.html
- To make things sparse I should threshold the classifier output but to determine the threshold I should actually set a parameter for the desired sparsity
- policy gradient is derived in Jiang et al 2012 - Learned Prioritization for Trading Off Accuracy and Speed and Branavan et al 2009 - Reinforcement Learning for Mapping Instructions to Actions
- SGDClassifier can fit both dense and sparse data without memory copy, but SVC can fit only dense data without memory copy
- For online learning, Vowpal Wabbit is good. In sklearn, SGDClassifier and SGDRegressor can do warm_start; PassiveAggressive classifier and regressor does online learning.
- LogisticRegression does not take instance weights

## Table summarizing feature construction and combination

Uses of binary mask (which features are obseved):
- simple concatenation with feature values (current use)
- polynomial terms with feature values
- clustering, so as to learn different classifiers for different clusters

Imputation of missing feature values:
- none/mean (current use)
- gaussian
- kNN
- locally weighted mixture of gaussian (active classification)

Combination of feature values:
- logistic loss linear classifier trained on full data
- logistic loss linear classifier trained on imputed data (current use)
- logistic loss with "marginalized corrupted features" linear classifier
- boosting classifier
- kNN
- locally weighted mixture of gaussian (active classification)

Number of learned classifiers:
- 1 (current use)
- all: at most min(2^F, N), where F is the number of features and N is the number of test instances
- K: apportioned based on clustering the masks

## Human cognition related work

http://vision.stanford.edu/documents/Fei-Fei_GistJoV2007.pdf
http://www.hegde.us/pdfs/Hegdé_ProgNeurobiol2008.pdf
http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0005927

this is an excerpt from abstract of the second paper:
"""
A more comprehensive, albeit unproven, alternative framework for understanding visual temporal dynamics is to view it as a sequential, Bayesian decision-making process. At each step, the visual system infers the likely nature visual scene by jointly evaluating the available processed image information and prior knowledge about the scene, including prior inferences. Whether the processing proceeds in a coarse-to-fine fashion depends largely on whether the underlying computations are hierarchical or not. Characterizing these inferential steps from the computational, perceptual and neural standpoints will be a key part of future work in this emerging field.
"""
