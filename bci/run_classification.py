from __future__ import division
import numpy as np
import sklearn.svm as sk
import preprocess
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.ensemble as skensemble

# This approach (if not this exact script) placed 47th in the BCI Kaggle challenge.
# There are three key steps that are carried out:
# 1) The data are normalized
# 2) Template scores are computed based on the ERPs and added to the feature space
# 3) A straight-forward random forest classifier is applied, with pseudo-optimized parameters

# Define train and test subject IDs
trainSubs = ['02','06','07','11','12','13','14','16','17','18','20','21','22','23','24','26']
testSubs = ['01','03','04','05','08','09','10','15','19','25']

    
# Specify which electrodes we want; only used Cz for final submission
myElectrodes = ['Cz']



# First we sort out the labels for the events
allTrainLabels = pd.read_csv('TrainLabels.csv')
labelIndices = allTrainLabels['IdFeedBack']
    
# Indices for training set and testing set
trainIndices = [i for i,k in enumerate(labelIndices) for sub in trainSubs if 'S'+sub in k]
testIndices =  [i for i,k in enumerate(labelIndices) for sub in testSubs if 'S'+sub in k]

# Testing and test/CV labels
trainLabel = allTrainLabels.loc[trainIndices,'Prediction']
testLabel = allTrainLabels.loc[testIndices,'Prediction']

# Now we can start with our training data.
# Load training data; data is normalised in the load_with_features function
print 'Loading training data'
eegTrainData = preprocess.load_with_features('train',trainSubs,myElectrodes)

# Get electrodes and time points
electrodes = eegTrainData.columns[3:len(eegTrainData)]
     
# And now for the testing
# Load testing data; data is normalised in the load_with_features function
print 'Loading testing data'
eegTestData = preprocess.load_with_features('test',subjects=testSubs,electrodes=myElectrodes)
    

# ERP templates
# Construct the template
template = preprocess.Erp_template_full()
template.fit(eegTrainData,trainLabel,myElectrodes)

# Compute template scores for training data
trainTemplateScores = template.predict(eegTrainData)
trainTemplateScores.index = eegTrainData.index

# Compute template scores for test data
testTemplateScores = template.predict(eegTestData)
testTemplateScores.index = eegTestData.index


trainData = eegTrainData
testData = eegTestData

# Add template scores to the data
trainData[trainTemplateScores.columns] = trainTemplateScores
testData[testTemplateScores.columns] = testTemplateScores

features = trainData.columns

# Random Forest Classifier (very original, I know)
myClassifier = skensemble.RandomForestClassifier(n_estimators=100,max_depth=5,max_features=1.0)

# Fit training data
myClassifier.fit(trainData[features],trainLabel)

# Predict test data
testPrediction = myClassifier.predict_proba(testData[features])

print 'Creating submission file'
mySubmission = pd.read_csv('SampleSubmission.csv')
mySubmission['Prediction'] = testPrediction[:,1]

fName = 'bci_submission_sh2.csv'
mySubmission.to_csv(fName,index=False)
