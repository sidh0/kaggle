import numpy as np
import sklearn as sk
import pandas as pd
import random
import scipy

def load_with_features(setType,subjects=['02'],electrodes=['Cz']):
    # Usage: data = load_with_features(setType,subjects,electrodes)
    # setType: 'train' or 'test'; will read from this directory
    # subjects: list/array of subject numbers
    # electrodes: list/array of electrodes
    #
    # This function will read the data for the chosen subjects and electrodes,
    # appropriately normalise it and pack it into one big dataframe.
    
    assert setType in ['test','train'], 'Invalid parameter for setType (valid is test, train)'
    
    
    nTimeSteps = 200

    sessions = [1,2,3,4,5]

    allElectrodes = []
    for elec in electrodes:
        allElectrodes.extend([elec+str(k) for k in range(nTimeSteps)])

    myColumns = ['session','startPosition','fb_index']+allElectrodes

    allDataList = []
    
    count = 0
    currentRow = 0

    for iSub,sub in enumerate(subjects):
        
        for iSess,session in enumerate(sessions):
            fName = setType + '/Data_S' + sub + '_Sess0' + str(session) + '.csv'

            print 'Loading file: ' + fName
            currentData = pd.read_csv(fName)
            feedback = currentData['FeedBackEvent']

            indices = np.where(feedback==1)[0]
            currentIndex = range(currentRow,currentRow+len(indices))
            

            newData = pd.DataFrame(columns=myColumns,index=currentIndex)
            newData = pd.DataFrame(columns=myColumns,index=currentIndex)
            for row,index in enumerate(indices):


                electrodeData = currentData[electrodes][index+20:index+20+nTimeSteps]
                electrodeData = electrodeData.subtract(electrodeData.mean(0),1)
                electrodeData = electrodeData.multiply(1/electrodeData.std(0),1)
                
                electrodeDataArray = electrodeData.values.transpose().ravel()

                newData.loc[currentRow,'session'] = session
                newData.loc[currentRow,'startPosition'] = index
                newData.loc[currentRow,'fb_index'] = row

                newData.loc[currentRow,allElectrodes] = electrodeDataArray


                currentRow += 1
            
            allDataList.append(newData)

    allData = pd.concat(allDataList)


    return allData

class Erp_template_full:    
    # Create an ERP template object which can be trained on a particular dataset.
    # This creates two templates, one for correct feedback and one for incorrect feedback.
    # The API is similar to scikit-learn (methods fit and predict)
    def fit(self,trainData,label,electrodes=['Cz']):
        
        myIndex = trainData.index
    
        allElectrodes = trainData.columns

        self.templateDict0 = {}
        self.templateDict1 = {}
        self.templateDict0_orth = {}
        self.templateDict1_orth = {}

        # Iterate over each electrode, computing the ERP for that electrode;
        # Store this template for later...

        for elec in electrodes:
            currentElectrodes = [k for k in allElectrodes if elec in k]
            currentData0 = trainData.iloc[np.where(label == 0)[0]][currentElectrodes]
            currentData1 = trainData.iloc[np.where(label == 1)[0]][currentElectrodes]
            
            template0 = currentData0.mean(0)
            template1 = currentData1.mean(0)

            self.templateDict0[elec] = template0
            self.templateDict1[elec] = template1



    def predict(self,testData):
        # Return the template scores (production of ERP multiplied by signal on a given trial)
        electrodes = self.templateDict0.keys()

        myIndex = testData.index
        allElectrodes = testData.columns[3:len(testData.columns)]
        templateScores = pd.DataFrame(index=myIndex)


        for j in range(2):
            if j == 0:
                templateDict = self.templateDict0
            elif j == 1:
                templateDict = self.templateDict1
                
            for elec in electrodes:
                currentElectrodes = [k for k in allElectrodes if elec in k]
            
                allCurrentData = testData[currentElectrodes]
            
                bigTemplateList = [templateDict[elec] for k in myIndex]
                bigTemplate = pd.concat(bigTemplateList,axis=1).transpose()
                
                bigTemplate.index = myIndex

                templateData = bigTemplate.multiply(allCurrentData)

                templateElecs = [k + 't' for k in currentElectrodes]
                templateScores[templateElecs] = templateData
                
        return templateScores
