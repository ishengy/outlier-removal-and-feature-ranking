# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 01:27:12 2020

Programming Assignment #1: The goal of this assignment was to apply the topics covered within
modules 1-4 on the iris dataset. The following techniques were applied: visualization, sorting, 
outlier removal, feature ranking.

@author: Ivan Sheng
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
import scipy

"""
The function to sort a dataframe by feature in ascending order
    
Parameters:
    dataframe (dataframe): the dataframe of the original dataset (iris datset in this case)
    sortbyFeature (string): the name of the feature to sort by
  
Returns: 
    sortedTable (dataframe): the sorted table of the original dataset
"""
def sortTable(dataframe,sortbyFeature):
    
    sortedTable = dataframe.sort_values(by=[sortbyFeature])
    sortedTable.reset_index(inplace = True, drop = True) 
    return sortedTable

"""
The function to plot the distribution of the data via histogram
    
Parameters:
    sortedTable (dataframe): the dataset to be plotted
    featureNames (list): a list of all feature names
        
Returns:
    Nothing
"""
def plotHist(sortedTable, featureNames, path, fileName):
    
    outpath = path + 'output//'
    #find all unique classes
    classes = sortedTable['class'].unique()
    plt.figure(figsize=(7,7))
    index=1
    for feature in featureNames:
        plt.subplot(2,2,index)
        for irisClass in classes:           
            sns.distplot(sortedTable[feature].loc[sortedTable['class']==irisClass], kde=False, label=str(irisClass))
        plt.legend(title = 'Class')
        index+=1
    plt.savefig(outpath+fileName)
    
"""
The function that cuts the dataset into 3 equal sections
    
Parameters:
    sortedTable (dataframe): the sorted dataset to be cut
        
Returns:
    tertiles (list): a list of the 3 cut datasets as dataframes
"""
def cutTertile(sortedTable):
    
    tertiles = []
    for index in range(3):
        #cut the first 50 data points
        cut = sortedTable.head(50)
        #remove the cut data points
        sortedTable = sortedTable.iloc[50:]
        tertiles.append(cut)
    return tertiles

"""
The function that generates the confusion matrix for the 3 classes for a specific feature.
It will also write the outputs to a file (f).
    
Parameters:
    tertiles (list): a list of 3 equally cut sorted dataframes
    feature (string): the feature to calculate the confusion matrix for
    f (file): the file to write the confusion matrix to
        
Returns: 
    Nothing
"""
def confusionMtx(tertiles, feature, f):
    f.write(feature + '\n')
    mtx = []
    for index in range(3):
        classNum = index+1
        #group each tertile by class and count by feature
        count = tertiles[index].groupby('class').count()[feature]
        mtx.append(count)
        #calculate true positives
        correct = count.at[classNum]/50
        f.write('Tertile ' + str(classNum) + ' True Positives for Class ' + str(classNum) + ': ' + str(correct) + '\n')
    mtx = pd.DataFrame(mtx).fillna(0)
    #change index to 1 - 3 so it's easier to identify
    confusion = mtx.set_index(pd.Index(range(1,len(mtx)+1))).T
    f.write(str(confusion) + '\n\n')

"""
The function used to plot out both a scatter plot and distribution curve for the classes by feature
    
Parameters: 
    dataframe (dataframe): the dataframe containing the dataset
    featureNames (list): the names of the features
        
Returns:
    Nothing
"""
def plotFeature(dataframe, featureNames, path, fileName):
    
    outpath = path + 'output//'
    sns.pairplot(dataframe, diag_kind = 'kde', hue='class', vars = featureNames)
    plt.title('Pairplot of Iris Features',x=-1.35, y= 3.3)
    plt.savefig(outpath+fileName)

"""
The function used to calculate the mahalanobis distance.
    
Parameters:
    irisClass (2D Numpy Array): the dataset for the iris class in numpy array format
        
Returns:
    d2 (numpy array): the squared mahalanobis distance vector of all points in the class
    
"""
def mahalanobis(irisClass):
    
    mu = np.mean(irisClass, axis=0)
    diff = irisClass - mu
    cov = np.cov(irisClass.T)
    inv_cov = np.linalg.inv(cov)
    mah_mtx = np.dot(np.dot(diff, inv_cov),diff.T)
    d2 = mah_mtx.diagonal()
    return d2

"""
The function used to calculate the critical value, used as an outlier threshold
    
Parameters:
    alpha (float64): the alpha value of the level of confidence
    n (int32): the sample size 
    numVars (int32): the number of features
        
Returns:
    cv (float64): the critical value
    
"""
def criticalValue(alpha,n,numVars):
    
    p = 1-alpha
    dfn = numVars
    dfd = n - numVars - 1
    inv_f = scipy.stats.f.ppf(p,dfn,dfd)
    num = dfn * (n-1)**2
    den = n*(dfd) + inv_f
    cv = num/den
    return cv

"""
The function used to remove outliers from an iris class
    
Parameters:
    irisClass (2D Numpy Array): the dataset for the iris class in numpy array format
    alpha (float64): the alpha value of the level of confidence
    numVars (int32): the number of features
        
Returns:
    leftover (2D Numpy Array): the dataset with outliers removed
"""    
def removeOutliers(irisClass, alpha, numVars):
    
    d2 = mahalanobis(irisClass)
    cv = criticalValue(alpha,len(irisClass),numVars)
    outliers = np.where(d2 > cv)[0]
    leftover = np.delete(irisClass,outliers, 0)
    return leftover 

"""
The function that plots the scatter of both the original and outlier removed datasets on top of each other
to visually identify removed outliers.
    
Parameters:
    irisClass (2D Numpy Array): the dataset for the iris class in numpy array format
    leftover (2D Numpy Array): the dataset with outliers removed
    feature1 (string): the first feature to plot against
    feature2 (string): the second feature to plot against
    featureNames (list): the names of the features
    className (string): the name of the iris class to plot 
    
Returns:
    Nothing
    """
def overlayLeftover(irisClass, leftover, feature1, feature2, featureNames, className, path):
    
    outpath = path + 'output//'
    label = [*range(0,50)]
    count =0
    #generate plot properties
    plt.title(className + ':' + featureNames[feature1] + ' vs ' + featureNames[feature2])
    plt.xlabel(featureNames[feature1])
    plt.ylabel(featureNames[feature2])
    
    #plot scatter of original data
    plt.scatter(irisClass[:,feature1],irisClass[:,feature2],label='Original Data', color= 'red', alpha=0.7)
    for x,y in zip(irisClass[:,feature1],irisClass[:,feature2]):       
        plt.annotate(label[count],
                     (x,y), 
                     textcoords='offset points', 
                     xytext=(0,10),
                     ha='center') 
        count +=1
    #overlay leftover data after outlier removal
    plt.scatter(leftover[:,feature1],leftover[:,feature2],label='Non-Outlier Data',color= 'blue',alpha=0.7)
    plt.legend()
    plt.savefig(outpath+className+"_scatter.png")

"""
The function used to remove the outliers and plot the scatters of all features by iris classes
    
Parameters: 
    allClasses (list): a list of the class datasets split into their own numpy arrays
    featureNames (list): the names of the features
        
Returns: 
    leftoverSet (list): a list of all datasets after outlier removal into their own numpy arrays
    leftoverSetClass (list): a list of corresponding classes to idenfity the classes of remainings points
"""
def printOverlayLeftover(allClasses, path, featureNames):
    
    classNames = ['Sentosa', 'Versicolor', 'Virginica']
    leftoverSet = []
    leftoverSetClass = []
    index = 1
    currentClass = 0
    
    #for all classes, remove outliers
    for iris in allClasses:
        leftover = removeOutliers(iris,0.05,4)
        features = [*range(4)]
        plt.figure(figsize=(15,10))
        #for each feature pair, as long as the features aren't the same, plot the scatter
        for feature1 in features:
            for feature2 in features:
                if (feature1 == feature2): 
                    continue
                else:                    
                    plt.subplot(2,3,index)
                    overlayLeftover(iris,leftover,feature1,feature2,featureNames,classNames[currentClass],path)
                    index+=1
            features.remove(feature1)
        
        #create a class mapping array so we can use for identification later
        currClassList = np.full(len(leftover), currentClass+1)
        currentClass += 1
        index=1
        leftoverSet.append(leftover)
        leftoverSetClass.append(currClassList)
    return leftoverSet, leftoverSetClass
        

"""
The function used to normalize the features of the classes
    
Parameters:
    allClasses (list): a list of the class datasets split into their own numpy arrays
        
Returns:
    allNorm (list): a list of the class datasets normalized for each feature
"""
def normalize(allClasses):
    
    allNorm = []
    for irisClass in allClasses:
        featureMin = np.amin(irisClass,axis=0)
        featureMax = np.amax(irisClass,axis=0)
        maxminDiff = featureMax-featureMin
        norm = (irisClass-featureMin)/maxminDiff
        allNorm.append(norm)
    return allNorm

"""
The function used to calculate the FDR for each feature.
    
Parameters:
    allClasses (list): a list of the class datasets split into their own numpy arrays
    featureNames (list): the names of the features
        
Returns:
    fdr_df (dataframe): a dataframe containing the FDR scores and the feature names
"""
def fdr(allClasses,featureNames):
    
    indices = [*range(len(allClasses))]
    fdr=0
    for index1 in indices:
        iris1 = allClasses[index1]
        for index2 in indices:
            iris2 = allClasses[index2]
            if(np.array_equal(iris1, iris2)):  
                continue
            else:
                fdrNUM = (np.mean(iris1,axis=0) - np.mean(iris2,axis=0))**2
                fdrDENOM = np.var(iris1,axis=0) + np.var(iris2,axis=0)
                fdr += fdrNUM/fdrDENOM
        indices.pop(0)
    fdr_df=pd.DataFrame(fdr,index = featureNames,columns=['FDR'])
    fdr_df.sort_values('FDR',ascending = False,inplace=True)
    return fdr_df

"""
The function used to calculate the Bhattacharrya Distance for each feature.
    
Parameters:
    allClasses (list): a list of the class datasets split into their own numpy arrays
    featureNames (list): the names of the features
        
Returns:
    bhattaDist_df (dataframe): a dataframe containing the bhattacharrya distances and the feature names
"""
def bhatta(allClasses,featureNames):
    
    classNames = ['sentosa', 'versi', 'virginica']
    bhattaDist = []
    comparisons= []
    indices = [*range(len(allClasses))]
    
    for index in indices:
        iris1 = allClasses[index]
        for index5 in indices:
            iris2 = allClasses[index5]
            if(np.array_equal(iris1, iris2)):
                continue
            else:
                left = (1/8)*((np.mean(iris1,axis=0) - np.mean(iris2,axis=0))**2)/((np.std(iris1,axis=0)+np.std(iris2,axis=0))/2)
                right = (0.5)*np.log((np.var(iris1,axis=0)+np.var(iris2,axis=0))/(2*np.std(iris1,axis=0)*np.std(iris2,axis=0)))
                bhatta = left+right
                classNames[index]
                classNames[index5]
                label = str(classNames[index] +"-" +classNames[index5])
                comparisons.append(label)
                bhattaDist.append(bhatta)
        indices.pop(0)
        bhattaDist_df = pd.DataFrame(bhattaDist,index=comparisons,columns=featureNames)
        
    return bhattaDist_df
                

"""
The function combines and converts the lists of numpy arrays to a dataframe
    
Parameters:
    npList1 (list): a list of the datasets split into their own numpy arrays
    npList2 (list): a list of the class labels split into their own numpy arrays
        
Returns:
    df (2D Numpy Array): a combined dataframe version of the two input lists of numpy arrays
"""
def recreateDF(npList1, npList2):
    
    featureNames = ['sepal_length','sepal_width','petal_length','petal_width','class']
    combine = []
    for index in range(3):
        addClass = np.hstack((npList1[index], npList2[index].reshape(-1, 1)))
        combine.append(addClass)
    df = pd.DataFrame(np.concatenate(combine))
    df.columns = featureNames
    return df

"""
The main method used to drive the program
"""
def Main():
    
    pd.set_option('display.max_columns', None)
    #turn off plots from showing
    plt.ioff()
    
    # Get path of iris dataset
    path = input('Full path of iris dataset location: ')
    file = 'iris.xlsx'
    iData = 'iris_data'
    iClass = 'class'
    featureNames = ['sepal_length','sepal_width','petal_length','petal_width']
    
    #open write files
    f= open(path+r'output\PA1_ISHENG_OUT.txt','w+')
    
    # Read iris dataset
    irisData = pd.read_excel(path+file,iData,header=None,names = featureNames)
    irisClass = pd.read_excel(path+file,iClass,header=None,names = ['class']) 
    iris_result = pd.concat([irisData, irisClass], axis=1, sort=False)
    
    sentosa = iris_result[iris_result['class'] == 1].iloc[:,:-1].values
    versi =  iris_result[iris_result['class'] == 2].iloc[:,:-1].values
    virginica =  iris_result[iris_result['class'] == 3].iloc[:,:-1].values
    
    allClasses = [sentosa, versi, virginica]
    
    f.write('Input File Head (First 5 data entries):\n')
    f.write(str(iris_result.head()) + '\n\n')
    print('Starting Program')
    
    f.write('Outputs:\n')
    f.write('\nScatter Plots are saved as PNG files in output folder \n\n')
    
    # Visualizing section
    print('Saving scatter plots of original data to output folder.')
    plotFeature(iris_result,featureNames,path,'original_data_pairplots.png')
    print('Done')
    
    # Sorting section
    print('Starting sorting section')
    f.write('Sorting Section: \n')
    for featureToSort in featureNames:
        sortedTable = sortTable(iris_result, featureToSort)
        tertiles = cutTertile(sortedTable)
        confusionMtx(tertiles,featureToSort,f)
    
    print('Saving distribution plots of original data to output folder.')
    plotHist(iris_result,featureNames, path,'original_data_dist.png')
    print('Done')
    
    # Outlier Removal Section
    print('Starting outlier removal process.')
    f.write('Outlier Removal: Scatter Plots saved as PNG file in output folder \n')
    removedOutliers, classList = printOverlayLeftover(allClasses,path,featureNames)
    removedOutliers_df = recreateDF(removedOutliers,classList)
    
    print('Saving scatter plots of dataset without outliers to output folder.')
    plotFeature(removedOutliers_df,featureNames,path,'removedOutlier_data_pairplots.png')
    
    print('Saving distribution plots of dataset without outliers to output folder.')
    plotHist(removedOutliers_df,featureNames,path,'removedOutlier_data_dist.png')
    print('Done')
    
    # Feature Ranking Section
    print('Starting feature ranking.')
    allNormClasses = normalize(removedOutliers)
    rankedFDR = fdr(allNormClasses,featureNames)
    rankedBhatta = bhatta(allNormClasses,featureNames)
    f.write('\n Feature Ranking Section: \n' + str(rankedFDR) + '\n')
    f.write('\n Bhattacharyya Distance: \n' + str(rankedBhatta))
    print('Assignment 1 finished, please find all outputs in the output folder.')
    
    # close write file
    f.close()
    
if __name__=="__main__": 
    Main()
