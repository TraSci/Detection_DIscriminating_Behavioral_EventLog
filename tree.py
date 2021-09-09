from sklearn import tree # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from matplotlib import pyplot as plt
from dtreeviz.trees import dtreeviz 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
import graphviz 
import numpy as np
import pandas as pd


class decisionTree():

    def __init__(self, data1, data2):
        self.real=data1
        self.sim=data2

    def getValueData(self,feature):
        return [sum(self.real[feature]), sum(self.sim[feature])]

    def featureTimeSelection(self, key):
        meanR=np.mean(self.real[key])
        meanS=np.mean(self.sim[key])
        medianR=np.median(self.real[key])
        medianS=np.median(self.sim[key])
        result=True
        if (max([meanR, meanS])-min([meanR, meanS]))<=5 and (max([medianR, medianS])-min([medianR, medianS]))<=5:
            result=True
        else:
            result=False
        
        return result

    def featureSelection(self):
        thresholdR=0.95*len(self.real)
        thresholdS=0.95*len(self.sim)
        X=self.dataset
        delete=['Log']
        columns=(self.dataset.drop('Log', axis=1)).columns
        self.columnsTrain=[]
        string=""
        j=0
        for i in range(len(columns)):
            key=columns[i]
            if "time:" not in key:
                if 0<=sum(self.real[key])<=(len(self.real)-thresholdR) and 0<=sum(self.sim[key])<=(len(self.sim)-thresholdS):
                    delete.append(key)
                elif thresholdR<=sum(self.real[key])<=(len(self.real)) and thresholdS<=sum(self.sim[key])<=len(self.sim):
                    delete.append(key)
                else:
                    string=string + str(j) + ". " + columns[i] + "   " + str(sum(self.real[key])) +  "    "  + str(sum(self.sim[key])) + "\n"
                    self.columnsTrain.append(columns[i])
                    j=j+1
            else:
                if self.featureTimeSelection(key):
                    delete.append(key)
                else:
                    string=string + str(j) + ". " + columns[i] + "   " + str(np.mean(self.real[key])) +  "    "  + str(np.mean(self.sim[key])) + "\n"
                    j=j+1
        X=X.drop(delete, axis=1)
        return X, string

    def classifier(self):
        self.dataset = pd.concat([self.real, self.sim])
        X,string = self.featureSelection()
        y = self.dataset.Log
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.30, random_state=1)
        min=int(len(self.dataset)*0.1)
        self.clf = tree.DecisionTreeClassifier(min_samples_split=min, min_samples_leaf=min)
        # Train Decision Tree Classifer
        self.clf = self.clf.fit(self.X_train,self.y_train)
        #Predict the response for test dataset
        self.y_pred = self.clf.predict(self.X_test)
        dot_data = tree.export_graphviz(self.clf, out_file=None, feature_names=self.X_train.columns,class_names=[ 'real', 'simulation'], filled=True, rounded=True)
        return metrics.accuracy_score(self.y_test, self.y_pred), string


    def plot_difference(self):
        X = self.dataset.drop('Log', axis=1)# Features
        y = self.dataset.Log
        accuracy=1  
        feature=[] 
        treeviz=[] 
        min=int(len(self.dataset)*0.1)                            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
        clf = tree.DecisionTreeClassifier(min_samples_split=min, min_samples_leaf=min)
        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)
        #Predict the response for test dataset
        y_pred = clf.predict(X_test)
        accuracy=metrics.accuracy_score(y_test, y_pred)
        index=clf.tree_.feature[0]  ### radice dell'albero
        root=X.columns[index]
        tupla=(root, str(round(metrics.accuracy_score(y_test, y_pred),2)))
        feature.append(tupla)
        treeviz=[clf, X,y]
        return treeviz

    def plotConfusionMatrix(self):
        disp = metrics.plot_confusion_matrix(self.clf, self.X_test, self.y_test,display_labels=['real', 'simulation'],cmap=plt.cm.Blues)
        disp.ax_.set_title("Confusion Matrix")
        plt.savefig('confusion_matrix.jpg')
        plt.show()

    def feature_importances(self):
        for name, val in zip(self.columnsTrain, self.clf.feature_importances_):
            if val>0.1:
                print(name + "->" + str(val))

    def plotTree(self, save=True):
        dot_data = tree.export_graphviz(self.clf, out_file=None, feature_names=self.columnsTrain,class_names=[ 'real', 'simulation'], filled=True, rounded=True) 
        graph = graphviz.Source(dot_data) 
        graph.render("tree")
