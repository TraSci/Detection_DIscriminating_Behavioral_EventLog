from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.util import sorting
from pm4py.objects.log.util import func

from pm4py.util import constants
import pandas as pd

from sklearn import tree 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np


class time():

    def __init__(self, log, df, type):
        self.log=log
        self.df=df
        self.size=len(df)
        self.type=type

    def timeProcessR(self):
        timeProcess=[]
        for trace in self.log:
            start=trace[0]['time:timestamp']
            end=trace[len(trace)-1]['time:timestamp']
            diff=self.days_hours_minutes(end-start)
            hour=diff[0]*1440 + diff[1]*60 + diff[2]
            timeProcess.append(hour)
        self.df['timeProcess']=timeProcess

    def days_hours_minutes(self, td):
        return td.days, td.seconds//3600, (td.seconds//60)%60

    def find_task(self):
        tasks=set()
        for i,c in enumerate(self.log):
            for j,e in enumerate(c):
                tasks.add(self.log[i][j]['concept:name'])
        return tasks

    def create_dfTime(self):
        data=dict()
        for elem in self.find_task():
            column="time:" + elem
            data[column]=[0] * self.size
        timedf= pd.DataFrame(data=data)   
        return timedf

    def timeFeatures(self):
        if self.type=='R':
            return self.timeTask_ProcessR()
        else:
            return self.timeTask_ProcessS()

    def timeTask_ProcessR(self):
        self.timeProcessR()
        i=0
        timedf=self.create_dfTime()
        for trace in self.log:
            for event in range(1,len(trace)):
                before=trace[event-1]['time:timestamp']
                now=trace[event]['time:timestamp']
                diff=self.days_hours_minutes(now-before)
                hour=diff[0]*1440 + diff[1]*60 + diff[2]
                column="time:"+trace[event]['concept:name']
                timedf[column][i]=hour
            i=i+1
        self.df=pd.concat([self.df, timedf], axis=1)
        return self.df

    def timeProcessS(self):
        timeProcess=[]
        for trace in self.log:
            start=trace[0]['Relative Time']
            end=trace[len(trace)-1]['Relative Time']
            timeProcess.append(end-start)
        self.df['timeProcess']=timeProcess

    def timeTask_ProcessS(self):
        self.timeProcessS()
        i=0
        timedf=self.create_dfTime()
        for trace in self.log:
            for event in range(1, len(trace)):
                start=trace[event-1]['Relative Time']
                end=trace[event]['Relative Time']
                column="time:"+ trace[event]['concept:name']
                timedf[column][i]=(end-start)
            i=i+1
        self.df=pd.concat([self.df, timedf], axis=1)
        return self.df


    def write_csv(self, name):
        self.output.to_csv(name, index=False)



class timeConstraint():

    def __init__(self, path_costraints, df, path_log, type):
        try:
            self.constraints = pd.read_csv(path_costraints, sep=',')
            self.df=df
            self.log=xes_importer.apply(path_log)
            self.type=type
        except FileNotFoundError:
            print("ERROR: File not found")

    def addTimeConstraint(self):
        self.target_activation()
        if self.type=='R':
            return self.constraintR()
        else:
            return self.constraintS()

    def target_activation(self):
        self.vincoli=dict()
        for i in range(len(self.constraints)):
            key=self.constraints['Constraint'][i]
            self.vincoli[key]=(self.constraints['Activation'][i], self.constraints['Target'][i])

    def create_dfTime(self):
        data=dict()
        for key in self.vincoli:
            column="time:" + key
            data[column]=[0] * len(self.df)
        timeV= pd.DataFrame(data=data) 
        return timeV

    def days_hours_minutes(self, td):
        return td.days, td.seconds//3600, (td.seconds//60)%60

    def constraintR(self):
        timeV=self.create_dfTime() 
        for key in self.vincoli:
            for trace in range(0, len(self.df)):
                if self.df[key][trace]==False:
                    timeV["time:"+key][trace]=-1
                else:
                    hour=0
                    activation=self.vincoli[key][0]
                    target=self.vincoli[key][1]
                    timeA=-1
                    timeB=-1
                    j=0
                    while j<len(self.log[trace]):
                        if self.log[trace][j]['concept:name']==activation and timeA==-1:
                            timeA=self.log[trace][j]['time:timestamp']
                        if self.log[trace][j]['concept:name']==target and timeB==-1:
                            timeB=self.log[trace][j]['time:timestamp']
                        j=j+1

                    if timeB==-1 or timeA==-1:
                        timeV["time:"+key][trace]=-1
                    else:
                        if timeA>timeB:
                            diff=self.days_hours_minutes(timeA-timeB)
                            hour=diff[0]*1440 + diff[1]*60 + diff[2]
                        else: 
                            diff=days_hours_minutes(timeB-timeA)
                            hour=diff[0]*1440 + diff[1]*60 + diff[2]
                        timeV["time:"+key][trace]=hour
        
        self.df = pd.concat([self.df, timeV], axis=1)
        return self.df
    
    def constraintS(self):
        timeV=self.create_dfTime()
        for key in self.vincoli:
            for trace in range(0, len(self.df)):
                if self.df[key][trace]==False:
                    timeV["time:"+key][trace]=-1
                else:
                    activation=self.vincoli[key][0]
                    target=self.vincoli[key][1]
                    timeA=-1
                    timeB=-1
                    j=0
                    while j<len(self.log[trace]):
                        if self.log[trace][j]['concept:name']==activation and timeA==-1:
                            timeA=self.log[trace][j]['Relative Time']
                        if self.log[trace][j]['concept:name']==target and timeB==-1:
                            timeB=self.log[trace][j]['Relative Time']
                        j=j+1
                    if timeB==-1 or timeA==-1:
                        timeV["time:"+key][trace]=-1
                    else:
                        diff=timeB-timeA
                        if diff<0:
                            diff=diff*(-1)
                        timeV["time:"+key][trace]=diff
        self.df = pd.concat([self.df, timeV], axis=1)
        return self.df