from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.enhancement.roles import algorithm as roles_discovery
from pm4py.objects.log.util import sorting
from pm4py.objects.log.util import func

from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.util import constants
import pandas as pd
import re

class checkCostraints:

    def __init__(self, path_costraints, df, log):
        try:
            self.constraints = pd.read_csv(path_costraints, sep=',')
            self.output=df
            #self.log=xes_importer.apply(path_log) 
            self.log=log
        except FileNotFoundError:
            print("ERROR: File not found")


    def insertCostraints(self):
        for i in range(len(self.constraints)):
            column=None
            if re.search('^Init', self.constraints['Constraint'][i]):
                column=self.Init(self.constraints['Activation'][i])
                
            if re.search('^End', self.constraints['Constraint'][i]):
                column=self.End(self.constraints['Activation'][i])
                
            if re.search('^Response', self.constraints['Constraint'][i]):
                column=self.Response(self.constraints['Activation'][i],self.constraints['Target'][i])
                
            if re.search('^AlternateResponse', self.constraints['Constraint'][i]):
                column=self.AlternateResponse(self.constraints['Activation'][i],self.constraints['Target'][i])
                
            if re.search('^Precedence', self.constraints['Constraint'][i]):
                column=self.Precedence(self.constraints['Activation'][i],self.constraints['Target'][i])
                
            if re.search('^AlternatePrecedence', self.constraints['Constraint'][i]):
                column=self.AlternatePrecedence(self.constraints['Activation'][i],self.constraints['Target'][i])
                
            if re.search('^CoExistence', self.constraints['Constraint'][i]):
                column=self.CoExistence(self.constraints['Activation'][i],self.constraints['Target'][i])
                
            if re.search('^Succession', self.constraints['Constraint'][i]):
                column=self.Succession(self.constraints['Activation'][i],self.constraints['Target'][i])
            if column!=None:
                self.output[self.constraints['Constraint'][i]]=column

        return self.output

    def Init(self, activity):
        column=[]
        for trace in self.log:
            if trace[0]['concept:name']==activity:
                column.append(True)
            else:
                column.append(False)
        return column


    def End(self, activity):
        column=[]
        for trace in self.log:
            if trace[0]['concept:name']==activity:
                column.append(True)
            else:
                column.append(False)
        return column

    def CoExistence(self, a, b):
        column=[]
        for trace in self.log:
            flagA= False 
            flagB=False
            for event in range(len(trace)):
                if trace[event]['concept:name']==a:
                    flagA=True
                if trace[event]['concept:name']==b:
                    flagB=True
            if flagA and flagB:
                column.append(True)
            else: 
                column.append(False)
        return column

    def AlternatePrecedence(self, a, b):
        column=[]
        for trace in self.log: 
            flagP= True
            event=0
            while event<len(trace) and flagP:
                if trace[event]['concept:name']==b:
                    ### cerco una 'a' precedente
                    i=event-1
                    flagA=False
                    while flagA==False and flagP and i>0:
                        if trace[i]['concept:name']==a:
                            flagA=True
                        if trace[i]['concept:name']==b:
                            flagP=False
                        i=i-1
                event=event+1
            column.append(flagP)
        return column

    def Response(self, a, b):
        column=[]
        for trace in self.log:
            flagR= True
            event=0 
            while event<len(trace) and flagR:
                if trace[event]['concept:name']==a:
                    ### cerco una b successiva
                    i=event+1
                    flagB=False
                    while flagB==False and i<len(trace):
                        if trace[i]['concept:name']==b:
                            flagB=True ### esco dal ciclo
                        i=i+1
                    flagR=flagB
                event=event+1
            column.append(flagR)
        return column

    def AlternateResponse(self,a, b):
        column=[]
        for trace in self.log: 
            flagR= True
            event=0
            while event<len(trace) and flagR:
                if trace[event]['concept:name']==a:
                    ### cerco una b successiva
                    i=event+1
                    flagB=False
                    while flagB==False and flagR and i<len(trace):
                        if trace[i]['concept:name']==b:
                            flagB=True ### esco dal ciclo
                        if trace[i]['concept:name']==a:
                            flagR=False
                        i=i+1
                    flagR=flagB
                event=event+1
            column.append(flagR)
        return column

    def Precedence(self,a, b):
        column=[]
        for trace in self.log: 
            flagP= True
            event=0
            while event<len(trace) and flagP:
                if trace[event]['concept:name']==b:
                    ### cerco una 'a' precedente
                    i=event-1
                    flagA=False
                    while flagA==False and flagP and i>0:
                        if trace[i]['concept:name']==a:
                            flagA=True
                        i=i-1
                    flagP=flagA
                event=event+1
            column.append(flagP)
        return column

    def Succession(self, a, b):
        column=[]
        for trace in self.log:
            flagS=True
            event=0
            while event<len(trace) and flagS:
                if trace[event]['concept:name']==a:
                    j=event+1
                    flagB=False
                    while j<len(trace) and flagB==False:
                        if trace[j]['concept:name']==b:
                            flagB=True
                        j=j+1
                    flagS=flagB
                event=event+1      
            column.append(flagS)
        return column


    def write_csv(self, name):
        self.output.to_csv(name, index=False)