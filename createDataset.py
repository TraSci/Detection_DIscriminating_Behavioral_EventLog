from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.util import sorting
from pm4py.objects.log.util import func

from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.filtering.log.attributes import attributes_filter

from pm4py.algo.discovery.footprints import algorithm as footprints_discovery

from pm4py.util import constants
import pandas as pd
import os

class createDataset():

    def __init__(self, path_real, path_sim):
        try:
            #self.costraints = pd.read_csv(path_costraints, sep=';')
            self.real=xes_importer.apply(path_real)
            self.sim=xes_importer.apply(path_sim)
            self.sizeReal=len(self.real)
            self.sizeSim=len(self.sim)
            self.attrib=set()
        except FileNotFoundError:
            print("ERROR: File not found")

    def discovery_features_xes(self, log):
        for trace in log:
            for event in range(len(trace)-1):
                column=trace[event]['concept:name']+'>'+trace[event+1]['concept:name']
                self.attrib.add(column)
    
    def addTask(self, log):
        for i,c in enumerate(log):
            for j,e in enumerate(c):
                self.attrib.add(log[i][j]['concept:name'])


    def discovery(self):
        self.discovery_features_xes(self.real)
        self.discovery_features_xes(self.sim)
        self.addTask(self.real)
        self.addTask(self.sim)
    
    def create_empty_df(self, size):
        data=dict()
        for elem in self.attrib:
            data[elem]=[0] * size
        return pd.DataFrame(data=data)

    def create_df(self, log, size):
        self.discovery()
        dataframe=self.create_empty_df(size)
        i=0
        for trace in log:
            for event in range(len(trace)):
                if event<len(trace)-1:
                    column=trace[event]['concept:name']+'>'+ trace[event+1]['concept:name']
                    dataframe[column][i]=dataframe[column][i]+1
                task=trace[event]['concept:name']
                dataframe[task][i]=dataframe[task][i]+1
            i=i+1
        return dataframe
    
    def dataset(self, name1, name2):
        real=self.create_df(self.real, self.sizeReal)
        real["Log"]=[0]*self.sizeReal
        sim=self.create_df(self.sim, self.sizeSim)
        sim["Log"]=[1]*self.sizeSim
        deleteAnd=self.delete_and()
        #print(deleteAnd)
        real=real.drop(deleteAnd, axis=1)
        sim=sim.drop(deleteAnd, axis=1)
        self.write_csv(real, name1)
        self.write_csv(sim, name2)
        print("Files csv generated")
        return real, sim

    def delete_and(self):
        fp_real = footprints_discovery.apply(self.real, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
        fp_sim = footprints_discovery.apply(self.sim, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
        delete_and=[]
        intersection=list(fp_real['parallel'].intersection(fp_sim['parallel']))
        for i in range(len(intersection)):
            task1=intersection[i][0]
            task2=intersection[i][1]
            delete_and.append(task1+'>'+task2)
        return delete_and

    def csv_to_xes(self):
        from pm4py.objects.log.exporter.xes import exporter as xes_exporter
        xes_exporter.apply(event_log, 'path')


    def write_csv(self, df, name):
        df.to_csv(name, index=False)


    def getReal(self):
        return self.real

    def getSim(self):
        return self.sim