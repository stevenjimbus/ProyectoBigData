#!/usr/bin/env python3

import pytest
from pyspark.sql.window import Window
from pyspark.sql import functions as FN
from .carga_preprocesado import *
import sys,os
from pathlib import Path

def test_loading_IndicesGlobalesDF(spark_session):
    sc=spark_session.sparkContext
    csvPath = Path("testDS1_loadcsv_world_indicators.csv")#Indices de desarrollo
    actual_ds =cargarDataset1(csvPath)   
    print("Dataset del CSV")
    actual_ds.show(n=10)
    actual_ds.printSchema()
   


    expected_ds = spark_session.createDataFrame(
        [
            ('Afghanistan',None,49.3,2080.0,10872072,3.6), 
            ('Austria','AUT',39.1,9550.23,71307,13.0), 
            ('Brazil','BRA',1.05,22574.0,None,10.8), 
            ('Chile','CHL',0.72,37465.45,127763267,12.7)           
        ], 
        [  'country','Code','poverty_percent','gdp_per_capita','population','years_of_education'])  
    expected_ds.show(n=10)     
    expected_ds.printSchema()   

    assert expected_ds.collect() == actual_ds.collect()


def test_loading_AtletasDF(spark_session):
    sc=spark_session.sparkContext
    csvPath = Path("testDS2_loadcsv_athletes.csv")#Informacion de atletas
    actual_ds =cargarDataset2(csvPath)   
    print("Dataset del CSV")
    actual_ds.show(n=10)
    actual_ds.printSchema()
   


    expected_ds = spark_session.createDataFrame(
        [
            (12,'Carlos Lara','AFG','male',31060,1.65,71.0,'wrestling',0,0,0,None,1377237,22600.2), 
            (34,'Jose Soto','AUT','female',33888,1.68,75.3,'aquatics',0,1,0,'Austria',13544345,55555.5), 
            (56,'Maria Perez','BRA','male',33176,1.63,62.0,'fencing',1,2,0,'Brazil',64523,33333.3), 
            (78,'Abdullah Alrashidi','CHL','male',23244,1.83,84.0,'shooting',None,0,1,'Chile',7897789,99999.99)           
        ], 
        [  'id','name','nationality','sex','dob','height','weight','sport','gold','silver','bronze','country','countrypopulation','countrygdp_per_capita'])  
    expected_ds.show()     
    expected_ds.printSchema()                                                             
    assert expected_ds.collect() == actual_ds.collect()

def test_transformDatasetIndicesGlobales(spark_session):
    sc=spark_session.sparkContext
    csvPath = Path("testDS1_loadcsv_world_indicators.csv")#Informacion de atletas
    indices_ds =cargarDataset1(csvPath) 
    actual_ds=transformDatasetIndicesGlobales(indices_ds) 

    print("Actual DS test_transformDatasetIndicesGlobales ")
    actual_ds.show()

    print("Expected test_transformDatasetIndicesGlobales")
    expected_ds = spark_session.createDataFrame(
        [
            ('Afghanistan',49.3,2080.0,10872072,3.6), 
            ('Austria',39.1,9550.23,71307,13.0), 
            ('Brazil',1.05,22574.0,None,10.8), 
            ('Chile',0.72,37465.45,127763267,12.7)           
        ], 
        [  'country','poverty_percent','gdp_per_capita','population','years_of_education'])  
    expected_ds.show(n=10)     
    expected_ds.printSchema()   

    


    assert expected_ds.collect() == actual_ds.collect()



def test_transformDatasetAtletas(spark_session):
    sc=spark_session.sparkContext
    csvPath = Path("testDS2_loadcsv_athletes.csv")#Informacion de atletas
    atletas_df =cargarDataset2(csvPath)  
    actual_ds=transformDatasetAtletas(atletas_df) 
    print("Dataset del CSV")
    actual_ds.show(n=10)
    actual_ds.printSchema()
   


    expected_ds = spark_session.createDataFrame(
        [
            (None,'male',1.65,71.0,'wrestling',0), 
            ('Austria','female',1.68,75.3,'aquatics',1), 
            ('Brazil','male',1.63,62.0,'fencing',1,), 
            (None,'male',1.83,84.0,'shooting',1)           
        ], 
        [  'country','sex','height','weight','sport','TieneMedalla'])  
    expected_ds.show()     
    expected_ds.printSchema()                                                             
    assert expected_ds.collect() == actual_ds.collect()
