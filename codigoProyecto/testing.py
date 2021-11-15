#!/usr/bin/env python3

import pytest
from pyspark.sql.window import Window
from pyspark.sql import functions as FN
from .CodigoProyecto import *
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
   

    print("expected")
    expected_ds = spark_session.createDataFrame(
        [
            (None,'male',1.65,71.0,'wrestling',0), 
            ('Austria','female',1.68,75.3,'aquatics',1), 
            ('Brazil','male',1.63,62.0,'fencing',1,), 
            ("Chile",'male',1.83,84.0,'shooting',1)           
        ], 
        [  'country','sex','height','weight','sport','TieneMedalla'])  
    expected_ds.show()     
    expected_ds.printSchema()                                                             
    assert expected_ds.collect() == actual_ds.collect()


def test_imputacionIndicesGlobales(spark_session):
    sc=spark_session.sparkContext
    csvPath = Path("testDS1_loadcsv_world_indicators.csv")#Informacion de atletas
    indices_ds =cargarDataset1(csvPath) 
    transformed_DF=transformDatasetIndicesGlobales(indices_ds) 
    actual_ds = imputacionIndicesGlobales(transformed_DF)

    print("Actual DS test_transformDatasetIndicesGlobales ")
    actual_ds.show()

    print("Expected test_transformDatasetIndicesGlobales")
    expected_ds = spark_session.createDataFrame(
        [
            ('Afghanistan',49.3,2080.0,10872072,3.6), 
            ('Austria',39.1,9550.23,71307,13.0), 
            ('Chile',0.72,37465.45,127763267,12.7)           
        ], 
        [  'country','poverty_percent','gdp_per_capita','population','years_of_education'])  
    expected_ds.show(n=10)     
    expected_ds.printSchema()   
    assert expected_ds.collect() == actual_ds.collect()



def test_imputacionAtletas(spark_session):
    sc=spark_session.sparkContext
    csvPath = Path("testDS2_loadcsv_athletes.csv")#Informacion de atletas
    atletas_df =cargarDataset2(csvPath)  
    tranformed_DF=transformDatasetAtletas(atletas_df) 
    actual_ds = imputacionAtletas(tranformed_DF)
    print("Dataset del CSV")
    actual_ds.show(n=10)
    actual_ds.printSchema()
   

    print("expected")
    expected_ds = spark_session.createDataFrame(
        [
            ('Austria','female',1.68,75.3,'aquatics',1), 
            ('Brazil','male',1.63,62.0,'fencing',1,), 
            ("Chile",'male',1.83,84.0,'shooting',1)           
        ], 
        [  'country','sex','height','weight','sport','TieneMedalla'])  
    expected_ds.show()     
    expected_ds.printSchema()                                                             
    assert expected_ds.collect() == actual_ds.collect()

def test_joinDataframes(spark_session):
    indicesDF = spark_session.createDataFrame(
        [
            ('Afghanistan',49.3,2080.0,10872072,3.6), 
            ('Austria',39.1,9550.23,71307,13.0), 
            ('Chile',0.72,37465.45,127763267,12.7)           
        ], 
        [  'country','poverty_percent','gdp_per_capita','population','years_of_education']) 

    atletasDF = spark_session.createDataFrame(
        [
            ('Afghanistan','female',1.68,75.3,'aquatics',1), 
            ('Austria','male',1.63,62.0,'fencing',1), 
            ("Chile",'male',1.83,84.0,'shooting',1)           
        ], 
        [  'country','sex','height','weight','sport','TieneMedalla'])  

    actual_ds = joinDataframes(indicesDF,atletasDF)


    expected_ds = spark_session.createDataFrame(
        [
            ('Afghanistan',49.3,2080.0,10872072,3.6,'female',1.68,75.3,'aquatics',1), 
            ('Austria',39.1,9550.23,71307,13.0,'male',1.63,62.0,'fencing',1), 
            ('Chile',0.72,37465.45,127763267,12.7,'male',1.83,84.0,'shooting',1)           
        ], 
        [  'country','poverty_percent','gdp_per_capita','population','years_of_education','sex','height','weight','sport','TieneMedalla'])  
    assert expected_ds.collect() == actual_ds.collect()

def test_MuestraEstratificado(spark_session):

    UnionDFs = spark_session.createDataFrame(
        [
            ('chile',49.3,2080.0,10872072,3.6,'female',1.68,75.3,'aquatics',1), 
            ('Afghanistan',49.3,2080.0,10872072,3.6,'female',1.68,75.3,'aquatics',1), 
            ('brazil',49.3,2080.0,10872072,3.6,'female',1.68,75.3,'aquatics',0), 
            ('etiopia',49.3,2080.0,10872072,3.6,'female',1.68,75.3,'aquatics',0), 
            ('Afghanistan',49.3,2080.0,10872072,3.6,'female',1.68,75.3,'aquatics',0), 
            ('spain',49.3,2080.0,10872072,3.6,'male',1.68,75.3,'cycling',1), 
            ('nicaragua',49.3,2080.0,10872072,3.6,'male',1.68,75.3,'cycling',1), 
            ('costa rica',49.3,2080.0,10872072,3.6,'male',1.68,75.3,'cycling',1), 
            ('belize',49.3,2080.0,10872072,3.6,'male',1.68,75.3,'cycling',0), 
            ('francia',49.3,2080.0,10872072,3.6,'male',1.68,75.3,'cycling',0), 
                      
        ], 
        [  'country','poverty_percent','gdp_per_capita','population','years_of_education','sex','height','weight','sport','TieneMedalla'])  
    UnionDFs.show()
    UnionDFs.printSchema()

    actual_ds=MuestraEstratificado(UnionDFs)
    actual_ds.show()
    actual_ds.printSchema()

    expected_ds = spark_session.createDataFrame(
        [
            ('chile',49.3,2080.0,10872072,3.6,'female',1.68,75.3,'aquatics',1), 
            ('Afghanistan',49.3,2080.0,10872072,3.6,'female',1.68,75.3,'aquatics',1), 
            ('brazil',49.3,2080.0,10872072,3.6,'female',1.68,75.3,'aquatics',0), 
            ('etiopia',49.3,2080.0,10872072,3.6,'female',1.68,75.3,'aquatics',0), 
            ('spain',49.3,2080.0,10872072,3.6,'male',1.68,75.3,'cycling',1), 
            ('nicaragua',49.3,2080.0,10872072,3.6,'male',1.68,75.3,'cycling',1), 
            ('costa rica',49.3,2080.0,10872072,3.6,'male',1.68,75.3,'cycling',1), 
            ('belize',49.3,2080.0,10872072,3.6,'male',1.68,75.3,'cycling',0), 
            ('francia',49.3,2080.0,10872072,3.6,'male',1.68,75.3,'cycling',0), 
                      
        ], 
        [  'country','poverty_percent','gdp_per_capita','population','years_of_education','sex','height','weight','sport','TieneMedalla'])  

    print("printing limit")
    expected_ds.limit(30).show()
    expected_ds.show()
    expected_ds.printSchema()


    assert expected_ds.collect() == actual_ds.collect()

    

