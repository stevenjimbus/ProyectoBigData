#!/usr/bin/env python3

import pytest
from pyspark.sql.window import Window
from pyspark.sql import functions as FN
from .carga_preprocesado import *
import sys,os
from pathlib import Path

def test_loading(spark_session):
    sc=spark_session.sparkContext
    csvPath1 = Path("testDS1_loadcsv_world_indicators.csv")#sys.argv[1]#Indices de desarrollo
    actual_ds =cargarDataset1(csvPath1)   
    print("Dataset del CSV")
    actual_ds.show(n=10)
    actual_ds.printSchema()
    print("hasta aqui voy bien")


    expected_ds = spark_session.createDataFrame(
        [
            ('Afghanistan',None,49.3,2080,10872072,3.6), 
            ('Austria','AUT',39.1,9550,71307,13.0), 
            ('Brazil','BRA',1.05,22574,None,10.8), 
            ('Chile','CHL',0.72,37465,127763267,12.7)           
        ], 
        [  'country','Code','poverty_percent','gdp_per_capita','population','years_of_education'])  
    expected_ds.show()     
    expected_ds.printSchema()                                                                                
    

    assert expected_ds.collect() == actual_ds.collect()

