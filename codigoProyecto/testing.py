#!/usr/bin/env python3

import pytest
from pyspark.sql.window import Window
from pyspark.sql import functions as FN
from .carga_preprocesado import *

def test_total_productos(spark_session):
    sc=spark_session.sparkContext
    csvPath1 = sys.argv[1]#Indices de desarrollo
    actual_ds = fn_total_productos(SubtotalsDF)    
    actual_ds.show()
    actual_ds.printSchema()

    #Output esperado
    #nombre|CantidadTotal
    expected_ds = spark_session.createDataFrame(
        [
            ("jugo" , 194), 
            ("sandia" , 130), 
            ("manzana" , 185), 
            ("galletas" , 239), 
            ("agua" , 118), 
            ("uvas" , 159), 
            ("cafe" , 176), 
            ("naranja" , 236) 
            
        ], 
        [   'nombre', 'CantidadTotal'])  
    expected_ds.show()                                                                                     
    

    assert expected_ds.collect() == actual_ds.collect()

