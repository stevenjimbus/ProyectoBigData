#!/bin/bash

spark-submit \
  --driver-class-path postgresql-42.2.14.jar \
  --jars postgresql-42.2.14.jar \
  carga_preprocesado.py \
  dataset1_world_indicators.csv \
  dataset2_athletes.csv \ 
 