from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import col, date_format, udf, isnan, when, count, isnull
from pyspark.sql.types import (DateType, IntegerType, FloatType, StringType,
                               StructField, StructType, TimestampType)
import sys

spark = SparkSession.builder.appName("carga_y_preprocesado").getOrCreate()


def cargarDataset1(csvPath1):
    #StringType, IntegerType, FloatType, DecimalType, StructField, StructType
    worldIndDF = spark \
        .read \
        .format("csv") \
        .option("path", csvPath1) \
        .option("header", True) \
        .schema(StructType([
                    StructField("country",StringType()),
                    StructField("Code",StringType()),
                    StructField("poverty_percent",FloatType()),
                    StructField("gdp_per_capita",FloatType()),
                    StructField("population",IntegerType()),
                    StructField("years_of_education",FloatType())])) \
        .load()
    print('Qty Filas: {}\n Cantidad Columnas: {}'.format(worldIndDF.count(), len(worldIndDF.columns)))
    worldIndDF.printSchema()
    worldIndDF.show(truncate=False,n=3)

    return worldIndDF

def cargarDataset2(csvPath2):
    AthletesDF = spark \
        .read \
        .format("csv") \
        .option("path", csvPath2) \
        .option("dateFormat", "MM-dd-yyyy")\
        .option("header", True) \
        .schema(StructType([
                    StructField("id",IntegerType()),
                    StructField("name",StringType()),
                    StructField("nationality",StringType()),
                    StructField("sex",StringType()),
                    StructField("dob",DateType()),
                    StructField("height",FloatType()),
                    StructField("weight",FloatType()),
                    StructField("sport",StringType()),
                    StructField("gold",IntegerType()),
                    StructField("silver",IntegerType()),
                    StructField("bronze",IntegerType()),
                    StructField("country",StringType()),
                    StructField("countrypopulation",FloatType()),
                    StructField("countrygdp_per_capita",FloatType()),

                    ])) \
        .load()
    print('Qty Filas: {}\n Cantidad Columnas: {}'.format(AthletesDF.count(), len(AthletesDF.columns)))
    AthletesDF.printSchema()
    AthletesDF.show(truncate=False,n=3)    
    return AthletesDF

def transformDatasetIndicesGlobales(Indices_DF):
    transformedDF = Indices_DF.select("country","poverty_percent","gdp_per_capita","population","years_of_education")
    transformedDF.show()
    return transformedDF

def transformDatasetAtletas(Atletas_DF):
    sumDF=Atletas_DF.withColumn('total_Medallas',Atletas_DF.gold + Atletas_DF.silver + Atletas_DF.bronze )
    binaryLabelDF = sumDF.withColumn('TieneMedalla', f.when(f.col('total_Medallas') > 0, 1).otherwise(0))
    binaryLabelDF.show(n=50)
    transformedDF = binaryLabelDF.select("country","sex","height","weight","sport","TieneMedalla")
    transformedDF.show()
    return transformedDF

def imputacionIndicesGlobales(indicesGlobalesDF):   
    print("Tamano Dataframe indicesGlobalesDF",(indicesGlobalesDF.count(), len(indicesGlobalesDF.columns)))
    print("Cantidad de valores NaN por columna indicesGlobalesDF")
    indicesGlobalesDF.select([count(when(isnan(c), c)).alias(c) for c in indicesGlobalesDF.columns]).show()
    print("Cantidad de valores Null por columna indicesGlobalesDF")
    indicesGlobalesDF.select([count(when(col(c).isNull(), c)).alias(c) for c in indicesGlobalesDF.columns]).show()
    cleanDF = indicesGlobalesDF.na.drop()
    print("Dataframe preprocesado de Indices Globales")
    cleanDF.show()
    print("Tamano Dataframe preprocesado de Indices Globales ",(cleanDF.count(), len(cleanDF.columns)))

    return cleanDF

def imputacionAtletas(atletasDF):
    
    print("Tamano Dataframe atletasDF",(atletasDF.count(), len(atletasDF.columns)))
    print("Cantidad de valores NaN por columna atletasDF")
    atletasDF.select([count(when(isnan(c), c)).alias(c) for c in atletasDF.columns]).show()
    print("Cantidad de valores Null por columna atletasDF")
    atletasDF.select([count(when(col(c).isNull(), c)).alias(c) for c in atletasDF.columns]).show()
    cleanDF = atletasDF.na.drop()
    print("Dataframe preprocesado de Atletas ")
    cleanDF.show()
    print("Tamano Dataframe preprocesado de Atletas",(cleanDF.count(), len(cleanDF.columns)))
    

    return True

def main():
    csvPath1 = sys.argv[1]#Indices de desarrollo
    csvPath2 = sys.argv[2]#Informacion de atletas
    ###Cargar datos desde .csv
    IndicesDesarrllo_por_paisDF = cargarDataset1(csvPath1)#Cargar RawDataset1
    AtletasDF = cargarDataset2(csvPath2)#Cargar RawDataset2
    ####Seleccionar features deseados para el modelo predictivo
    mainColumnsIndicesDF = transformDatasetIndicesGlobales(IndicesDesarrllo_por_paisDF)#Seleccionar features deseados para el modelo predictivo
    mainColumnsAtletasDF = transformDatasetAtletas(AtletasDF)#Seleccionar features deseados para el modelo predictivo
    #Imputacion de valores faltantes
    imputacionIndicesGlobales(mainColumnsIndicesDF)
    imputacionAtletas(mainColumnsAtletasDF)


    return True

if __name__ == '__main__':
    main()
