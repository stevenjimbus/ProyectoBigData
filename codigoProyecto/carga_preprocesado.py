from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import col, date_format, udf
from pyspark.sql.types import (DateType, IntegerType, FloatType, StringType,
                               StructField, StructType, TimestampType)

spark = SparkSession.builder.appName("carga_y_preprocesado").getOrCreate()


def cargarDataset1():
    #StringType, IntegerType, FloatType, DecimalType, StructField, StructType
    worldIndDF = spark \
        .read \
        .format("csv") \
        .option("path", "dataset1_world_indicators.csv") \
        .option("header", True) \
        .schema(StructType([
                    StructField("Entity",StringType()),
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

def cargarDataset2():
    AthletesDF = spark \
        .read \
        .format("csv") \
        .option("path", "dataset2_athletes.csv") \
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

def transformDatasetAtletas(Atletas_DF):
    sumDF=Atletas_DF.withColumn('total_Medallas',Atletas_DF.gold + Atletas_DF.silver + Atletas_DF.bronze )
    #sumDF.show(n=50)
    binaryLabelDF = sumDF.withColumn('TieneMedalla', f.when(f.col('total_Medallas') > 0, 1).otherwise(0))
    binaryLabelDF.show(n=50)
    cleanDF = binaryLabelDF.select("country","sex","height","weight","sport","TieneMedalla")
    cleanDF.show()
    return True


def main():
    IndicesDesarrllo_por_paisDF = cargarDataset1()
    AtletasDF = cargarDataset2()
    transformDatasetAtletas(AtletasDF)

    return True

if __name__ == '__main__':
    main()
