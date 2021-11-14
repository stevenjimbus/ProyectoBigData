import findspark
findspark.init('/usr/lib/python3.7/site-packages/pyspark')

from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import col, date_format, udf, isnan, when, count, isnull
from pyspark.sql.types import (DateType, IntegerType, FloatType, StringType,
                               StructField, StructType, TimestampType,LongType,DoubleType)

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.feature import VectorAssembler
import sys


spark = SparkSession \
    .builder \
    .appName("Basic JDBC pipeline") \
    .config("spark.driver.extraClassPath", "postgresql-42.2.14.jar") \
    .config("spark.executor.extraClassPath", "postgresql-42.2.14.jar") \
    .getOrCreate()

def cargarDataset1(csvPath1):
    
    worldIndDF = spark \
        .read \
        .format("csv") \
        .option("path", csvPath1) \
        .option("header", True) \
        .schema(StructType([
                    StructField("country",StringType()),
                    StructField("Code",StringType()),
                    StructField("poverty_percent",DoubleType()),
                    StructField("gdp_per_capita",DoubleType()),
                    StructField("population",LongType()),
                    StructField("years_of_education",DoubleType())])) \
        .load()
    print("Carga inicial de Indicadores GlobalesDF")
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
                    StructField("id",LongType()),
                    StructField("name",StringType()),
                    StructField("nationality",StringType()),
                    StructField("sex",StringType()),
                    StructField("dob",LongType()),
                    StructField("height",DoubleType()),
                    StructField("weight",DoubleType()),
                    StructField("sport",StringType()),
                    StructField("gold",LongType()),
                    StructField("silver",LongType()),
                    StructField("bronze",LongType()),
                    StructField("country",StringType()),
                    StructField("countrypopulation",LongType()),
                    StructField("countrygdp_per_capita",DoubleType()),

                    ])) \
        .load()
    print("Carga inicial de AtletasDF")
    print('Qty Filas: {}\n Cantidad Columnas: {}'.format(AthletesDF.count(), len(AthletesDF.columns)))
    AthletesDF.printSchema()
    AthletesDF.show(truncate=False,n=3)    
    return AthletesDF

def transformDatasetIndicesGlobales(Indices_DF):
    print("Seleccionamos columnas especificas de Indices_DF (hacemos drop a columna CODE)")
    transformedDF = Indices_DF.select("country","poverty_percent","gdp_per_capita","population","years_of_education")
    transformedDF.show()
    return transformedDF

def transformDatasetAtletas(Atletas_DF):
    print("Raw")
    Atletas_DF.show()
    print("AtletasSinNulls_DF") 
    AtletasSinNulls_DF = Atletas_DF.na.fill(value=0,subset=["gold","silver","bronze"])
    AtletasSinNulls_DF.show()
    print("Sumamos la cantidad de medallas por participantes")   
    sumDF=AtletasSinNulls_DF.withColumn('total_Medallas',(AtletasSinNulls_DF.gold + AtletasSinNulls_DF.silver + AtletasSinNulls_DF.bronze).cast(LongType()) )
    print("Creamos columna TieneMedalla: ###Participante Ganó medalla -> 1  ### Participante No Ganó medalla -> 0###")
    binaryLabelDF = sumDF.withColumn('TieneMedalla', (f.when(f.col('total_Medallas') > 0, 1).otherwise(0)).cast(LongType()))
    binaryLabelDF.show()
    print("Seleccionamos las columnas que nos interesa del dataset binario Ganó/No Ganó medalla")
    transformedDF = binaryLabelDF.select("country","sex","height","weight","sport","TieneMedalla")
    transformedDF.show()
    return transformedDF

def imputacionIndicesGlobales(indicesGlobalesDF):   
    print("Tamano Dataframe indicesGlobalesDF",(indicesGlobalesDF.count(), len(indicesGlobalesDF.columns)))
    print("Cantidad de valores NaN por columna indicesGlobalesDF")
    indicesGlobalesDF.show()
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
    return cleanDF



def joinDataframes(DF1,DF2):
    jointDFs = DF1.join(DF2, ['country'])
    jointDFs.show()
    print("Tamano Dataframe jointDFs",(jointDFs.count(), len(jointDFs.columns)))
    return jointDFs




def MuestraEstratificado(UnionDFs):
    uniqueSex = UnionDFs.select("sex").distinct()   
    listauniqueSex = [row.sex for row in uniqueSex.collect()]
    print("listauniquSex:",listauniqueSex)


    uniqueSports = UnionDFs.select("sport").distinct()
    listauniqueSports = [row.sport for row in uniqueSports.collect()]
    print("listauniqueSports:",listauniqueSports)


    uniqueMedallas = UnionDFs.select("TieneMedalla").distinct()
    listauniqueMedallas = [row.TieneMedalla for row in uniqueMedallas.collect()]
    print("listauniqueSports:",listauniqueMedallas)

    print("empty DF")
    emptyRDD = spark.sparkContext.emptyRDD()
    sampledf = spark.createDataFrame(emptyRDD,UnionDFs.schema)
    sampledf.printSchema()    

    print("Inicio de for loop")
    for deporte in listauniqueSports:
        for genero in listauniqueSex:     
            print("DF ganadores")
            deporteDF = UnionDFs.filter((UnionDFs.sport  == deporte))            
            GanadoresDF = deporteDF.filter((deporteDF.TieneMedalla  == 1) & \
                                            (deporteDF.sex  == genero) ) 
                                    
            qtyGanadores = GanadoresDF.count()
            print("qtyGanadores",qtyGanadores)
            GanadoresDF.show(truncate=False, n=500)         
            NoGanadoresFullDF = deporteDF.filter((deporteDF.TieneMedalla  == 0) & \
                                                (deporteDF.sex  == genero) ) 
            shuffleDF = NoGanadoresFullDF.sample(fraction=1.0)#Aleatorizar Dataframe de No Ganadores

            print("DF NO ganadores")
            NoGanadoresDF = shuffleDF.limit(qtyGanadores)             
            sampleBySportDF = GanadoresDF.union(NoGanadoresDF)
            sampledf = sampledf.union(sampleBySportDF)
            print("********break******************")
    print("Tamano Dataframe sampledf",(sampledf.count(), len(sampledf.columns)))
    sampledf.show()
    return sampledf


#############################################
############Ojo estoy leyendo desde DB#######
#############################################
def OneHotEncoder():
    sample_df = leer_desde_DB("MuestraEstrat")
   
    stringIndexer = StringIndexer(inputCol="sport", outputCol="sportIndex")
    model = stringIndexer.fit(sample_df)
    indexed = model.transform(sample_df)
    indexed.show(n=500)


    encoder = OneHotEncoder()
    encoder.setInputCols(["sportIndex"])
    encoder.setOutputCols(["sportVec"])

    model = encoder.fit(indexed)
    model.setOutputCols(["sportVec"])
    model.getHandleInvalid()
    model.show()
    model.transform(indexed).head().output



    print("type single_col_model:", type(model))
    model.show()


    return True


def escribir_en_DB(DF,nombreDF):
    DF \
        .write \
        .format("jdbc") \
        .mode('overwrite') \
        .option("url", "jdbc:postgresql://host.docker.internal:5433/postgres") \
        .option("user", "postgres") \
        .option("password", "testPassword") \
        .option("dbtable", nombreDF) \
        .save()
    return True

def leer_desde_DB(nombreDF):
    df = spark \
        .read \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://host.docker.internal:5433/postgres") \
        .option("user", "postgres") \
        .option("password", "testPassword") \
        .option("dbtable", nombreDF) \
        .load()
    return df

def main():
    """
    csvPath1 = sys.argv[1]#Indices de desarrollo
    csvPath2 = sys.argv[2]#Informacion de atletas
    ###Cargar datos desde .csv
    IndicesDesarrllo_por_paisDF = cargarDataset1(csvPath1)#Cargar RawDataset1
    AtletasDF = cargarDataset2(csvPath2)#Cargar RawDataset2
    ####Seleccionar features deseados para el modelo predictivo
    mainColumnsIndicesDF = transformDatasetIndicesGlobales(IndicesDesarrllo_por_paisDF)#Seleccionar features deseados para el modelo predictivo
    mainColumnsAtletasDF = transformDatasetAtletas(AtletasDF)#Seleccionar features deseados para el modelo predictivo
    #Imputacion de valores faltantes
    IndicesPreprocesadosDF = imputacionIndicesGlobales(mainColumnsIndicesDF)
    AtletasPreprocesadosDF = imputacionAtletas(mainColumnsAtletasDF)

    #Union/cruzar datasets
    UnionDFs = joinDataframes(IndicesPreprocesadosDF,AtletasPreprocesadosDF)

    muestraEstratificadaDF = MuestraEstratificado(UnionDFs)   
    """

    OneHotEncoder()




    #Escritura a base de datos
    #escribir_en_DB(IndicesPreprocesadosDF ,"IndicesGlobales")#Escribir IndicesGlobales a base de datos
    #escribir_en_DB(AtletasPreprocesadosDF , "InfoAtletasOlimp")#Escribir InfoAtletasOlimp a base de datos
    #escribir_en_DB(muestraEstratificadaDF , "MuestraEstrat")#Escribir muestraEstratificadaDF a base de datos


    return True

if __name__ == '__main__':
    main()
