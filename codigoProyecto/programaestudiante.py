import sys
import os,glob,shutil
import json
import csv
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext  
from pyspark.sql import functions as FN
from pyspark.sql.functions import explode, expr
from pyspark.sql.types import (StructType, IntegerType, DoubleType, ArrayType)

spark = SparkSession.builder.appName("Supermercado").getOrCreate()

#https://medium.com/expedia-group-tech/working-with-json-in-apache-spark-1ecf553c2a8c 
#https://www.py4u.net/discuss/1209855

def readFiles(argumentsFromTerminal):
    RawDF = spark.read.option("multiline","true").json(argumentsFromTerminal)
    #RawDF.printSchema()
    #RawDF.show()

    explodedDF = (RawDF
    .select(FN.explode('compras'), 'numero_caja')
    .select(FN.explode('col'), 'numero_caja')
    .withColumn('nombre', FN.col('col.nombre') )
    .withColumn('cantidad', FN.col('col.cantidad') )    
    .withColumn('precio_unitario', FN.col('col.precio_unitario') )
    .drop('col')
    )
    return explodedDF

def fn_total_productos(inputDF):
    groupedDF = inputDF.groupBy("nombre").sum()    
    summaryDF = groupedDF.select(FN.col('nombre'),
                                 FN.col('sum(cantidad)').alias('CantidadTotal'))               
    summaryDF.repartition(1)\
        .write.format('csv').option('header',True)\
        .mode('overwrite').option('sep',',').save('/src/compras/total_productos')      
    
    path = '/src/compras/total_productos'
    os.chdir(path)
    result = glob.glob('*.{}'.format('csv'))
    os.rename(result[0],'total_productos.csv')
    shutil.copyfile('/src/compras/total_productos/total_productos.csv', '/src/compras/total_productos.csv')
    return summaryDF

def fn_total_por_caja(inputDF):    
    groupedDF = inputDF.groupBy("numero_caja").sum() 
    summaryDF = groupedDF.select(FN.col('numero_caja'),
                                 FN.col('sum(SubTotal)').alias('Total_Dinero_Vendido'))
    summaryDF.repartition(1)\
        .write.format('csv').option('header',True)\
        .mode('overwrite').option('sep',',').save('/src/compras/total_por_caja')       
    
    path = '/src/compras/total_por_caja'
    os.chdir(path)
    result = glob.glob('*.{}'.format('csv'))
    os.rename(result[0],'total_por_caja.csv')
    shutil.copyfile('/src/compras/total_por_caja/total_por_caja.csv', '/src/compras/total_por_caja.csv')
    return summaryDF

def fn_metricas(SubTotals_DF):    
    total_productos_DF = fn_total_productos(SubTotals_DF)
    total_por_caja_DF = fn_total_por_caja(SubTotals_DF)
    total_por_caja_DF.show(n = 9999) 
    print("Estoy En estadisticos")

    #caja_con_mas_ventas    
    RowCajaVentaMax=total_por_caja_DF.orderBy(FN.desc("Total_Dinero_Vendido")).take(1)
    IDCajaVentaMax = int(RowCajaVentaMax[0].numero_caja)
    TotalCajaVentaMax = RowCajaVentaMax[0].Total_Dinero_Vendido
    print("IDCajaVentaMax",IDCajaVentaMax)
    print("TotalCajaVentaMax",TotalCajaVentaMax)

    #caja_con_menos_ventas
    RowCajaVentaMin=total_por_caja_DF.orderBy(FN.asc("Total_Dinero_Vendido")).take(1)
    IDCajaVentaMin = RowCajaVentaMin[0].numero_caja
    TotalCajaVentaMin = RowCajaVentaMin[0].Total_Dinero_Vendido
    print("IDCajaVentaMin",IDCajaVentaMin)
    print("TotalCajaVentaMin",TotalCajaVentaMin)

    #Percentliles

    P25 = total_por_caja_DF.agg(expr("percentile(Total_Dinero_Vendido, array(0.25))").\
                                    alias("percentile")).take(1)[0].percentile[0] 
    P50 = total_por_caja_DF.agg(expr("percentile(Total_Dinero_Vendido, array(0.50))").\
                                    alias("percentile")).take(1)[0].percentile[0] 
    P75 = total_por_caja_DF.agg(expr("percentile(Total_Dinero_Vendido, array(0.75))").\
                                    alias("percentile")).take(1)[0].percentile[0] 
    print("P25",P25)
    print("P50",P50)
    print("P75",P75)

    #producto_mas_vendido_por_unidad
    total_productos_DF.show(n=9999)

    RowArticuloMayorQTY=total_productos_DF.orderBy(FN.desc("CantidadTotal")).take(1)
    IDArticuloMayorQTY = RowArticuloMayorQTY[0].nombre
    QtyArticuloMayorQTY = RowArticuloMayorQTY[0].CantidadTotal
    print("IDArticuloMayorQTY",IDArticuloMayorQTY)
    print("QtyArticuloMayorQTY",QtyArticuloMayorQTY)
    print("caja_con_mas_UnidadesVendidas")

    #producto_de_mayor_ingreso_Dinero
    groupedDF = SubTotals_DF.groupBy("nombre").sum()  
    groupedDF.show() 
    print("producto_de_mayor_ingreso") 
    summaryDF = groupedDF.select(FN.col('nombre'),
                                 FN.col('sum(SubTotal)').alias('IngresoTotalPorArticulo')) 
    summaryDF.show()
    RowArticuloMayorVenta=summaryDF.orderBy(FN.desc("IngresoTotalPorArticulo")).take(1)
    IDArticuloMayorVenta = RowArticuloMayorVenta[0].nombre
    QtyArticuloMayorVenta = RowArticuloMayorVenta[0].IngresoTotalPorArticulo
    print("IDArticuloMayorVenta",IDArticuloMayorVenta)
    print("QtyArticuloMayorVenta",QtyArticuloMayorVenta)
    print("producto_de_mayor_ingreso_Dinero")


    with open('/src/compras/metricas.csv', 'w') as myfile:
        wr = csv.writer(myfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        wr.writerow(["Metrica", "Valor"])
        wr.writerow(["caja_con_mas_ventas", IDCajaVentaMax])
        wr.writerow(["caja_con_menos_ventas", IDCajaVentaMin ])
        wr.writerow(["percentil_25_por_caja", P25])
        wr.writerow(["percentil_50_por_caja", P50])
        wr.writerow(["percentil_75_por_caja", P75])
        wr.writerow(["producto_mas_vendido_por_unidad", IDArticuloMayorQTY  ])
        wr.writerow(["producto_de_mayor_ingreso", IDArticuloMayorVenta])
    
    listaMetricas = [IDCajaVentaMax, IDCajaVentaMin, P25, P50, P75, IDArticuloMayorQTY, IDArticuloMayorVenta]

    return listaMetricas

   

def main():
    arguments = sys.argv[1:]
    print("arguments",arguments)
    InitialDF = readFiles(arguments)     
    SubtotalsDF = InitialDF.withColumn("SubTotal", \
                        InitialDF.cantidad * InitialDF.precio_unitario)
    SubtotalsDF.show(n = 9999) 
    print("SubtotalsDF")
    print("***********************")
    total_productosDF = fn_total_productos(SubtotalsDF)
    total_productosDF.show(n = 9999) 
    print("total_productos")
    print("***********************")
    total_por_cajaDF = fn_total_por_caja(SubtotalsDF)    
    total_por_cajaDF.show(n = 9999) 
    print("total_por_caja")
    print("***********************")
    metricasDF = fn_metricas(SubtotalsDF)
    
    return True
    


        
if __name__ == '__main__':
    main()


