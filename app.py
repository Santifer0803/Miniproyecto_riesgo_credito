# -*- coding: utf-8 -*-
"""
Dashboard de riesgo de crédito

Aquí se juntan todas las funciones que se ejecutan para la que la aplicación
se ejecute correctamente, junto con los gráficos y demás elementos
"""

# Se agregan las librerias correspondientes
import pandas as pd
import numpy as np

# Se guardan los nombres de las bases de datos necesarias
nombres = [
    "macro_historico", "tipo_cambio", "escenarios_macro", "depositos_btb",
    "deudores", "operaciones_crediticias", "operaciones_no_reportadas",
    "cuotas_atrasadas", "modificaciones", "vinculos_refinanciacion",
    "medidas_covid", "garantias_operaciones", "garantias_reales",
    "garantias_valores", "garantias_facturas_cedidas", "garantias_mobiliarias",
    "gravamenes", "bienes_realizables", "operaciones_bienes_realizables",
    "ventas_bienes", "flujos_recuperacion", "bienes_no_reportados",
    "catalogo_salidas", "financieros_entidad", "financieros_deudores",
]

# Diccionario para guardar las bases de datos
dfs = {}

# Se carga cada base de datos, teniendo en cuenta el formato de la fecha
for n in nombres:
    ruta = f"data/{n}.csv"
    cols = pd.read_csv(ruta, nrows=0).columns
    fechas = [c for c in cols if "fecha" in c.lower()]
    dfs[n] = pd.read_csv(ruta, parse_dates=fechas)
    
# Se unen los dataframes de operaciones crediticias y deudores 
dfs["operaciones_crediticias"] = dfs["operaciones_crediticias"].merge(
    dfs["deudores"][["deudor_id", "tipo_persona", "generador_divisas", "cobertura_fx_prop", "fecha_registro"]],
    on = "deudor_id",
    how = "left"
)
    
# Se mapean las condiciones de cada categoría BUST
condiciones = [
    (dfs["operaciones_crediticias"]["back_to_back"] == 0) & (dfs["operaciones_crediticias"]["tipo_persona"] == "Fisica") & (dfs["operaciones_crediticias"]["tipo_cartera_origen"] == "Vivienda") & (dfs["operaciones_crediticias"]["moneda"] == "MN"),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0) & (dfs["operaciones_crediticias"]["tipo_persona"] == "Fisica") & (dfs["operaciones_crediticias"]["tipo_cartera_origen"] == "Vivienda") & (dfs["operaciones_crediticias"]["moneda"] == "ME"),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0) & (dfs["operaciones_crediticias"]["tipo_persona"] == "Fisica") & (dfs["operaciones_crediticias"]["tipo_cartera_origen"] == "Consumo"),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0) & (dfs["operaciones_crediticias"]["tipo_persona"] == "Fisica") & (dfs["operaciones_crediticias"]["tipo_cartera_origen"] == "Tarjeta"),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0) & (dfs["operaciones_crediticias"]["tipo_persona"] == "Fisica") & (dfs["operaciones_crediticias"]["tipo_cartera_origen"] == "Capital_trabajo"),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0) & (dfs["operaciones_crediticias"]["tipo_persona"] == "Juridica") & (dfs["operaciones_crediticias"]["moneda"] == "MN"),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0) & (dfs["operaciones_crediticias"]["tipo_persona"] == "Juridica") & (dfs["operaciones_crediticias"]["moneda"] == "ME") & (dfs["operaciones_crediticias"]["generador_divisas"] == 0),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0) & (dfs["operaciones_crediticias"]["tipo_persona"] == "Juridica") & (dfs["operaciones_crediticias"]["moneda"] == "ME") & (dfs["operaciones_crediticias"]["generador_divisas"] == 1),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0) & (dfs["operaciones_crediticias"]["tipo_persona"] == "Fisica") & (dfs["operaciones_crediticias"]["tipo_cartera_origen"] == "Vehiculo")
]

# Etiquetas para las categorías BUST
etiquetas = ["Vivienda MN",
"Vivienda ME",
"Consumo",
"Tarjetas de crédito",
"Personal empresarial", 
"Empresarial MN", 
"Empresarial ME con exposición cambiaria", 
"Empresarial ME sin exposición cambiaria", 
"Vehiculos"
]

# Se crea una columna con el nombre de las cateegorías BUST
dfs["operaciones_crediticias"]["categorias_bust"] = np.select(condiciones, etiquetas, default = "Otro")