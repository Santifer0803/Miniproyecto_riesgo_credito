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
    "macro_historico",
    "tipo_cambio",
    "escenarios_macro",
    "depositos_btb",
    "deudores",
    "operaciones_crediticias",
    "operaciones_no_reportadas",
    "cuotas_atrasadas",
    "modificaciones",
    "vinculos_refinanciacion",
    "medidas_covid",
    "garantias_operaciones",
    "garantias_reales",
    "garantias_valores",
    "garantias_facturas_cedidas",
    "garantias_mobiliarias",
    "gravamenes",
    "bienes_realizables",
    "operaciones_bienes_realizables",
    "ventas_bienes",
    "flujos_recuperacion",
    "bienes_no_reportados",
    "catalogo_salidas",
    "financieros_entidad",
    "financieros_deudores",
]

# Diccionario para guardar las bases de datos
dfs = {}

# Se carga cada base de datos, teniendo en cuenta el formato de la fecha
for n in nombres:
    ruta = f"data/{n}.csv"
    cols = pd.read_csv(ruta, nrows=0).columns
    fechas = [c for c in cols if "fecha" in c.lower()]
    dfs[n] = pd.read_csv(ruta, parse_dates=fechas)

# Se eliminan las variables innecesarias
del cols, fechas, n, nombres, ruta

# Se unen los dataframes de bienes realizables y bienes no reportados
dfs["bienes_realizables"] = dfs["bienes_realizables"].merge(
    dfs["bienes_no_reportados"], on="bien_id", how="left"
)

# Se unen los dataframes de operaciones bienes realizables y bienes realizables
dfs["operaciones_bienes_realizables"] = dfs[
    "operaciones_bienes_realizables"
].merge(dfs["bienes_realizables"], on="bien_id", how="left")

# Se unen los dataframes de operaciones bienes realizables y ventas bienes
dfs["operaciones_bienes_realizables"] = dfs[
    "operaciones_bienes_realizables"
].merge(dfs["ventas_bienes"], on="bien_id", how="left")

# Se pone la moneda a las operaciones de bienes realizables
dfs["operaciones_bienes_realizables"] = dfs[
    "operaciones_bienes_realizables"
].merge(
    dfs["operaciones_crediticias"][["operacion_id", "moneda"]],
    on="operacion_id",
    how="left",
)

# Se unen los dataframes de operaciones crediticias y deudores
dfs["operaciones_crediticias"] = dfs["operaciones_crediticias"].merge(
    dfs["deudores"][
        [
            "deudor_id",
            "tipo_persona",
            "generador_divisas",
            "cobertura_fx_prop",
            "fecha_registro",
        ]
    ],
    on="deudor_id",
    how="left",
)

# Se unen los dataframes de operaciones crediticias y depositos back to back
dfs["operaciones_crediticias"] = dfs["operaciones_crediticias"].merge(
    dfs["depositos_btb"][
        ["operacion_id", "fecha_corte", "saldo_bloqueado_mo"]
    ].rename(
        columns={
            "fecha_corte": "fecha_corte_btb",
            "saldo_bloqueado_mo": "saldo_bloqueado_mo_btb",
        }
    ),
    on="operacion_id",
    how="left",
)

# Se unen los dataframes de operaciones crediticias y operaciones no reportadas
dfs["operaciones_crediticias"] = (
    dfs["operaciones_crediticias"]
    .merge(
        dfs["operaciones_no_reportadas"][
            [
                "operacion_id",
                "fecha_ultimo_reporte",
                "motivo_local",
                "mora_ultima",
                "judicial_ultimo",
                "capital_final_evento_mo",
                "fraccion_saldo_atendido_prop",
            ]
        ].rename(
            columns={
                "motivo_local": "motivo_local_nr",
                "mora_ultima": "mora_ultima_nr",
                "judicial_ultimo": "judicial_ultimo_nr",
            }
        ),
        left_on=["operacion_id", "fecha"],
        right_on=["operacion_id", "fecha_ultimo_reporte"],
        how="left",
    )
    .drop(columns=["fecha_ultimo_reporte"])
)

# Se unen los dataframes de operaciones crediticias y modificaciones
dfs["operaciones_crediticias"] = dfs["operaciones_crediticias"].merge(
    dfs["modificaciones"][
        [
            "modificacion_id",
            "operacion_id",
            "fecha",
            "tipo_local",
            "dificultad_financiera",
            "mora_previa_mes",
            "mora_posterior",
            "saldo_previo_mes_mo",
            "saldo_posterior_mo",
            "motivo_covid",
        ]
    ].rename(
        columns={
            "tipo_local": "motivo_mod",
            "dificultad_financiera": "dificultad_financiera_mod",
            "mora_previa_mes": "mora_previa_mod",
            "mora_posterior": "mora_posterior_mod",
            "saldo_previo_mes_mo": "saldo_previo_mes_mod",
            "saldo_posterior_mo": "saldo_posterior_mod",
            "motivo_covid": "motivo_covid_mod",
        }
    ),
    on=["operacion_id", "fecha"],
    how="left",
)

# Se unen todos los dataframes de garantías
dfs["garantias_reales"] = pd.concat(
    [
        dfs["garantias_reales"],
        dfs["garantias_valores"],
        dfs["garantias_facturas_cedidas"],
        dfs["garantias_mobiliarias"],
    ],
    ignore_index=True,
)

# Se une la información de las garantías con las operaciones de garantías
dfs["garantias_operaciones"] = dfs["garantias_operaciones"].merge(
    dfs["garantias_reales"][
        ["garantia_id", "entidad", "tipo", "fecha_avaluo"]
    ].rename(columns={"tipo": "tipo_gar", "fecha_avaluo": "fecha_avaluo_gar"}),
    on="garantia_id",
    how="left",
)

# Se une la información de las garantías con las operaciones de garantías
dfs["garantias_operaciones"] = dfs["garantias_operaciones"].merge(
    dfs["gravamenes"][
        ["garantia_id", "grado", "acreedor_entidad", "monto_preferente_mo"]
    ].rename(
        columns={
            "grado": "grado_prelacion_grav",
            "acreedor_entidad": "acreedor_entidad_grav",
        }
    ),
    on="garantia_id",
    how="left",
)

# Se mapean las condiciones de cada categoría BUST
condiciones = [
    (dfs["operaciones_crediticias"]["back_to_back"] == 0)
    & (dfs["operaciones_crediticias"]["tipo_persona"] == "Fisica")
    & (dfs["operaciones_crediticias"]["tipo_cartera_origen"] == "Vivienda")
    & (dfs["operaciones_crediticias"]["moneda"] == "MN"),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0)
    & (dfs["operaciones_crediticias"]["tipo_persona"] == "Fisica")
    & (dfs["operaciones_crediticias"]["tipo_cartera_origen"] == "Vivienda")
    & (dfs["operaciones_crediticias"]["moneda"] == "ME"),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0)
    & (dfs["operaciones_crediticias"]["tipo_persona"] == "Fisica")
    & (dfs["operaciones_crediticias"]["tipo_cartera_origen"] == "Consumo"),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0)
    & (dfs["operaciones_crediticias"]["tipo_persona"] == "Fisica")
    & (dfs["operaciones_crediticias"]["tipo_cartera_origen"] == "Tarjeta"),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0)
    & (dfs["operaciones_crediticias"]["tipo_persona"] == "Fisica")
    & (
        dfs["operaciones_crediticias"]["tipo_cartera_origen"]
        == "Capital_trabajo"
    ),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0)
    & (dfs["operaciones_crediticias"]["tipo_persona"] == "Juridica")
    & (dfs["operaciones_crediticias"]["moneda"] == "MN"),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0)
    & (dfs["operaciones_crediticias"]["tipo_persona"] == "Juridica")
    & (dfs["operaciones_crediticias"]["moneda"] == "ME")
    & (dfs["operaciones_crediticias"]["generador_divisas"] == 0)
    | (dfs["operaciones_crediticias"]["back_to_back"] == 0)
    & (dfs["operaciones_crediticias"]["tipo_persona"] == "Juridica")
    & (dfs["operaciones_crediticias"]["moneda"] == "ME")
    & (dfs["operaciones_crediticias"]["cobertura_fx_prop"] < 1),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0)
    & (dfs["operaciones_crediticias"]["tipo_persona"] == "Juridica")
    & (dfs["operaciones_crediticias"]["moneda"] == "ME")
    & (dfs["operaciones_crediticias"]["generador_divisas"] == 1)
    & (dfs["operaciones_crediticias"]["cobertura_fx_prop"] >= 1),
    (dfs["operaciones_crediticias"]["back_to_back"] == 0)
    & (dfs["operaciones_crediticias"]["tipo_persona"] == "Fisica")
    & (dfs["operaciones_crediticias"]["tipo_cartera_origen"] == "Vehiculo"),
]

# Etiquetas para las categorías BUST
etiquetas = [
    "Vivienda MN",
    "Vivienda ME",
    "Consumo",
    "Tarjetas de crédito",
    "Personal empresarial",
    "Empresarial MN",
    "Empresarial ME con exposición cambiaria",
    "Empresarial ME sin exposición cambiaria",
    "Vehiculos",
]

# Se crea una columna con el nombre de las cateegorías BUST
dfs["operaciones_crediticias"]["categorias_bust"] = np.select(
    condiciones, etiquetas, default="Otro"
)

# Se eliminan las variables innecesarias
del condiciones, etiquetas

# Base final de operaciones de bienes
df_operaciones_bienes_realizables = dfs["operaciones_bienes_realizables"]

# Base final de flujos de recuperación
df_flujos_recuperacion = dfs["flujos_recuperacion"]

# Base final del catálogo de salidas
df_catalogo_salidas = dfs["catalogo_salidas"]

# Base final de cuotas atrasadas de los deudores
df_cuotas_atrasadas = dfs["cuotas_atrasadas"]

# Base final con la información de los deudores
df_deudores = dfs["deudores"]

# Base final de variables macroeconómicas históricas
df_macro_historico = dfs["macro_historico"]

# Base final de variables macroeconómicas proyectadas (escenario base/adverso)
df_macro_proyectado = dfs["escenarios_macro"]

# Base final de vehículos para refinanciación
df_vinculos_refinanciacion = dfs["vinculos_refinanciacion"]

# Base final de datos financieros de la entidad
df_financieros_entidad = dfs["financieros_entidad"]

# Base final de datos financieros de los deudores
df_financieros_deudores = dfs["financieros_deudores"]

# Base final de operaciones crediticas de las entidades
df_operaciones_crediticias = dfs["operaciones_crediticias"]

# Base final de las operaciones de garantías
df_garantias_operaciones = dfs["garantias_operaciones"]

# Base final con el tipo de cambio mensual
df_tipo_cambio = dfs["tipo_cambio"]

# Se eliminan las variables innecesarias
del dfs
