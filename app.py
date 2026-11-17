# -*- coding: utf-8 -*-

# %%

# Se agregan las librerías correspondientes
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import importlib.util
import subprocess
import sys
from pathlib import Path

"""
Carga y limpieza inicial de los datos

Aquí se cargan todas las bases de datos y se tratan, relacionándolas o limpiándolas
"""

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
    dfs["operaciones_crediticias"][["operacion_id", "moneda"]].drop_duplicates(
        "operacion_id"
    ),
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

# Se une la información de las garantías con la de gravamenes
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

# Se crea una columna con el nombre de las categorías BUST
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

# Base final de operaciones crediticas de las entidades
df_operaciones_crediticias = dfs["operaciones_crediticias"]

# Base final de las operaciones de garantías
df_garantias_operaciones = dfs["garantias_operaciones"]

# Base final con el tipo de cambio mensual
df_tipo_cambio = dfs["tipo_cambio"]

# Se eliminan las variables innecesarias
del dfs

# %%

""""
Funciones a utilizar en el proyecto

Aquí se definen todas las funciones que se usarán para calcular lo solicitado
a lo largo del trabajo
"""

""" PD y PC históricas """

def preparar_inventario_crediticio(
    df_operaciones, fecha_corte, excluir_btb=True, excluir_otro=True
):
    """
    Prepara el inventario crediticio para una fecha de corte, aplicando
    filtros de exclusión y verificando que no existan operaciones duplicadas.

    Parámetros
    ----------
    df_operaciones : pd.DataFrame
        Base histórica de operaciones crediticias.
    fecha_corte : str
        Fecha de evaluación en formato 'YYYY-MM-DD'.
    excluir_btb : bool, opcional
        Si es True, excluye las operaciones back-to-back. Por defecto, True.
    excluir_otro : bool, opcional
        Si es True, excluye las operaciones de categoría 'Otro'. Por defecto, True.

    Retorna
    -------
    pd.DataFrame
        Inventario crediticio filtrado para la fecha de corte indicada.

    Raises
    ------
    ValueError
        Si existen identificadores de operación duplicados en el inventario.
    """

    fecha_corte = pd.Timestamp(fecha_corte)

    inventario = df_operaciones.loc[
        df_operaciones["fecha"] == fecha_corte
    ].copy()

    if excluir_btb:
        inventario = inventario.loc[inventario["back_to_back"] == 0]

    if excluir_otro:
        inventario = inventario.loc[inventario["categorias_bust"] != "Otro"]

    if inventario["operacion_id"].duplicated().any():
        raise ValueError("Existen operaciones duplicadas en el corte.")

    return inventario

def diagnosticar_salidas(
    df_operaciones,
    inventario_inicial,
    inventario_final,
    fecha_inicial,
    fecha_final,
):
    """
    Identifica las operaciones presentes en el inventario inicial que
    están ausentes en el inventario final y recupera su último registro
    disponible dentro del período analizado.

    Parámetros
    ----------
    df_operaciones : pd.DataFrame
        Base histórica de operaciones con sus fechas y motivos de salida.
    inventario_inicial : pd.DataFrame
        Inventario de operaciones al inicio del período.
    inventario_final : pd.DataFrame
        Inventario de operaciones al final del período.
    fecha_inicial : str o datetime
        Fecha inicial del período de análisis.
    fecha_final : str o datetime
        Fecha final del período de análisis.

    Retorna
    -------
    pd.DataFrame
        Último registro disponible de cada operación ausente, incluyendo
        el motivo de salida registrado. Si no existe un motivo, se asigna
        "SIN MOTIVO REGISTRADO".
    """

    fecha_inicial = pd.Timestamp(fecha_inicial)
    fecha_final = pd.Timestamp(fecha_final)

    ids_iniciales = set(inventario_inicial["operacion_id"])

    ids_finales = set(inventario_final["operacion_id"])

    ids_ausentes = ids_iniciales - ids_finales

    historico = df_operaciones.loc[
        df_operaciones["operacion_id"].isin(ids_ausentes)
        & df_operaciones["fecha"].between(fecha_inicial, fecha_final)
    ].copy()

    historico = historico.sort_values(["operacion_id", "fecha"])

    ultimos_registros = (
        historico.groupby("operacion_id", as_index=False).tail(1).copy()
    )

    ultimos_registros["motivo_diagnostico"] = ultimos_registros[
        "motivo_local_nr"
    ].fillna("SIN MOTIVO REGISTRADO")

    return ultimos_registros

def resumir_modificaciones_crediticias(
    df_operaciones, fecha_corte, meses_ventana=12
):
    """
    Resume las modificaciones crediticias realizadas por operación
    dentro de una ventana temporal anterior a la fecha de corte.

    Identifica modificaciones asociadas con dificultades financieras,
    mora previa superior a 60 días, COVID-19 y aumentos del saldo
    superiores al 2 %.

    Parámetros
    ----------
    df_operaciones : pd.DataFrame
        Base histórica de operaciones crediticias que contiene
        la información de las modificaciones realizadas.

    fecha_corte : str o pd.Timestamp
        Fecha máxima hasta la cual se consideran las modificaciones.

    meses_ventana : int, opcional
        Número de meses anteriores a la fecha de corte que se
        consideran en el análisis. Por defecto, 12 meses.

    Retorna
    -------
    resumen : pd.DataFrame
        DataFrame con una fila por operación modificada durante
        la ventana seleccionada. Contiene el número de modificaciones,
        indicadores de dificultad financiera, mora previa superior
        a 60 días, COVID-19, crecimiento del saldo superior al 2 %
        y fecha de la última modificación.
    """

    fecha_corte = pd.Timestamp(fecha_corte)

    fecha_inicio = fecha_corte - pd.DateOffset(months=meses_ventana)

    mods = df_operaciones.loc[
        (df_operaciones["modificada_mes"] == 1)
        & (df_operaciones["fecha"] > fecha_inicio)
        & (df_operaciones["fecha"] <= fecha_corte)
    ].copy()

    # Indicadores por evento de modificación
    mods["mod_dificultad"] = mods["dificultad_financiera_mod"] == 1

    mods["mod_mora_60"] = mods["mora_previa_mod"] > 60

    mods["mod_covid"] = mods["motivo_covid_mod"] == 1

    # Variación del saldo: evitar división por cero
    saldo_previo = mods["saldo_previo_mes_mod"]

    mods["variacion_saldo"] = np.where(
        saldo_previo > 0,
        mods["saldo_posterior_mod"] / saldo_previo - 1,
        np.nan,
    )

    mods["mod_crecimiento_2"] = mods["variacion_saldo"] > 0.02

    # Resumen por operación
    resumen = (
        mods.groupby("operacion_id")
        .agg(
            n_modificaciones=("modificacion_id", "nunique"),
            alguna_dificultad=("mod_dificultad", "any"),
            alguna_mora_60=("mod_mora_60", "any"),
            alguna_covid=("mod_covid", "any"),
            algun_crecimiento_2=("mod_crecimiento_2", "any"),
            ultima_modificacion=("fecha", "max"),
        )
        .reset_index()
    )

    return resumen

def construir_indicadores_crediticios(
    df_operaciones, fecha_corte, meses_modificaciones=12
):
    """
    Construye indicadores de deterioro crediticio por operación
    a una fecha de corte, considerando mora, cobro judicial y
    modificaciones recientes. No asigna el estado BUST definitivo.

    Parámetros
    ----------
    df_operaciones : pd.DataFrame
        Base histórica de operaciones crediticias.
    fecha_corte : str o pd.Timestamp
        Fecha de evaluación de los indicadores crediticios.
    meses_modificaciones : int, opcional
        Número de meses considerados para identificar modificaciones
        recientes. Por defecto, 12 meses.

    Retorna
    -------
    indicadores : pd.DataFrame
        Inventario de operaciones a la fecha de corte, enriquecido
        con información de modificaciones e indicadores de mora,
        cobro judicial, dificultad financiera, crecimiento del saldo
        y deterioro crediticio básico y ampliado.
    """

    fecha_corte = pd.Timestamp(fecha_corte)

    # 1. Inventario en la fecha de evaluación
    inventario = preparar_inventario_crediticio(
        df_operaciones=df_operaciones, fecha_corte=fecha_corte
    )

    # 2. Modificaciones recientes
    resumen_mods = resumir_modificaciones_crediticias(
        df_operaciones=df_operaciones,
        fecha_corte=fecha_corte,
        meses_ventana=meses_modificaciones,
    )

    # 3. Unir información
    indicadores = inventario.merge(
        resumen_mods, on="operacion_id", how="left", validate="one_to_one"
    )

    # 4. Indicadores básicos
    indicadores["ind_mora_90"] = indicadores["mora_dias"] > 90

    indicadores["ind_judicial"] = indicadores["cobro_judicial"] == 1

    indicadores["ind_mod_mora_60"] = (
        indicadores["alguna_mora_60"].fillna(False).astype(bool)
    )

    indicadores["ind_mod_dificultad"] = (
        indicadores["alguna_dificultad"].fillna(False).astype(bool)
    )

    indicadores["ind_mod_crecimiento_2"] = (
        indicadores["algun_crecimiento_2"].fillna(False).astype(bool)
    )

    # 5. Indicador básico de deterioro
    indicadores["deterioro_basico"] = (
        indicadores["ind_mora_90"] | indicadores["ind_judicial"]
    )

    # 6. Indicador ampliado preliminar
    indicadores["deterioro_ampliado"] = (
        indicadores["deterioro_basico"] | indicadores["ind_mod_mora_60"]
    )

    return indicadores


def clasificar_inventario_inicial_crediticio(
    df_operaciones, fecha_corte, meses_ventana=12
):
    """
    Clasifica las operaciones del inventario inicial en estado Normal
    o Dudoso para el cálculo de PD y PC, según sus condiciones de
    morosidad, cobro judicial, modificaciones y eventos de salida.

    Excluye operaciones back-to-back, cartera Otro y cancelaciones,
    utilizando únicamente información disponible hasta la fecha de corte.

    Parameters
    ----------
    df_operaciones : pd.DataFrame
        Base histórica de operaciones crediticias con información
        de saldos, morosidad, modificaciones y eventos de salida.

    fecha_corte : str o pd.Timestamp
        Fecha de referencia (T0) para construir y clasificar
        el inventario inicial.

    meses_ventana : int, default 12
        Número de meses anteriores a la fecha de corte utilizados
        para identificar modificaciones crediticias.

    Returns
    -------
    inventario : pd.DataFrame
        Inventario de operaciones vigentes en T0, con indicadores
        de deterioro, resumen de modificaciones y clasificación
        crediticia inicial (Normal o Dudoso).
    """

    fecha_corte = pd.Timestamp(fecha_corte)
    inicio = fecha_corte - pd.DateOffset(months=meses_ventana)

    datos = df_operaciones.loc[df_operaciones["fecha"] <= fecha_corte].copy()

    # Universo BUST
    datos = datos.loc[
        (datos["back_to_back"] != 1) & (datos["categorias_bust"] != "Otro")
    ].copy()

    # Inventario de operaciones reportadas exactamente en T0
    inventario = datos.loc[datos["fecha"] == fecha_corte].copy()

    # Verificar unicidad
    if inventario["operacion_id"].duplicated().any():
        raise ValueError(
            "Existen operaciones duplicadas en la fecha de corte."
        )

    # Excluir cancelaciones en T0
    inventario = inventario.loc[
        inventario["motivo_local_nr"].ne("CANCELACION")
    ].copy()

    # Eventos de modificación en los últimos 12 meses
    mods = datos.loc[
        (datos["modificada_mes"] == 1)
        & (datos["fecha"] > inicio)
        & (datos["fecha"] <= fecha_corte)
    ].copy()

    mods["mod_mora_60"] = mods["mora_previa_mod"] > 60

    resumen_mods = (
        mods.groupby("operacion_id")
        .agg(
            n_modificaciones=("modificacion_id", "nunique"),
            mod_mora_60=("mod_mora_60", "any"),
        )
        .reset_index()
    )

    inventario = inventario.merge(
        resumen_mods, on="operacion_id", how="left", validate="one_to_one"
    )

    inventario["n_modificaciones"] = (
        inventario["n_modificaciones"].fillna(0).astype(int)
    )

    inventario["mod_mora_60"] = (
        inventario["mod_mora_60"].fillna(False).astype(bool)
    )

    # Condiciones iniciales de dudoso
    inventario["ind_mora_90"] = inventario["mora_dias"] > 90

    inventario["ind_judicial"] = inventario["cobro_judicial"] == 1

    inventario["ind_adjudicado"] = inventario["motivo_local_nr"].eq(
        "ADJUDICACION"
    )

    inventario["ind_incobrable"] = inventario["motivo_local_nr"].eq("CASTIGO")

    inventario["dudoso_t0"] = (
        inventario["ind_mora_90"]
        | inventario["ind_judicial"]
        | inventario["mod_mora_60"]
        | inventario["ind_adjudicado"]
        | inventario["ind_incobrable"]
    )

    inventario["estado_t0"] = inventario["dudoso_t0"].map(
        {True: "Dudoso", False: "Normal"}
    )

    return inventario

def evaluar_operaciones_t12(
    df_operaciones, inventario_t0, fecha_t0, meses_horizonte=12
):
    """
    Evalúa el estado de las operaciones del inventario inicial al cierre
    del horizonte, identificando condiciones básicas de deterioro y
    eventos de salida. Conserva una fila por operación y no incorpora
    todavía el ajuste por modificaciones.

    Parámetros
    ----------
    df_operaciones : pd.DataFrame
        Base histórica con las observaciones mensuales de las operaciones.
    inventario_t0 : pd.DataFrame
        Inventario inicial con identificadores, entidad, categoría BUST
        y estado inicial de cada operación.
    fecha_t0 : str o pd.Timestamp
        Fecha de corte inicial de la evaluación.
    meses_horizonte : int, opcional
        Número de meses del horizonte de evaluación (por defecto, 12).

    Retorna
    -------
    pd.DataFrame
        Inventario inicial enriquecido con la última observación disponible,
        indicadores de seguimiento, permanencia al cierre, mora, cobro
        judicial, eventos de salida y clasificación básica de deterioro
        al horizonte evaluado.
    """

    fecha_t0 = pd.Timestamp(fecha_t0)

    fecha_t12 = fecha_t0 + pd.DateOffset(months=meses_horizonte)

    ids = inventario_t0["operacion_id"]

    # Observaciones dentro del horizonte
    seguimiento = df_operaciones.loc[
        df_operaciones["operacion_id"].isin(ids)
        & (df_operaciones["fecha"] > fecha_t0)
        & (df_operaciones["fecha"] <= fecha_t12)
    ].copy()

    # Última observación de cada operación
    ultimo = (
        seguimiento.sort_values("fecha")
        .groupby("operacion_id", as_index=False)
        .tail(1)
        .copy()
    )

    columnas = [
        "operacion_id",
        "fecha",
        "mora_dias",
        "cobro_judicial",
        "motivo_local_nr",
        "mora_ultima_nr",
        "judicial_ultimo_nr",
    ]

    ultimo = ultimo[columnas].rename(
        columns={
            "fecha": "fecha_ultima_t12",
            "mora_dias": "mora_t12",
            "cobro_judicial": "judicial_t12",
            "motivo_local_nr": "salida_t12",
            "mora_ultima_nr": "mora_salida_t12",
            "judicial_ultimo_nr": "judicial_salida_t12",
        }
    )

    resultado = inventario_t0[
        ["operacion_id", "entidad", "categorias_bust", "estado_t0"]
    ].merge(ultimo, on="operacion_id", how="left", validate="one_to_one")

    resultado["con_seguimiento"] = resultado["fecha_ultima_t12"].notna()

    resultado["reportada_t12"] = resultado["fecha_ultima_t12"] == fecha_t12

    resultado["ind_mora_90_t12"] = resultado["mora_t12"] > 90

    resultado["ind_judicial_t12"] = resultado["judicial_t12"] == 1

    resultado["ind_adjudicacion_t12"] = (
        resultado["salida_t12"] == "ADJUDICACION"
    )

    resultado["ind_castigo_t12"] = resultado["salida_t12"] == "CASTIGO"

    resultado["ind_cancelacion_t12"] = resultado["salida_t12"] == "CANCELACION"

    resultado["ind_refinanciacion_t12"] = (
        resultado["salida_t12"] == "REFINANCIACION"
    )

    resultado["dudoso_basico_t12"] = (
        resultado["ind_mora_90_t12"]
        | resultado["ind_judicial_t12"]
        | resultado["ind_adjudicacion_t12"]
        | resultado["ind_castigo_t12"]
        | (
            resultado["ind_cancelacion_t12"]
            & (
                (resultado["mora_salida_t12"] > 60)
                | (resultado["judicial_salida_t12"] == 1)
            )
        )
    )

    return resultado

def evaluar_crecimiento_modificaciones(
    df_operaciones, seguimiento_t12, fecha_t0
):
    
    """
    Evalúa las modificaciones crediticias realizadas durante los 12 meses
    anteriores y posteriores a T0, identificando su frecuencia, la mora
    previa y el crecimiento del saldo hasta T+12. Determina si dicho
    crecimiento supera el 2% y señala los casos no evaluables.
    
    Parámetros
    ----------
    df_operaciones : pd.DataFrame
        Base histórica de operaciones crediticias que contiene las fechas,
        modificaciones, mora previa y saldos correspondientes.
    
    seguimiento_t12 : pd.DataFrame
        Base de seguimiento de las operaciones hasta T+12, que incluye
        su identificador y el indicador de reporte en dicho corte.
    
    fecha_t0 : str o pd.Timestamp
        Fecha de corte inicial utilizada para definir la ventana de
        evaluación de 12 meses anteriores y posteriores.
    
    Retorna
    -------
    resultado : pd.DataFrame
        DataFrame que conserva la información de seguimiento y agrega
        el número y grupo de modificaciones, la mora previa a la primera
        modificación, el saldo previo a la última modificación, el saldo
        en T+12, el crecimiento porcentual y los indicadores de
        evaluabilidad, crecimiento superior al 2%, clasificación como
        dudoso por crecimiento y modificaciones pendientes de revisión.
    """

    fecha_t0 = pd.Timestamp(fecha_t0)
    inicio = fecha_t0 - pd.DateOffset(months=12)
    fin = fecha_t0 + pd.DateOffset(months=12)

    if seguimiento_t12["operacion_id"].duplicated().any():
        raise ValueError("Existen operaciones duplicadas en seguimiento_t12.")

    datos = df_operaciones.copy()
    datos["fecha"] = pd.to_datetime(datos["fecha"])

    ids = seguimiento_t12["operacion_id"]

    # --------------------------------------
    # 1. Modificaciones en ventana conjunta
    # --------------------------------------

    mods = datos.loc[
        datos["operacion_id"].isin(ids)
        & datos["fecha"].gt(inicio)
        & datos["fecha"].le(fin)
        & datos["modificada_mes"].eq(1)
    ].copy()

    mods = mods.sort_values(["operacion_id", "fecha", "modificacion_id"])

    # --------------------------------------
    # 2. Primera modificación
    # --------------------------------------

    primera = (
        mods.groupby("operacion_id", as_index=False)
        .head(1)[["operacion_id", "mora_previa_mod"]]
        .rename(columns={"mora_previa_mod": "mora_primera_mod"})
    )

    # --------------------------------------
    # 3. Última modificación
    # --------------------------------------

    ultima = (
        mods.groupby("operacion_id", as_index=False)
        .tail(1)[["operacion_id", "fecha", "saldo_previo_mes_mod"]]
        .rename(
            columns={
                "fecha": "fecha_ultima_mod",
                "saldo_previo_mes_mod": "saldo_previo_ultima_mod",
            }
        )
    )

    # --------------------------------------
    # 4. Número de modificaciones
    # --------------------------------------

    conteo = (
        mods.groupby("operacion_id")["modificacion_id"]
        .nunique()
        .rename("n_modificaciones_ventana")
        .reset_index()
    )

    # --------------------------------------
    # 5. Saldo observado exactamente en T+12
    # --------------------------------------

    saldos_t12 = datos.loc[
        datos["fecha"].eq(fin) & datos["operacion_id"].isin(ids),
        ["operacion_id", "saldo_principal_mo"],
    ].rename(columns={"saldo_principal_mo": "saldo_t12"})

    # --------------------------------------
    # 6. Consolidación
    # --------------------------------------

    resultado = (
        seguimiento_t12.merge(
            conteo, on="operacion_id", how="left", validate="one_to_one"
        )
        .merge(primera, on="operacion_id", how="left", validate="one_to_one")
        .merge(ultima, on="operacion_id", how="left", validate="one_to_one")
        .merge(
            saldos_t12, on="operacion_id", how="left", validate="one_to_one"
        )
    )

    resultado["n_modificaciones_ventana"] = (
        resultado["n_modificaciones_ventana"].fillna(0).astype(int)
    )

    for columna in [
        "mora_primera_mod",
        "saldo_previo_ultima_mod",
        "saldo_t12",
    ]:
        resultado[columna] = pd.to_numeric(resultado[columna], errors="coerce")

    # --------------------------------------
    # 7. Condiciones metodológicas
    # --------------------------------------

    n_mod = resultado["n_modificaciones_ventana"]

    resultado["tiene_modificaciones"] = n_mod.gt(0)

    resultado["grupo_modificaciones"] = np.select(
        [n_mod.eq(0), n_mod.between(1, 3), n_mod.gt(3)],
        [
            "Sin modificaciones",
            "1 a 3 modificaciones",
            "Más de 3 modificaciones",
        ],
        default="Sin clasificar",
    )

    resultado["mora_primera_menor_60"] = resultado["mora_primera_mod"].lt(60)

    resultado["mora_primera_mayor_60"] = resultado["mora_primera_mod"].gt(60)

    # El valor exactamente igual a 60 queda
    # identificado para revisión.
    resultado["mora_primera_igual_60"] = resultado["mora_primera_mod"].eq(60)

    # --------------------------------------
    # 8. Evaluabilidad del crecimiento
    # --------------------------------------

    resultado["crecimiento_evaluable"] = (
        resultado["tiene_modificaciones"]
        & resultado["mora_primera_mod"].notna()
        & resultado["saldo_previo_ultima_mod"].gt(0)
        & resultado["saldo_t12"].notna()
        & resultado["reportada_t12"].eq(True)
    )

    resultado["crecimiento_modificaciones"] = np.nan

    mascara = resultado["crecimiento_evaluable"]

    resultado.loc[mascara, "crecimiento_modificaciones"] = (
        resultado.loc[mascara, "saldo_t12"]
        / resultado.loc[mascara, "saldo_previo_ultima_mod"]
        - 1
    )

    resultado["crecimiento_superior_2"] = resultado[
        "crecimiento_modificaciones"
    ].gt(0.02)

    # --------------------------------------
    # 9. Regla adicional de clasificación
    # --------------------------------------

    # Se conserva la distinción de grupos
    # para la clasificación posterior.
    #
    # No se considera dudosa una operación
    # únicamente por tener modificaciones.

    resultado["dudoso_por_crecimiento"] = (
        resultado["crecimiento_evaluable"]
        & resultado["crecimiento_superior_2"]
    )

    # Identificar operaciones que requieren
    # revisión por falta de información.
    resultado["modificacion_pendiente"] = (
        resultado["tiene_modificaciones"] & ~resultado["crecimiento_evaluable"]
    )

    return resultado

def clasificar_operaciones_t12(evaluacion_mods, fecha_t12, revision_ref=None):
    """
    Clasifica las operaciones crediticias en T+12 como Normal, Dudoso,
    Pendiente o Cancelado, considerando el deterioro básico, las
    modificaciones crediticias y los eventos de salida. Permite
    reclasificar provisionalmente refinanciaciones verificadas.

    Parámetros
    ----------
    evaluacion_mods : pd.DataFrame
        Base de operaciones con indicadores de deterioro, modificaciones,
        seguimiento y eventos de salida evaluados en T+12.

    fecha_t12 : str o pd.Timestamp
        Fecha de cierre del período de evaluación T+12.

    revision_ref : pd.DataFrame, opcional
        Base de verificación de operaciones refinanciadas y sus sucesoras.
        Permite reclasificar como normales aquellas refinanciaciones
        pendientes cuya sucesora cumple los criterios establecidos.
        Por defecto es None.

    Retorna
    -------
    resultado : pd.DataFrame
        Base de operaciones con su clasificación consolidada en T+12,
        el motivo de clasificación y los indicadores de trazabilidad.
        Incluye las columnas estado_t12, motivo_t12,
        dudoso_por_modificaciones, modificacion_no_evaluable y
        cancelacion_no_dudosa.
    """

    fecha_t12 = pd.Timestamp(fecha_t12)

    resultado = evaluacion_mods.copy()

    # -------------------------------------
    # 1. Indicadores básicos
    # -------------------------------------

    basico = resultado["dudoso_basico_t12"].fillna(False).astype(bool)

    tiene_modificaciones = (
        resultado["tiene_modificaciones"].fillna(False).astype(bool)
    )

    evaluable = resultado["crecimiento_evaluable"].fillna(False).astype(bool)

    crecimiento_2 = (
        resultado["crecimiento_superior_2"].fillna(False).astype(bool)
    )

    mora_mayor_60 = (
        resultado["mora_primera_mayor_60"].fillna(False).astype(bool)
    )

    mora_menor_60 = (resultado["mora_primera_mod"] <= 60).fillna(False)

    # -------------------------------------
    # 2. Modificaciones
    # -------------------------------------

    n_mod = resultado["n_modificaciones_ventana"]

    grupo_1_3 = n_mod.between(1, 3)
    grupo_mas_3 = n_mod.gt(3)

    # Crecimiento relevante para el
    # procedimiento adicional.
    #
    # Se conservan ambos grupos para
    # facilitar la trazabilidad.
    deterioro_mod_1_3 = (
        grupo_1_3 & evaluable & crecimiento_2 & (mora_menor_60 | mora_mayor_60)
    )

    deterioro_mod_mas_3 = (
        grupo_mas_3
        & evaluable
        & crecimiento_2
        & (mora_menor_60 | mora_mayor_60)
    )

    deterioro_modificaciones = deterioro_mod_1_3 | deterioro_mod_mas_3

    # -------------------------------------
    # 3. Eventos de salida
    # -------------------------------------

    cancelacion = resultado["ind_cancelacion_t12"].fillna(False).astype(bool)

    adjudicacion = resultado["ind_adjudicacion_t12"].fillna(False).astype(bool)

    castigo = resultado["ind_castigo_t12"].fillna(False).astype(bool)

    refinanciacion = (
        resultado["ind_refinanciacion_t12"].fillna(False).astype(bool)
    )

    sin_seguimiento = ~resultado["con_seguimiento"].fillna(False).astype(bool)

    # Cancelaciones sin deterioro básico
    cancelacion_evaluable = (
        resultado["mora_salida_t12"].notna()
        & resultado["judicial_salida_t12"].notna()
    )

    cancelacion_no_dudosa = cancelacion & ~basico & cancelacion_evaluable

    cancelacion_sin_datos = cancelacion & ~cancelacion_evaluable

    # -------------------------------------
    # 4. Casos pendientes
    # -------------------------------------

    modificacion_no_evaluable = (
        tiene_modificaciones
        & ~evaluable
        & ~basico
        & ~cancelacion
        & ~adjudicacion
        & ~castigo
    )

    pendiente = (
        sin_seguimiento
        | refinanciacion
        | modificacion_no_evaluable
        | cancelacion_sin_datos
    )

    # -------------------------------------
    # 5. Clasificación consolidada
    # -------------------------------------

    resultado["estado_t12"] = "Normal"
    resultado["motivo_t12"] = "Sin deterioro identificado"

    # Cancelaciones no dudosas
    resultado.loc[cancelacion_no_dudosa, "estado_t12"] = "Cancelado"

    resultado.loc[cancelacion_no_dudosa, "motivo_t12"] = (
        "Cancelación sin deterioro básico"
    )

    # Pendientes
    resultado.loc[pendiente, "estado_t12"] = "Pendiente"

    resultado.loc[pendiente, "motivo_t12"] = "Requiere revisión"

    # Dudosas por modificaciones
    resultado.loc[deterioro_modificaciones, "estado_t12"] = "Dudoso"

    resultado.loc[deterioro_modificaciones, "motivo_t12"] = (
        "Deterioro por modificaciones"
    )

    # Dudosas por criterio básico
    resultado.loc[basico, "estado_t12"] = "Dudoso"

    resultado.loc[basico, "motivo_t12"] = "Deterioro básico o salida dudosa"

    # -------------------------------------
    # 6. Indicadores de trazabilidad
    # -------------------------------------

    resultado["dudoso_por_modificaciones"] = deterioro_modificaciones

    resultado["modificacion_no_evaluable"] = modificacion_no_evaluable

    resultado["cancelacion_no_dudosa"] = cancelacion_no_dudosa

    # Validación de unicidad
    if resultado["operacion_id"].duplicated().any():
        raise ValueError("Existen operaciones duplicadas.")

    # -------------------------------------
    # 7. Refinanciaciones verificadas
    # -------------------------------------

    if revision_ref is not None:

        ids_verificados = revision_ref.loc[
            revision_ref["fecha_ultima_sucesora"].eq(fecha_t12)
            & revision_ref["mora_sucesora"].eq(0)
            & revision_ref["judicial_sucesora"].eq(0)
            & revision_ref["modificacion_dudosa"].eq(False),
            "operacion_id",
        ]

        mascara = (
            resultado["operacion_id"].isin(ids_verificados)
            & resultado["estado_t12"].eq("Pendiente")
            & resultado["salida_t12"].eq("REFINANCIACION")
        )

        resultado.loc[mascara, "estado_t12"] = "Normal"

        resultado.loc[mascara, "motivo_t12"] = (
            "Refinanciación con sucesora normal " "al cierre (provisional)"
        )

    return resultado

def calcular_pd_pc_historicas(
    clasificacion, agrupar_por=("entidad", "categorias_bust")
):
    """
    Calcula la Probabilidad de Incumplimiento (PD) y la Probabilidad
    de Cura (PC) históricas a partir de las transiciones de estado
    de las operaciones durante un horizonte de 12 meses.

    El cálculo considera los deterioros, las recuperaciones y los
    eventos de salida. Las operaciones pendientes permanecen en
    los denominadores y se reportan por separado.

    Parámetros
    ----------
    clasificacion : pd.DataFrame
        DataFrame con la clasificación inicial (T0) y final (T+12)
        de las operaciones, así como sus eventos de salida y las
        variables necesarias para determinar incumplimientos y curas.

    agrupar_por : tuple, opcional
        Columnas utilizadas para agrupar los resultados.
        Por defecto, ("entidad", "categorias_bust").

    Retorna
    -------
    resumen : pd.DataFrame
        DataFrame con los componentes agregados de PD y PC, el número
        de operaciones pendientes y las tasas históricas calculadas
        en valores decimales (PD, PC) y porcentuales (PD_pct, PC_pct).

    datos : pd.DataFrame
        Copia del DataFrame original con los indicadores individuales
        utilizados para calcular los componentes de PD y PC, así
        como las operaciones pendientes.
    """

    datos = clasificacion.copy()
    grupos = list(agrupar_por)

    # -----------------------------------------
    # 1. Validaciones iniciales
    # -----------------------------------------

    columnas_necesarias = [
        "operacion_id",
        "estado_t0",
        "estado_t12",
        "salida_t12",
        "mora_salida_t12",
        "judicial_salida_t12",
        *grupos,
    ]

    faltantes = [c for c in columnas_necesarias if c not in datos.columns]

    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")

    if datos["operacion_id"].duplicated().any():
        raise ValueError("Existen operaciones duplicadas.")

    estados_validos = {"Normal", "Dudoso", "Pendiente", "Cancelado"}

    estados_invalidos = (
        set(datos["estado_t12"].dropna().unique()) - estados_validos
    )

    if estados_invalidos:
        raise ValueError(
            f"Estados T12 no reconocidos: " f"{estados_invalidos}"
        )

    # -----------------------------------------
    # 2. Cohortes iniciales
    # -----------------------------------------

    normal_t0 = datos["estado_t0"].eq("Normal")
    dudoso_t0 = datos["estado_t0"].eq("Dudoso")

    normal_t12 = datos["estado_t12"].eq("Normal")
    dudoso_t12 = datos["estado_t12"].eq("Dudoso")
    pendiente_t12 = datos["estado_t12"].eq("Pendiente")

    # -----------------------------------------
    # 3. Eventos de salida
    # -----------------------------------------

    salida = datos["salida_t12"]

    adjudicacion = salida.eq("ADJUDICACION")
    castigo = salida.eq("CASTIGO")
    cancelacion = salida.eq("CANCELACION")
    refinanciacion = salida.eq("REFINANCIACION")

    mora_salida = pd.to_numeric(datos["mora_salida_t12"], errors="coerce")

    judicial_salida = pd.to_numeric(
        datos["judicial_salida_t12"], errors="coerce"
    )

    # Información suficiente para evaluar
    # una cancelación.
    cancelacion_evaluable = (
        cancelacion & mora_salida.notna() & judicial_salida.notna()
    )

    # Cancelación sin deterioro identificado
    # al momento de la salida.
    cancelacion_elegible = (
        cancelacion_evaluable & mora_salida.le(60) & judicial_salida.eq(0)
    )

    cancelacion_dudosa = cancelacion_evaluable & (
        mora_salida.gt(60) | judicial_salida.eq(1)
    )

    # -----------------------------------------
    # 4. Componentes de PD
    # -----------------------------------------

    datos["pd_A"] = normal_t0.astype(int)

    # Adjudicaciones
    datos["pd_C"] = (normal_t0 & adjudicacion).astype(int)

    # Castigos
    datos["pd_D"] = (normal_t0 & castigo).astype(int)

    # Deterioro distinto de adjudicación
    # o castigo.
    datos["pd_B"] = (
        normal_t0
        & (dudoso_t12 | cancelacion_dudosa)
        & ~adjudicacion
        & ~castigo
    ).astype(int)

    # Pendientes de PD:
    # Solo operaciones inicialmente normales
    # que continúan pendientes en T+12.

    datos["pd_pendientes"] = (
        normal_t0
        & pendiente_t12
        & datos["pd_B"].eq(0)
        & datos["pd_C"].eq(0)
        & datos["pd_D"].eq(0)
    ).astype(int)

    # -----------------------------------------
    # 5. Componentes de PC
    # -----------------------------------------

    datos["pc_A"] = dudoso_t0.astype(int)

    # Cancelaciones elegibles
    datos["pc_E"] = (dudoso_t0 & cancelacion_elegible).astype(int)

    # Recuperaciones con operación activa
    datos["pc_B"] = (
        dudoso_t0
        & normal_t12
        & ~cancelacion
        & ~adjudicacion
        & ~castigo
        & ~refinanciacion
    ).astype(int)

    # Pendientes de PC:
    # Solo operaciones inicialmente dudosas
    # que continúan pendientes en T+12.

    datos["pc_pendientes"] = (
        dudoso_t0 & pendiente_t12 & datos["pc_B"].eq(0) & datos["pc_E"].eq(0)
    ).astype(int)

    # -----------------------------------------
    # 6. Validaciones de numeradores
    # -----------------------------------------

    if datos[["pd_B", "pd_C", "pd_D"]].sum(axis=1).gt(1).any():
        raise ValueError("Duplicidad en numerador de PD.")

    if datos[["pc_B", "pc_E"]].sum(axis=1).gt(1).any():
        raise ValueError("Duplicidad en numerador de PC.")

    # Los pendientes no deben contarse
    # simultáneamente como eventos.
    if (
        datos[["pd_B", "pd_C", "pd_D"]].sum(axis=1).gt(0)
        & datos["pd_pendientes"].eq(1)
    ).any():
        raise ValueError("PD: evento y pendiente simultáneos.")

    if (
        datos[["pc_B", "pc_E"]].sum(axis=1).gt(0)
        & datos["pc_pendientes"].eq(1)
    ).any():
        raise ValueError("PC: cura y pendiente simultáneos.")

    # -----------------------------------------
    # 7. Resumen por entidad y BUST
    # -----------------------------------------

    columnas_sumar = [
        "pd_A",
        "pd_B",
        "pd_C",
        "pd_D",
        "pd_pendientes",
        "pc_A",
        "pc_B",
        "pc_E",
        "pc_pendientes",
    ]

    resumen = (
        datos.groupby(grupos, dropna=False)[columnas_sumar].sum().reset_index()
    )

    # -----------------------------------------
    # 8. Tasas históricas
    # -----------------------------------------

    numerador_pd = resumen["pd_B"] + resumen["pd_C"] + resumen["pd_D"]

    numerador_pc = resumen["pc_B"] + resumen["pc_E"]

    resumen["PD"] = numerador_pd.div(resumen["pd_A"].replace(0, np.nan))

    resumen["PC"] = numerador_pc.div(resumen["pc_A"].replace(0, np.nan))

    resumen["PD_pct"] = resumen["PD"] * 100
    resumen["PC_pct"] = resumen["PC"] * 100

    return resumen, datos

""" PD y PC proyectadas """

def proyectar_pd_pc(
    pd_pc_app,
    df_macro_historico,
    df_macro_proyectado,
    df_operaciones_crediticias,
    df_tipo_cambio,
):
    """
    Proyecta las probabilidades de incumplimiento (PD) y cura (PC)
    para 2026-2028, por entidad, categoría BUST y escenario
    macroeconómico, mediante regresiones logísticas binomiales.

    Los modelos incorporan interceptos por segmento, restricciones
    de signo y penalización del coeficiente macroeconómico.
    Adicionalmente, proyecta el crecimiento crediticio y los saldos
    de cartera, manteniendo la participación de cada segmento.

    Parámetros
    ----------
    pd_pc_app : pd.DataFrame
        Datos históricos de PD y PC por entidad, categoría BUST
        y cohorte, incluyendo los conteos de cada transición.

    df_macro_historico : pd.DataFrame
        Variables macroeconómicas históricas utilizadas para
        estimar los modelos y calcular las referencias de crecimiento.

    df_macro_proyectado : pd.DataFrame
        Variables macroeconómicas proyectadas por escenario
        para los años 2026, 2027 y 2028.

    df_operaciones_crediticias : pd.DataFrame
        Información histórica de operaciones crediticias utilizada
        para calcular los saldos y el crecimiento de la cartera.

    df_tipo_cambio : pd.DataFrame
        Tipos de cambio históricos utilizados para convertir
        los saldos en moneda extranjera a moneda nacional.

    Retorna
    -------
    proyecciones : pd.DataFrame
        DataFrame con las proyecciones por entidad, categoría BUST,
        escenario y año. Incluye las siguientes variables:

        - PD: probabilidad de incumplimiento proyectada.
        - PC: probabilidad de cura proyectada.
        - PD_pct: PD expresada en porcentaje.
        - PC_pct: PC expresada en porcentaje.
        - crecimiento_pct: crecimiento anual proyectado de cartera.
        - saldo_proyectado_mn: saldo de cartera proyectado en
          moneda nacional.
    """

    variables = {"PD": "tc_crc_usd", "PC": "tasa_mn_pct"}
    claves = ["entidad", "categorias_bust"]
    h = pd_pc_app.copy()
    h["ano"] = h["cohorte"].astype(int) + 1
    h["pd_n"] = h[["pd_B", "pd_C", "pd_D"]].sum(axis=1)
    h["pc_n"] = h[["pc_B", "pc_E"]].sum(axis=1)
    h["segmento"] = (
        h["entidad"].astype(str) + " | " + h["categorias_bust"].astype(str)
    )

    mh = df_macro_historico.copy()
    mp = df_macro_proyectado.copy()
    mh["ano"] = pd.to_datetime(mh["fecha"]).dt.year
    mp["ano"] = pd.to_datetime(mp["fecha"]).dt.year
    cols = list(variables.values())
    mh = mh.groupby("ano", as_index=False)[cols].mean()
    mp = mp.groupby(["escenario", "ano"], as_index=False)[cols].mean()
    mp = mp.loc[mp["ano"].isin([2026, 2027, 2028])].copy()
    h = h.merge(mh, on="ano", how="left", validate="many_to_one")
    if h[cols].isna().any().any() or mp[cols].isna().any().any():
        raise ValueError(
            "Faltan valores macroeconomicos historicos o proyectados"
        )

    segmentos = sorted(h["segmento"].unique())
    k = len(segmentos)
    catalogo = (
        h[claves + ["segmento"]]
        .drop_duplicates()
        .set_index("segmento")
        .loc[segmentos]
        .reset_index()
    )
    resultados = []

    for metrica, variable in variables.items():
        num, den = ("pd_n", "pd_A") if metrica == "PD" else ("pc_n", "pc_A")
        d = h.loc[h[den] > 0].copy()
        y, n = d[num].to_numpy(float), d[den].to_numpy(float)
        if np.any(y < 0) or np.any(y > n):
            raise ValueError(f"Conteos binomiales invalidos para {metrica}")

        centro = d[variable].mean()
        escala = d[variable].std(ddof=0)
        if not np.isfinite(escala) or escala < 1e-10:
            escala = 1.0
        x_seg = pd.get_dummies(
            pd.Categorical(d["segmento"], categories=segmentos), dtype=float
        ).to_numpy()
        x_macro = ((d[variable].to_numpy(float) - centro) / escala).reshape(
            -1, 1
        )
        X = np.column_stack([x_seg, x_macro])

        totales = (
            d.groupby("segmento")[[num, den]]
            .sum()
            .reindex(segmentos, fill_value=0)
        )
        p0 = ((totales[num] + 0.5) / (totales[den] + 1)).clip(1e-6, 1 - 1e-6)
        beta0 = np.r_[np.log(p0 / (1 - p0)).to_numpy(), 0.0]
        limites = [(None, None)] * k + (
            [(0, None)] if metrica == "PD" else [(None, 0)]
        )

        def objetivo(beta):
            z = X @ beta
            return np.sum(n * np.logaddexp(0, z) - y * z) + 5.0 * beta[-1] ** 2

        def gradiente(beta):
            g = X.T @ (n * expit(X @ beta) - y)
            g[-1] += 10.0 * beta[-1]
            return g

        ajuste = minimize(
            objetivo,
            beta0,
            jac=gradiente,
            bounds=limites,
            method="L-BFGS-B",
            options={"maxiter": 3000},
        )
        if not ajuste.success:
            raise RuntimeError(f"No convergio {metrica}: {ajuste.message}")

        for _, fila in mp.iterrows():
            xf = np.column_stack(
                [
                    np.eye(k),
                    np.full((k, 1), (fila[variable] - centro) / escala),
                ]
            )
            temp = catalogo[claves].copy()
            temp["escenario"] = fila["escenario"]
            temp["ano"] = int(fila["ano"])
            temp[metrica] = expit(xf @ ajuste.x)
            resultados.append(temp)

    indices = claves + ["escenario", "ano"]
    pd_f = pd.concat(
        [r for r in resultados if "PD" in r.columns], ignore_index=True
    )
    pc_f = pd.concat(
        [r for r in resultados if "PC" in r.columns], ignore_index=True
    )
    proyecciones = pd_f.merge(pc_f, on=indices, validate="one_to_one")
    proyecciones["PD_pct"] = 100 * proyecciones["PD"]
    proyecciones["PC_pct"] = 100 * proyecciones["PC"]
    # Crecimiento crediticio y saldos, manteniendo cuotas de mercado
    op = df_operaciones_crediticias.copy()
    fx = df_tipo_cambio.copy()
    op["fecha"] = pd.to_datetime(op["fecha"]).dt.normalize()
    fx["fecha"] = pd.to_datetime(fx["fecha"]).dt.normalize()
    op = op.loc[
        op["fecha"].dt.year.between(2022, 2025) & op["fecha"].dt.month.eq(12)
    ].copy()
    op = op.loc[
        ~op["categorias_bust"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["otro", "otros"])
        & pd.to_numeric(op["back_to_back"], errors="coerce").eq(0)
    ].copy()
    op["moneda"] = op["moneda"].astype(str).str.strip().str.upper()
    fx["moneda"] = fx["moneda"].astype(str).str.strip().str.upper()
    if fx.duplicated(["fecha", "moneda"]).any():
        raise ValueError("Hay tipos de cambio duplicados por fecha y moneda")
    op = op.merge(
        fx[["fecha", "moneda", "crc_por_unidad"]],
        on=["fecha", "moneda"],
        how="left",
        validate="many_to_one",
    )
    if (~op["moneda"].isin(["MN", "ME"])).any():
        raise ValueError("Se encontraron monedas distintas de MN y ME")
    if op.loc[op["moneda"].eq("ME"), "crc_por_unidad"].isna().any():
        raise ValueError("Faltan tipos de cambio para operaciones en ME")
    op["saldo_mn"] = np.where(
        op["moneda"].eq("MN"),
        op["saldo_principal_mo"],
        op["saldo_principal_mo"] * op["crc_por_unidad"],
    )
    if op["saldo_mn"].isna().any():
        raise ValueError("Hay saldos sin conversion")
    op["ano"] = op["fecha"].dt.year
    cartera_historica = op.groupby("ano")["saldo_mn"].sum().sort_index()
    if not set(range(2022, 2026)).issubset(cartera_historica.index):
        raise ValueError("Faltan cierres anuales 2022-2025")
    tasas_hist = cartera_historica.pct_change().dropna() * 100
    referencia = tasas_hist.median()
    macro_ref = df_macro_historico.copy()
    macro_ref["ano"] = pd.to_datetime(macro_ref["fecha"]).dt.year
    macro_ref = macro_ref.loc[macro_ref["ano"].between(2022, 2025)]
    pib_ref = macro_ref["pib_interanual_pct"].mean()
    tmn_ref = macro_ref["tasa_mn_pct"].mean()
    if not np.isfinite([referencia, pib_ref, tmn_ref]).all():
        raise ValueError("Faltan datos para calcular crecimiento crediticio")

    # Supuestos documentados: 0.5 pp por punto de PIB, -0.5 pp por punto de TMN,
    # y +0.5 pp de recuperacion por cada ano posterior a 2026.
    macro_crec = df_macro_proyectado.copy()
    macro_crec["ano"] = pd.to_datetime(macro_crec["fecha"]).dt.year
    macro_crec = macro_crec.groupby(["escenario", "ano"], as_index=False)[
        ["pib_interanual_pct", "tasa_mn_pct"]
    ].mean()
    macro_crec = macro_crec.loc[macro_crec["ano"].between(2026, 2028)].copy()
    macro_crec["crecimiento_pct"] = (
        referencia
        + 0.5 * (macro_crec["pib_interanual_pct"] - pib_ref)
        - 0.5 * (macro_crec["tasa_mn_pct"] - tmn_ref)
        + 0.5 * (macro_crec["ano"] - 2026)
    )
    if (macro_crec["crecimiento_pct"] <= -100).any():
        raise ValueError("Crecimiento inferior o igual a -100%")
    for escenario, d in macro_crec.groupby("escenario"):
        d = d.sort_values("ano")
        if len(d) != 3 or not (d["crecimiento_pct"].diff().iloc[1:] > 0).all():
            raise ValueError(
                f"Las tasas no son estrictamente crecientes: {escenario}"
            )
    tasas = macro_crec.pivot(
        index="ano", columns="escenario", values="crecimiento_pct"
    )
    if {"Base", "Adverso"}.issubset(tasas.columns) and not (
        tasas["Adverso"] < tasas["Base"]
    ).all():
        raise ValueError("El crecimiento adverso no es inferior al base")

    inicial = (
        op.loc[op["ano"].eq(2025)]
        .groupby(claves, as_index=False)["saldo_mn"]
        .sum()
        .rename(columns={"saldo_mn": "saldo_proyectado_mn"})
    )
    saldos = []
    for escenario, grupo in macro_crec.groupby("escenario"):
        cartera = inicial.copy()
        for fila in grupo.sort_values("ano").itertuples(index=False):
            cartera["saldo_proyectado_mn"] *= 1 + fila.crecimiento_pct / 100
            paso = cartera.copy()
            paso["escenario"] = escenario
            paso["ano"] = int(fila.ano)
            paso["crecimiento_pct"] = fila.crecimiento_pct
            saldos.append(paso)
    saldos = pd.concat(saldos, ignore_index=True)
    proyecciones = proyecciones.merge(
        saldos, on=indices, how="left", validate="one_to_one"
    )
    if (
        proyecciones[["saldo_proyectado_mn", "crecimiento_pct"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Hay segmentos PD/PC sin saldo proyectado")
    return proyecciones.sort_values(["escenario", "ano"] + claves).reset_index(
        drop=True
    )

# %%

# ==========================================================
# CONSTRUCCIÓN DE PD Y PC HISTÓRICAS Y PROYECTADAS
# ==========================================================

# 1. PD Y PC HISTÓRICAS
# Cohortes 2019-2024, observadas 12 meses después

resultados_historicos = []

for cohorte in range(2019, 2025):

    fecha_t0 = f"{cohorte}-12-31"
    fecha_t12 = f"{cohorte + 1}-12-31"

    # Clasificación inicial
    inventario_t0 = clasificar_inventario_inicial_crediticio(
        df_operaciones_crediticias, fecha_t0
    )

    # Seguimiento a 12 meses
    seguimiento_t12 = evaluar_operaciones_t12(
        df_operaciones_crediticias, inventario_t0, fecha_t0
    )

    # Evaluación de modificaciones
    evaluacion_mods = evaluar_crecimiento_modificaciones(
        df_operaciones_crediticias, seguimiento_t12, fecha_t0
    )

    # Clasificación final
    clasificacion_t12 = clasificar_operaciones_t12(evaluacion_mods, fecha_t12)

    # Calcular PD y PC históricas
    resumen, detalle = calcular_pd_pc_historicas(clasificacion_t12)

    resumen["cohorte"] = cohorte

    resultados_historicos.append(resumen)

# DataFrame 1: PD y PC históricas
pd_pc_historicas = pd.concat(resultados_historicos, ignore_index=True)

pd_pc_historicas = pd_pc_historicas.sort_values(
    ["cohorte", "entidad", "categorias_bust"]
).reset_index(drop=True)

# Reemplazar NaN de PC y PC_pct por cero
pd_pc_historicas[["PC", "PC_pct"]] = pd_pc_historicas[["PC", "PC_pct"]].fillna(
    0
)

# 2. PD Y PC PROYECTADAS
# Incluye crecimiento crediticio y saldos proyectados

proyecciones_pd_pc = proyectar_pd_pc(
    pd_pc_historicas,
    df_macro_historico,
    df_macro_proyectado,
    df_operaciones_crediticias,
    df_tipo_cambio,
)

# %%

""" Recuperaciones y severidad de la pérdida """

def seleccionar_garantias_admisibles(df_garantias_operaciones):
    """
    Selecciona las garantías admisibles según su tipo y grado de prelación.

    Se consideran admisibles las garantías de los tipos permitidos que
    tengan primer grado de prelación, así como las de segundo grado
    cuando la misma entidad mantiene también el primer grado sobre
    la garantía.

    Parámetros
    ----------
    df_garantias_operaciones : pd.DataFrame
        DataFrame con la información de las garantías asociadas a las
        operaciones, incluyendo tipo, identificador, entidad acreedora
        y grado de prelación.

    Retorna
    -------
    pd.DataFrame
        DataFrame con las garantías que cumplen los criterios de
        admisibilidad, conservando las columnas originales.
    """

    g = df_garantias_operaciones.copy()

    # Tipos de garantía permitidos
    tipos_validos = [
        "Edificio",
        "Residencia_habitual",
        "Terreno",
        "Titulo_valor",
        "Vehiculo",
    ]

    g = g[g["tipo_gar"].isin(tipos_validos)].copy()

    # Normalizar prelación
    g["grado_prelacion_grav"] = pd.to_numeric(
        g["grado_prelacion_grav"], errors="coerce"
    )

    # Primer grado: admisible
    primer_grado = g["grado_prelacion_grav"].eq(1)

    # Segundo grado: admisible si la misma entidad
    # mantiene también el primer grado sobre la garantía
    claves = ["garantia_id", "entidad"]

    titulares_primer_grado = (
        g.loc[
            primer_grado & g["acreedor_entidad_grav"].eq(g["entidad"]), claves
        ]
        .drop_duplicates()
        .assign(tiene_primer_grado=True)
    )

    g = g.merge(
        titulares_primer_grado, on=claves, how="left", validate="many_to_one"
    )

    segundo_valido = (
        g["grado_prelacion_grav"].eq(2)
        & g["acreedor_entidad_grav"].eq(g["entidad"])
        & g["tiene_primer_grado"].fillna(False)
    )

    g = g.loc[primer_grado | segundo_valido].copy()

    return g.drop(columns="tiene_primer_grado")

def valorar_garantias_2025(
    garantias_admisibles, df_operaciones_crediticias, df_tipo_cambio
):
    """
    Calcula el valor de las garantías admisibles al cierre de 2025,
    considerando gravámenes preferentes, asignaciones compartidas
    y límites de cobertura según el saldo de las operaciones.

    Parámetros
    ----------
    garantias_admisibles : pd.DataFrame
        Base de garantías admisibles vinculadas a las operaciones
        crediticias, incluyendo sus valores, gravámenes y porcentajes
        de asignación.

    df_operaciones_crediticias : pd.DataFrame
        Base de operaciones crediticias utilizada para identificar
        las operaciones vigentes al cierre de 2025 y sus saldos.

    df_tipo_cambio : pd.DataFrame
        Base de tipos de cambio utilizada para convertir los saldos
        y valores de las garantías a colones costarricenses.

    Retorna
    -------
    g : pd.DataFrame
        Base de garantías vinculadas a las operaciones elegibles,
        con sus valores ajustados en colones y el valor final del
        colateral asignado a cada operación, limitado por el valor
        disponible de la garantía y el saldo de la operación.
    """

    corte = pd.Timestamp("2025-12-31")

    g = garantias_admisibles.copy()

    # Conservar una sola fila por garantía y operación.
    # Si existen grados 1 y 2, priorizar el grado 1.
    g = (
        g.sort_values("grado_prelacion_grav")
        .drop_duplicates(
            subset=["entidad", "garantia_id", "operacion_id"], keep="first"
        )
        .copy()
    )

    op = df_operaciones_crediticias.copy()

    tc = df_tipo_cambio.copy()

    # Operaciones vigentes al cierre de 2025
    op["fecha"] = pd.to_datetime(op["fecha"])

    op = op.loc[
        op["fecha"].eq(corte)
        & ~op["categorias_bust"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["otro", "otros"])
        & op["back_to_back"].eq(0),
        [
            "operacion_id",
            "entidad",
            "categorias_bust",
            "moneda",
            "saldo_principal_mo",
        ],
    ].copy()

    if op.duplicated(["operacion_id", "entidad"]).any():
        raise ValueError("Existen operaciones duplicadas al corte.")

    # Tipo de cambio al cierre
    tc["fecha"] = pd.to_datetime(tc["fecha"])

    tc = tc.loc[
        tc["fecha"].eq(corte), ["moneda", "crc_por_unidad"]
    ].drop_duplicates("moneda")

    # Saldo de cada operación en CRC
    op = op.merge(tc, on="moneda", how="left", validate="many_to_one")

    if op["crc_por_unidad"].isna().any():
        raise ValueError("Faltan tipos de cambio para operaciones.")

    op["saldo_operacion_crc"] = op["saldo_principal_mo"] * op["crc_por_unidad"]

    # Vincular garantías con operaciones
    g = g.merge(
        op[
            [
                "operacion_id",
                "entidad",
                "categorias_bust",
                "saldo_operacion_crc",
            ]
        ],
        on=["operacion_id", "entidad"],
        how="inner",
        validate="many_to_one",
    )

    # Conversión de garantías a CRC
    g = g.merge(tc, on="moneda", how="left", validate="many_to_one")

    if g["crc_por_unidad"].isna().any():
        raise ValueError("Faltan tipos de cambio para garantías.")

    # Deducción de gravámenes preferentes para no terrenos
    preferente = np.where(
        g["tipo_gar"].eq("Terreno"), 0, g["monto_preferente_mo"].fillna(0)
    )

    g["avaluo_ajustado_crc"] = (
        g["monto_maximo_cobertura_mo"] - preferente
    ).clip(lower=0) * g["crc_por_unidad"]

    # Valor asignado a cada operación
    g["valor_asignado_crc"] = (
        g["avaluo_ajustado_crc"] * g["peso_asignacion_prop"]
    ).clip(lower=0)

    # Límite 1: cada garantía compartida no puede
    # superar su valor disponible.
    claves_garantia = ["entidad", "garantia_id"]

    valor_disponible = g.groupby(claves_garantia)[
        "avaluo_ajustado_crc"
    ].transform("min")

    total_garantia = g.groupby(claves_garantia)[
        "valor_asignado_crc"
    ].transform("sum")

    factor_garantia = np.where(
        total_garantia > 0, np.minimum(1, valor_disponible / total_garantia), 0
    )

    g["valor_ajustado_garantia_crc"] = (
        g["valor_asignado_crc"] * factor_garantia
    )

    # Límite 2: la cobertura conjunta de una operación
    # no puede superar su saldo.
    claves_operacion = ["entidad", "operacion_id"]

    total_operacion = g.groupby(claves_operacion)[
        "valor_ajustado_garantia_crc"
    ].transform("sum")

    factor_operacion = np.where(
        total_operacion > 0,
        np.minimum(1, g["saldo_operacion_crc"] / total_operacion),
        0,
    )

    g["valor_colateral_crc"] = (
        g["valor_ajustado_garantia_crc"] * factor_operacion
    )

    return g

def calcular_recuperaciones_historicas(
    df_operaciones_bienes_realizables, df_flujos_recuperacion
):
    """
    Calcula las recuperaciones históricas netas de los bienes adjudicados
    y vendidos, considerando los gastos de venta y administración.
    Genera un resumen por tipo de bien con las tasas de recuperación,
    descuentos y plazos medianos de venta.

    Parámetros
    ----------
    df_operaciones_bienes_realizables : pd.DataFrame
        Base de bienes adjudicados con información sobre su valor de
        adjudicación, venta, gastos asociados, fechas y tipo de bien.

    df_flujos_recuperacion : pd.DataFrame
        Base de flujos de recuperación con los costos históricos de
        administración asociados a cada bien.

    Retorna
    -------
    bienes : pd.DataFrame
        Detalle de los bienes vendidos con sus recuperaciones netas,
        costos de administración, plazos de venta y tasas de recuperación.

    resumen : pd.DataFrame
        Resumen agregado por tipo de bien con el número de bienes vendidos,
        valor total de adjudicación, recuperación neta total, plazo mediano
        de venta, tasa de recuperación y descuento de recuperación.
    """

    bienes = df_operaciones_bienes_realizables.copy()
    flujos = df_flujos_recuperacion.copy()

    # Solo bienes efectivamente vendidos
    bienes = bienes.loc[
        bienes["vendido"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "si", "sí"])
    ].copy()

    # Costos históricos de administración
    costos = (
        flujos.loc[flujos["concepto"].eq("Administracion")]
        .groupby("bien_id")["importe_crc"]
        .apply(lambda x: x.abs().sum())
        .rename("costo_administracion_crc")
        .reset_index()
    )

    bienes = bienes.merge(
        costos, on="bien_id", how="left", validate="one_to_one"
    )

    bienes["costo_administracion_crc"] = bienes[
        "costo_administracion_crc"
    ].fillna(0)

    bienes["recuperacion_neta_crc"] = (
        bienes["venta_bruta_crc"].fillna(0)
        - bienes["gasto_venta_crc"].fillna(0).abs()
        - bienes["costo_administracion_crc"]
    ).clip(lower=0)

    bienes["valor_adjudicacion_crc"] = pd.to_numeric(
        bienes["valor_adjudicacion_crc"], errors="coerce"
    )

    bienes["fecha_adjudicacion"] = pd.to_datetime(bienes["fecha_adjudicacion"])

    bienes["fecha_venta"] = pd.to_datetime(bienes["fecha_venta"])

    bienes["plazo_venta_meses"] = (
        bienes["fecha_venta"] - bienes["fecha_adjudicacion"]
    ).dt.days / 30.4375

    bienes["tasa_recuperacion"] = np.where(
        bienes["valor_adjudicacion_crc"] > 0,
        bienes["recuperacion_neta_crc"] / bienes["valor_adjudicacion_crc"],
        np.nan,
    )

    resumen = (
        bienes.groupby("tipo")
        .agg(
            bienes_vendidos=("bien_id", "nunique"),
            valor_adjudicacion_crc=("valor_adjudicacion_crc", "sum"),
            recuperacion_neta_crc=("recuperacion_neta_crc", "sum"),
            plazo_venta_meses=("plazo_venta_meses", "median"),
        )
        .reset_index()
    )

    resumen["tasa_recuperacion"] = np.where(
        resumen["valor_adjudicacion_crc"] > 0,
        resumen["recuperacion_neta_crc"] / resumen["valor_adjudicacion_crc"],
        np.nan,
    )

    resumen["descuento_recuperacion"] = 1 - resumen["tasa_recuperacion"].clip(
        0, 1
    )

    return bienes, resumen

def proyectar_recuperaciones_garantias(
    garantias_valoradas,
    resumen_recuperaciones,
    df_macro_historico,
    df_macro_proyectado,
):
    """
    Proyecta el valor de las garantías y las recuperaciones esperadas
    para 2026-2028, utilizando tasas históricas de recuperación y
    la evolución proyectada de índices macroeconómicos.

    Parámetros
    ----------
    garantias_valoradas : pd.DataFrame
        Base de garantías valoradas, con su tipo, entidad,
        categoría BUST y valor del colateral en colones.

    resumen_recuperaciones : pd.DataFrame
        Resumen por tipo de garantía con las tasas históricas
        de recuperación y los plazos de venta estimados.

    df_macro_historico : pd.DataFrame
        Base histórica de índices macroeconómicos. Se utilizan
        los valores de diciembre de 2025 como referencia.

    df_macro_proyectado : pd.DataFrame
        Proyecciones de índices macroeconómicos por escenario
        y año para el período 2026-2028.

    Retorna
    -------
    pd.DataFrame
        Base con las recuperaciones proyectadas por entidad,
        categoría BUST, tipo de garantía, escenario y año.
        Incluye el valor original del colateral, el factor de
        actualización macroeconómica, el colateral proyectado
        y la recuperación proyectada en colones.
    """

    g = garantias_valoradas.copy()
    r = resumen_recuperaciones.copy()
    mh = df_macro_historico.copy()
    mp = df_macro_proyectado.copy()

    # Relación entre tipos de garantía e índices macroeconómicos
    mapa_indices = {
        "Edificio": "indice_inmuebles",
        "Residencia_habitual": "indice_inmuebles",
        "Terreno": "indice_inmuebles",
        "Vehiculo": "indice_vehiculos",
        "Titulo_valor": "indice_titulos",
    }

    # Índices de referencia al cierre de 2025
    mh["fecha"] = pd.to_datetime(mh["fecha"])
    base = mh.loc[mh["fecha"].eq("2025-12-31")]

    if base.empty:
        raise ValueError("No hay índices macroeconómicos al 31/12/2025.")

    indices_base = base.iloc[0]

    # Resumen del colateral por entidad, BUST y tipo
    colateral = g.groupby(
        ["entidad", "categorias_bust", "tipo_gar"], as_index=False
    )["valor_colateral_crc"].sum()

    colateral = colateral.merge(
        r[["tipo", "tasa_recuperacion", "plazo_venta_meses"]],
        left_on="tipo_gar",
        right_on="tipo",
        how="left",
        validate="many_to_one",
    )

    if colateral["tasa_recuperacion"].isna().any():
        raise ValueError("Faltan tasas históricas de recuperación.")

    # Preparar escenarios
    mp = mp.loc[mp["ano"].isin([2026, 2027, 2028])].copy()

    escenarios = mp[
        [
            "escenario",
            "ano",
            "indice_inmuebles",
            "indice_vehiculos",
            "indice_titulos",
        ]
    ].drop_duplicates(["escenario", "ano"])

    # Cruzar cada segmento con cada escenario y año
    resultado = colateral.merge(escenarios, how="cross")

    # Factor de actualización por tipo de garantía
    resultado["factor_macro"] = np.nan

    for tipo, indice in mapa_indices.items():
        mascara = resultado["tipo_gar"].eq(tipo)

        resultado.loc[mascara, "factor_macro"] = (
            resultado.loc[mascara, indice] / indices_base[indice]
        )

    resultado["colateral_proyectado_crc"] = (
        resultado["valor_colateral_crc"] * resultado["factor_macro"]
    )

    resultado["recuperacion_proyectada_crc"] = resultado[
        "colateral_proyectado_crc"
    ] * resultado["tasa_recuperacion"].clip(0, 1)

    return resultado

def descontar_recuperaciones(
    recuperaciones_proyectadas, df_operaciones_crediticias, df_tipo_cambio
):
    """
    Calcula el valor presente de las recuperaciones proyectadas,
    utilizando tasas efectivas anuales ponderadas por saldo para
    cada combinación de entidad y categoría de cartera.

    Parámetros
    ----------
    recuperaciones_proyectadas : pd.DataFrame
        DataFrame con las recuperaciones proyectadas en CRC y los
        plazos estimados de venta en meses.

    df_operaciones_crediticias : pd.DataFrame
        DataFrame con las operaciones crediticias, sus saldos,
        tasas efectivas anuales y clasificaciones de cartera.

    df_tipo_cambio : pd.DataFrame
        DataFrame con los tipos de cambio utilizados para convertir
        los saldos de las operaciones a CRC.

    Retorna
    -------
    pd.DataFrame
        DataFrame con las recuperaciones proyectadas originales y
        las columnas adicionales:
        - tasa_descuento: tasa efectiva anual ponderada por saldo.
        - factor_descuento: factor aplicado según el plazo de venta.
        - recuperacion_vp_crc: valor presente de la recuperación en CRC.
    """

    r = recuperaciones_proyectadas.copy()
    op = df_operaciones_crediticias.copy()
    tc = df_tipo_cambio.copy()

    # Operaciones al cierre de 2025
    op["fecha"] = pd.to_datetime(op["fecha"])

    op = op.loc[
        op["fecha"].eq("2025-12-31")
        & op["back_to_back"].eq(0)
        & ~op["categorias_bust"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["otro", "otros"])
    ].copy()

    # Convertir saldos a CRC para ponderar tasas
    tc["fecha"] = pd.to_datetime(tc["fecha"])

    tc = tc.loc[
        tc["fecha"].eq("2025-12-31"), ["moneda", "crc_por_unidad"]
    ].drop_duplicates("moneda")

    op = op.merge(tc, on="moneda", how="left", validate="many_to_one")

    op["saldo_crc"] = op["saldo_principal_mo"] * op["crc_por_unidad"]

    # Tasa efectiva anual ponderada por saldo
    op["tasa_ponderada"] = op["tasa_efectiva_anual_prop"] * op["saldo_crc"]

    tasas = (
        op.groupby(["entidad", "categorias_bust"])
        .agg(
            tasa_numerador=("tasa_ponderada", "sum"),
            saldo_total=("saldo_crc", "sum"),
        )
        .reset_index()
    )

    tasas["tasa_descuento"] = tasas["tasa_numerador"] / tasas["saldo_total"]

    # Incorporar tasa de descuento a las recuperaciones
    r = r.merge(
        tasas[["entidad", "categorias_bust", "tasa_descuento"]],
        on=["entidad", "categorias_bust"],
        how="left",
        validate="many_to_one",
    )

    if r["tasa_descuento"].isna().any():
        raise ValueError("Faltan tasas de descuento.")

    if (r["tasa_descuento"] <= -1).any():
        raise ValueError("Existen tasas de descuento inválidas.")

    # Valor presente de las recuperaciones
    r["factor_descuento"] = (1 + r["tasa_descuento"]) ** (
        -r["plazo_venta_meses"] / 12
    )

    r["recuperacion_vp_crc"] = (
        r["recuperacion_proyectada_crc"] * r["factor_descuento"]
    )

    return r

def calcular_cobertura_recuperable(recuperaciones_vp, garantias_valoradas):
    """
    Calcula la cobertura recuperable y la pérdida dado el incumplimiento
    (LGL) de las operaciones garantizadas, por entidad, categoría BUST,
    escenario y año, limitando las recuperaciones al saldo garantizado.

    Parámetros
    ----------
    recuperaciones_vp : pd.DataFrame
        Contiene las recuperaciones a valor presente, expresadas en CRC,
        por entidad, categoría BUST, escenario y año.

    garantias_valoradas : pd.DataFrame
        Contiene las garantías valoradas y los saldos de las operaciones
        garantizadas, expresados en CRC.

    Retorna
    -------
    resultado : pd.DataFrame
        Contiene, por entidad, categoría BUST, escenario y año, el saldo
        garantizado, las recuperaciones a valor presente, las recuperaciones
        limitadas al saldo, la cobertura recuperable y la LGL garantizada.
    """

    # Saldo de operaciones con garantías, sin duplicarlas
    saldos = garantias_valoradas[
        ["entidad", "categorias_bust", "operacion_id", "saldo_operacion_crc"]
    ].drop_duplicates(["entidad", "operacion_id"])

    saldos = (
        saldos.groupby(["entidad", "categorias_bust"], as_index=False)[
            "saldo_operacion_crc"
        ]
        .sum()
        .rename(columns={"saldo_operacion_crc": "saldo_garantizado_crc"})
    )

    recuperaciones = recuperaciones_vp.groupby(
        ["entidad", "categorias_bust", "escenario", "ano"], as_index=False
    )["recuperacion_vp_crc"].sum()

    resultado = recuperaciones.merge(
        saldos,
        on=["entidad", "categorias_bust"],
        how="left",
        validate="many_to_one",
    )

    # No permitir recuperaciones superiores al saldo
    resultado["recuperacion_limitada_crc"] = np.minimum(
        resultado["recuperacion_vp_crc"], resultado["saldo_garantizado_crc"]
    )

    resultado["cobertura_recuperable"] = np.where(
        resultado["saldo_garantizado_crc"] > 0,
        resultado["recuperacion_limitada_crc"]
        / resultado["saldo_garantizado_crc"],
        0,
    )

    resultado["LGL_garantizada"] = 1 - resultado["cobertura_recuperable"]

    return resultado

def calcular_lgl_cartera(
    cobertura_recuperable, df_operaciones_crediticias, df_tipo_cambio
):
    """
    Calcula la pérdida dado el incumplimiento (LGL) por entidad y
    categoría de cartera, considerando las recuperaciones estimadas
    de garantías respecto al saldo total de la cartera.

    Parámetros
    ----------
    cobertura_recuperable : pd.DataFrame
        Base con las recuperaciones limitadas de garantías, expresadas
        en colones, por entidad y categoría de cartera.

    df_operaciones_crediticias : pd.DataFrame
        Base de operaciones crediticias utilizada para calcular los
        saldos de cartera al 31 de diciembre de 2025. Se excluyen
        operaciones back-to-back y categorías "Otro" u "Otros".

    df_tipo_cambio : pd.DataFrame
        Base de tipos de cambio utilizada para convertir los saldos
        de las operaciones a colones al 31 de diciembre de 2025.

    Retorna
    -------
    pd.DataFrame
        Base con los saldos totales de cartera, las recuperaciones
        limitadas al saldo de cada segmento y la LGL calculada
        por entidad y categoría de cartera.
    """

    op = df_operaciones_crediticias.copy()
    tc = df_tipo_cambio.copy()

    op["fecha"] = pd.to_datetime(op["fecha"])
    tc["fecha"] = pd.to_datetime(tc["fecha"])

    op = op.loc[
        op["fecha"].eq("2025-12-31")
        & op["back_to_back"].eq(0)
        & ~op["categorias_bust"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["otro", "otros"])
    ].copy()

    tc = tc.loc[
        tc["fecha"].eq("2025-12-31"), ["moneda", "crc_por_unidad"]
    ].drop_duplicates("moneda")

    op = op.merge(tc, on="moneda", how="left", validate="many_to_one")

    if op["crc_por_unidad"].isna().any():
        raise ValueError("Faltan tipos de cambio.")

    op["saldo_crc"] = op["saldo_principal_mo"] * op["crc_por_unidad"]

    cartera = (
        op.groupby(["entidad", "categorias_bust"], as_index=False)["saldo_crc"]
        .sum()
        .rename(columns={"saldo_crc": "saldo_cartera_crc"})
    )

    resultado = cobertura_recuperable.merge(
        cartera,
        on=["entidad", "categorias_bust"],
        how="left",
        validate="many_to_one",
    )

    if resultado["saldo_cartera_crc"].isna().any():
        raise ValueError("Faltan saldos de cartera para algunos segmentos.")

    # Recuperaciones de garantías frente a la cartera completa
    resultado["recuperacion_cartera_crc"] = np.minimum(
        resultado["recuperacion_limitada_crc"], resultado["saldo_cartera_crc"]
    )

    resultado["LGL_cartera"] = np.where(
        resultado["saldo_cartera_crc"] > 0,
        1
        - (
            resultado["recuperacion_cartera_crc"]
            / resultado["saldo_cartera_crc"]
        ),
        1,
    )

    return resultado

def valorar_bienes_pendientes(
    df_operaciones_bienes_realizables,
    resumen_recuperaciones,
    df_macro_historico,
    df_macro_proyectado,
):
    """
    Proyecta el valor de recuperación de los bienes adjudicados pendientes
    de venta para los años 2026-2028, considerando tasas históricas de
    recuperación, escenarios macroeconómicos y fracciones asignadas.

    Parámetros
    ----------
    df_operaciones_bienes_realizables : pd.DataFrame
        Base de bienes adjudicados, con su tipo, valor de adjudicación,
        estado de venta y fracción asignada.

    resumen_recuperaciones : pd.DataFrame
        Resumen de tasas históricas de recuperación y plazos de venta
        por tipo de bien.

    df_macro_historico : pd.DataFrame
        Indicadores macroeconómicos históricos utilizados como referencia
        para ajustar el valor de los bienes, con base en diciembre de 2025.

    df_macro_proyectado : pd.DataFrame
        Indicadores macroeconómicos proyectados por escenario y año
        para el período 2026-2028.

    Retorna
    -------
    resultado : pd.DataFrame
        Base de bienes pendientes de venta con su recuperación proyectada
        por escenario y año, incorporando el ajuste macroeconómico,
        la tasa histórica de recuperación y la fracción asignada.
    """

    bienes = df_operaciones_bienes_realizables.copy()
    tasas = resumen_recuperaciones.copy()
    macro = df_macro_proyectado.copy()

    # Seleccionar bienes no vendidos
    vendido = (
        bienes["vendido"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "si", "sí"])
    )

    bienes = bienes.loc[~vendido].copy()

    # Incorporar tasas históricas y plazos de venta
    bienes = bienes.merge(
        tasas[["tipo", "tasa_recuperacion", "plazo_venta_meses"]],
        on="tipo",
        how="left",
        validate="many_to_one",
    )

    if bienes["tasa_recuperacion"].isna().any():
        raise ValueError("Faltan tasas históricas de recuperación.")

    # Índices macroeconómicos por tipo de bien
    mapa = {
        "Edificio": "indice_inmuebles",
        "Residencia_habitual": "indice_inmuebles",
        "Terreno": "indice_inmuebles",
        "Vehiculo": "indice_vehiculos",
        "Titulo_valor": "indice_titulos",
    }

    mh = df_macro_historico.copy()
    mh["fecha"] = pd.to_datetime(mh["fecha"])

    indices_base = mh.loc[mh["fecha"].eq("2025-12-31")].iloc[0]

    escenarios = macro.loc[
        macro["ano"].isin([2026, 2027, 2028]),
        [
            "escenario",
            "ano",
            "indice_inmuebles",
            "indice_vehiculos",
            "indice_titulos",
        ],
    ].drop_duplicates(["escenario", "ano"])

    resultado = bienes.merge(escenarios, how="cross")

    resultado["factor_macro"] = np.nan

    for tipo, indice in mapa.items():
        mascara = resultado["tipo"].eq(tipo)

        resultado.loc[mascara, "factor_macro"] = (
            resultado.loc[mascara, indice] / indices_base[indice]
        )

    # Aplicar fracción asignada para evitar doble conteo
    resultado["valor_asignado_crc"] = (
        resultado["valor_adjudicacion_crc"]
        * resultado["fraccion_asignada_prop"]
    )

    resultado["recuperacion_proyectada_crc"] = (
        resultado["valor_asignado_crc"]
        * resultado["factor_macro"]
        * resultado["tasa_recuperacion"].clip(0, 1)
    )

    return resultado

def descontar_bienes_pendientes(bienes_pendientes, df_operaciones_crediticias):
    """
    Calcula el valor presente de las recuperaciones proyectadas de bienes
    pendientes de venta, utilizando las tasas efectivas anuales de las
    operaciones crediticias al cierre de 2025. Cuando no se dispone de
    una tasa, utiliza la mediana histórica de la entidad correspondiente.

    Parámetros
    ----------
    bienes_pendientes : pd.DataFrame
        DataFrame con los bienes pendientes de venta, sus recuperaciones
        proyectadas y los plazos estimados de venta en meses.

    df_operaciones_crediticias : pd.DataFrame
        DataFrame con el historial de operaciones crediticias, incluyendo
        las tasas efectivas anuales, entidades y fechas de observación.

    Retorna
    -------
    pd.DataFrame
        DataFrame de bienes pendientes con las tasas de descuento,
        los factores de descuento y las recuperaciones proyectadas
        expresadas a valor presente en colones (recuperacion_vp_crc).
    """

    b = bienes_pendientes.copy()
    op = df_operaciones_crediticias.copy()

    op["fecha"] = pd.to_datetime(op["fecha"])

    # Tasas de las operaciones al cierre de 2025
    tasas = op.loc[
        op["fecha"].eq("2025-12-31"),
        ["entidad", "operacion_id", "tasa_efectiva_anual_prop"],
    ].drop_duplicates(["entidad", "operacion_id"])

    b = b.merge(
        tasas,
        on=["entidad", "operacion_id"],
        how="left",
        validate="many_to_one",
    )

    # Para operaciones que ya no están vigentes,
    # utilizar la mediana histórica de tasas de la entidad.
    tasas_entidad = op.groupby("entidad")["tasa_efectiva_anual_prop"].median()

    b["tasa_descuento"] = b["tasa_efectiva_anual_prop"].fillna(
        b["entidad"].map(tasas_entidad)
    )

    if b["tasa_descuento"].isna().any():
        raise ValueError("Faltan tasas de descuento.")

    if (b["tasa_descuento"] <= -1).any():
        raise ValueError("Existen tasas de descuento inválidas.")

    # Valor presente de la recuperación
    b["factor_descuento"] = (1 + b["tasa_descuento"]) ** (
        -b["plazo_venta_meses"] / 12
    )

    b["recuperacion_vp_crc"] = (
        b["recuperacion_proyectada_crc"] * b["factor_descuento"]
    )

    return b

def actualizar_avaluos_terrenos(garantias_admisibles, df_macro_historico):
    """
    Actualiza el valor de las garantías de tipo terreno a diciembre de 2025,
    utilizando la variación histórica del índice de precios de inmuebles.

    Parámetros
    ----------
    garantias_admisibles : pd.DataFrame
        Base de garantías admisibles con el tipo de garantía, la fecha
        del avalúo y el monto máximo de cobertura.

    df_macro_historico : pd.DataFrame
        Base de variables macroeconómicas históricas que contiene
        las fechas y el índice de precios de inmuebles.

    Retorna
    -------
    pd.DataFrame
        Base de garantías con el monto máximo de cobertura actualizado
        para terrenos y el factor de actualización aplicado. Las demás
        garantías conservan su monto original.
    """

    g = garantias_admisibles.copy()
    mh = df_macro_historico.copy()

    g["fecha_avaluo_gar"] = pd.to_datetime(g["fecha_avaluo_gar"])

    mh["fecha"] = pd.to_datetime(mh["fecha"])

    # Índice mensual de inmuebles
    mh["mes"] = mh["fecha"].dt.to_period("M")

    indices = mh.sort_values("fecha").drop_duplicates("mes", keep="last")[
        ["mes", "indice_inmuebles"]
    ]

    # Índice de referencia: diciembre de 2025
    indice_2025 = indices.loc[
        indices["mes"].eq(pd.Period("2025-12")), "indice_inmuebles"
    ].iloc[0]

    # Vincular índice del mes del avalúo
    g["mes"] = g["fecha_avaluo_gar"].dt.to_period("M")

    g = g.merge(indices, on="mes", how="left", validate="many_to_one")

    # Factor de actualización únicamente para terrenos
    g["factor_actualizacion_terreno"] = 1.0

    mascara = (
        g["tipo_gar"].eq("Terreno")
        & g["indice_inmuebles"].notna()
        & g["indice_inmuebles"].gt(0)
    )

    g.loc[mascara, "factor_actualizacion_terreno"] = (
        indice_2025 / g.loc[mascara, "indice_inmuebles"]
    )

    # Actualizar el monto de cobertura utilizado
    # como aproximación del avalúo
    g["monto_maximo_cobertura_mo"] = (
        g["monto_maximo_cobertura_mo"] * g["factor_actualizacion_terreno"]
    )

    return g.drop(columns=["mes", "indice_inmuebles"])

def preparar_insumos_secciones_4_5(proyecciones_pd_pc, lgl_cartera):
    """
    Prepara los insumos para las secciones 4 y 5, integrando las
    proyecciones de PD y PC con los resultados de LGL de la sección 3.
    Calcula la LGD ajustada por cura y define la exposición proyectada.

    Parámetros
    ----------
    proyecciones_pd_pc : pd.DataFrame
        Resultados proyectados de PD, PC y saldos por entidad,
        categoría BUST, escenario y año.

    lgl_cartera : pd.DataFrame
        Resultados de LGL, saldos garantizados y recuperaciones
        de cartera por entidad, categoría BUST, escenario y año.

    Retorna
    -------
    pd.DataFrame
        Base integrada con PD, PC, LGL, LGD ajustada por cura y
        exposición proyectada en moneda nacional (EAD), ordenada
        por entidad, categoría BUST, escenario y año.

    """

    claves = ["entidad", "categorias_bust", "escenario", "ano"]

    # Seleccionar resultados de la sección 3
    columnas_lgl = claves + [
        "LGL_cartera",
        "LGL_garantizada",
        "saldo_garantizado_crc",
        "saldo_cartera_crc",
        "recuperacion_cartera_crc",
    ]

    # Unir resultados de secciones 2 y 3
    insumos = proyecciones_pd_pc.merge(
        lgl_cartera[columnas_lgl], on=claves, how="left", validate="one_to_one"
    )

    # Validar integridad
    if insumos["LGL_cartera"].isna().any():
        raise ValueError("Existen segmentos sin LGL.")

    for columna in ["PD", "PC", "LGL_cartera"]:
        if not insumos[columna].between(0, 1).all():
            raise ValueError(f"Valores inválidos en {columna}.")

    # LGD ajustada por probabilidad de cura
    insumos["LGD"] = insumos["LGL_cartera"] * (1 - insumos["PC"])

    # Exposición proyectada (moneda nacional)
    insumos["EAD_proyectada_mn"] = insumos["saldo_proyectado_mn"]

    return insumos.sort_values(claves).reset_index(drop=True)

# %%

# ==========================================================
# CONSTRUCCIÓN DE LA LGL
# ==========================================================

# 1. Seleccionar garantías admisibles
garantias_admisibles = seleccionar_garantias_admisibles(
    df_garantias_operaciones
)

# 2. Actualizar avalúos de terrenos a diciembre de 2025
garantias_actualizadas = actualizar_avaluos_terrenos(
    garantias_admisibles, df_macro_historico
)

# 3. Valorar garantías de las operaciones
garantias_valoradas = valorar_garantias_2025(
    garantias_actualizadas, df_operaciones_crediticias, df_tipo_cambio
)

# 4. Calcular recuperaciones históricas
detalle_recuperaciones, resumen_recuperaciones = (
    calcular_recuperaciones_historicas(
        df_operaciones_bienes_realizables, df_flujos_recuperacion
    )
)

# 5. Proyectar recuperaciones de garantías
recuperaciones_proyectadas = proyectar_recuperaciones_garantias(
    garantias_valoradas,
    resumen_recuperaciones,
    df_macro_historico,
    df_macro_proyectado,
)

# 6. Descontar recuperaciones a valor presente
recuperaciones_vp = descontar_recuperaciones(
    recuperaciones_proyectadas, df_operaciones_crediticias, df_tipo_cambio
)

# 7. Calcular cobertura recuperable
cobertura_recuperable = calcular_cobertura_recuperable(
    recuperaciones_vp, garantias_valoradas
)

# 8. Calcular LGL para toda la cartera
lgl_cartera = calcular_lgl_cartera(
    cobertura_recuperable, df_operaciones_crediticias, df_tipo_cambio
)

# 9. Valorar bienes adjudicados pendientes de venta
bienes_pendientes = valorar_bienes_pendientes(
    df_operaciones_bienes_realizables,
    resumen_recuperaciones,
    df_macro_historico,
    df_macro_proyectado,
)

# 10. Descontar bienes pendientes a valor presente
bienes_pendientes_vp = descontar_bienes_pendientes(
    bienes_pendientes, df_operaciones_crediticias
)

# ============================================================
# ENTREGA A SECCIONES 4 Y 5
# ============================================================

# 11. Integrar PD, PC y LGL
# proyecciones_pd_pc debe haberse calculado en la sección 2
insumos_secciones_4_5 = preparar_insumos_secciones_4_5(
    proyecciones_pd_pc, lgl_cartera
)

# Se eliminan variables innecesarias
variables_eliminar = [
    "garantias_admisibles",
    "garantias_actualizadas",
    "garantias_valoradas",
    "detalle_recuperaciones",
    "resumen_recuperaciones",
    "recuperaciones_proyectadas",
    "recuperaciones_vp",
    "cobertura_recuperable",
    "bienes_pendientes",
    "revision_lgl",
    "resumen_final",
    "terrenos",
]

for variable in variables_eliminar:
    if variable in globals():
        del globals()[variable]

variables_eliminar = [
    "clasificacion_t12",
    "cohorte",
    "detalle",
    "evaluacion_mods",
    "fecha_t0",
    "fecha_t12",
    "inventario_t0",
    "pd_pc_historicos",
    "resumen",
    "seguimiento_t12",
    "variable",
    "resultados_historicos"
]

for nombre in variables_eliminar:
    globals().pop(nombre, None)

del variables_eliminar, nombre

# %%

""" Pérdida esperada """

"""
En esta parte ya no volvemos a estimar la PD, la PC ni las recuperaciones.
Esos cálculos los hicieron las secciones anteriores y están reunidos en
insumos_secciones_4_5. Lo que nos faltaba era construir la exposición que
corresponde a cada año y aplicar la fórmula de pérdida esperada del BUST.
Para 2026 hay que separar créditos normales y dudosos a diciembre de 2025;
para 2027 y 2028 se utiliza la EAD proyectada del año anterior.
"""

def construir_saldos_iniciales_pe(operaciones, tipos_cambio, fecha="2025-12-31"):
    """Obtiene CN, CD y número de operaciones por entidad y cartera BUST.

    Partimos de la misma función de clasificación que se usó para PD y PC.
    De esta manera el inventario excluye back-to-back, la categoría Otro y
    las cancelaciones que la propia función descarta. No creamos otra regla
    distinta solo para calcular las pérdidas.

    El resultado incluye tanto saldos como cantidades; las cantidades nos
    servirán más adelante para construir una medida relativa por operación.
    """

    # Esta función ya existía en el código de nuestros compañeros. Nos devuelve
    # una fila por operación vigente en el corte y una etiqueta estado_t0:
    # "Normal" o "Dudoso", según mora, cobro judicial y modificaciones.
    inventario = clasificar_inventario_inicial_crediticio(operaciones, fecha)

    # Copiamos solo los tipos de cambio del corte. pd.Timestamp convierte el
    # texto de la fecha en una fecha que pandas puede comparar correctamente.
    fx = tipos_cambio.loc[
        tipos_cambio["fecha"].eq(pd.Timestamp(fecha)),
        ["moneda", "crc_por_unidad"],
    ].copy()

    # validate evita que un tipo de cambio repetido multiplique accidentalmente
    # las filas de créditos cuando hagamos el merge. La relación es varios
    # créditos para una sola tasa de cambio por moneda.
    if fx.duplicated("moneda").any():
        raise ValueError("Hay más de un tipo de cambio por moneda en 2025.")

    inventario = inventario.merge(
        fx, on="moneda", how="left", validate="many_to_one"
    )
    if inventario["crc_por_unidad"].isna().any():
        raise ValueError("Hay créditos de 2025 sin tipo de cambio.")

    # saldo_principal_mo está en la moneda original. Multiplicarlo por la
    # cantidad de colones por unidad permite sumar MN y ME sin mezclar monedas.
    inventario["saldo_crc"] = (
        inventario["saldo_principal_mo"] * inventario["crc_por_unidad"]
    )
    if (inventario["saldo_crc"] < 0).any() or not np.isfinite(
        inventario["saldo_crc"].to_numpy(dtype=float)
    ).all():
        raise ValueError("El inventario inicial contiene saldos inválidos.")

    # Primero sumamos por estado. groupby agrupa las filas que comparten
    # entidad, cartera y estado; agg nos deja calcular varias estadísticas.
    # nunique cuenta operaciones distintas, no simplemente número de filas.
    resumen = (
        inventario.groupby(
            ["entidad", "categorias_bust", "estado_t0"], as_index=False
        )
        .agg(saldo_crc=("saldo_crc", "sum"), n_operaciones=("operacion_id", "nunique"))
    )

    # pivot_table convierte los valores "Normal" y "Dudoso" en columnas.
    # fill_value=0 es necesario cuando una cartera no tiene algún estado.
    saldos = resumen.pivot_table(
        index=["entidad", "categorias_bust"],
        columns="estado_t0",
        values="saldo_crc",
        aggfunc="sum",
        fill_value=0,
    ).rename(columns={"Normal": "CN_2025_crc", "Dudoso": "CD_2025_crc"})

    # Hacemos lo mismo con el número de operaciones. Reindex crea una columna
    # de ceros si el estado no existiera en toda la base; así no falla la suma.
    conteos = resumen.pivot_table(
        index=["entidad", "categorias_bust"],
        columns="estado_t0",
        values="n_operaciones",
        aggfunc="sum",
        fill_value=0,
    ).rename(columns={"Normal": "N_normales_2025", "Dudoso": "N_dudosas_2025"})

    saldos = saldos.reindex(
        columns=["CN_2025_crc", "CD_2025_crc"], fill_value=0
    )
    conteos = conteos.reindex(
        columns=["N_normales_2025", "N_dudosas_2025"], fill_value=0
    )
    saldos = saldos.join(conteos).reset_index()
    saldos["saldo_total_2025_crc"] = (
        saldos["CN_2025_crc"] + saldos["CD_2025_crc"]
    )
    saldos["N_total_2025"] = (
        saldos["N_normales_2025"] + saldos["N_dudosas_2025"]
    )

    # Confirmamos que no dejamos fuera alguna operación mientras agregábamos.
    if not np.isclose(saldos["saldo_total_2025_crc"].sum(),
                      inventario["saldo_crc"].sum(), rtol=0, atol=1):
        raise ValueError("Los saldos no concilian con el inventario inicial.")
    if saldos["N_total_2025"].sum() != inventario["operacion_id"].nunique():
        raise ValueError("Los conteos no concilian con el inventario inicial.")
    return saldos, inventario


def calcular_perdida_esperada(insumos, saldos_2025):
    """Aplica las fórmulas de la sección 4.2 del documento para 2026–2028.

    Todas las tasas PD, PC, LGL y LGD ya son proporciones entre 0 y 1.
    No hay que dividirlas otra vez entre 100. Los resultados monetarios se
    expresan en colones y se calculan por entidad, cartera, año y escenario.
    """

    claves = ["entidad", "categorias_bust"]
    claves_completas = claves + ["escenario", "ano"]
    datos = insumos.copy()

    # Con merge incorporamos CN y CD de 2025 a cada uno de los escenarios y
    # años. many_to_one exige un único inventario para cada entidad/cartera.
    datos = datos.merge(saldos_2025, on=claves, how="left",
                        validate="many_to_one")
    if datos.duplicated(claves_completas).any():
        raise ValueError("Hay combinaciones duplicadas de entidad, cartera, año y escenario.")
    if datos[["CN_2025_crc", "CD_2025_crc", "N_total_2025"]].isna().any().any():
        raise ValueError("Hay proyecciones sin saldos o conteos de diciembre 2025.")
    if set(datos["ano"]) != {2026, 2027, 2028}:
        raise ValueError("Las proyecciones deben cubrir 2026, 2027 y 2028.")

    # La conciliación compara el saldo de nuestro inventario con el usado para
    # calcular LGL. En la revisión previa las 21 carteras coincidieron con una
    # diferencia máxima minúscula, propia de los números de punto flotante.
    if not np.allclose(datos["saldo_total_2025_crc"],
                       datos["saldo_cartera_crc"], atol=1, rtol=0):
        raise ValueError("El inventario no coincide con el saldo de la LGL.")

    # Revisamos antes de calcular: ninguna tasa puede ser negativa ni superar
    # 100 %. También cuidamos que las exposiciones sean números finitos.
    for columna in ["PD", "PC", "LGL_cartera", "LGD"]:
        if not datos[columna].between(0, 1).all():
            raise ValueError(f"Hay valores fuera de [0,1] en {columna}.")
    for columna in ["saldo_proyectado_mn", "saldo_total_2025_crc"]:
        if (datos[columna] < 0).any() or not np.isfinite(datos[columna]).all():
            raise ValueError(f"La exposición {columna} tiene valores inválidos.")

    # Ordenar antes de shift es esencial: shift(1) toma la fila anterior del
    # mismo grupo. Por ejemplo, para 2027 necesitamos el saldo de 2026 del
    # MISMO escenario, entidad y cartera, nunca el de otra institución.
    datos = datos.sort_values(["escenario", "entidad", "categorias_bust", "ano"])
    grupo = datos.groupby(["escenario"] + claves, sort=False)
    datos["EAD_ano_anterior_crc"] = grupo["EAD_proyectada_mn"].shift(1)

    # En 2026 aún no existe una proyección del 2025 dentro de la tabla, así
    # que usamos la exposición observada del inventario inicial.
    es_2026 = datos["ano"].eq(2026)
    datos.loc[es_2026, "EAD_ano_anterior_crc"] = datos.loc[
        es_2026, "saldo_total_2025_crc"
    ]
    if datos["EAD_ano_anterior_crc"].isna().any():
        raise ValueError("Falta exposición del año anterior para alguna proyección.")

    # Primer año: una operación que era normal todavía puede deteriorarse,
    # por eso usamos PD * LGD * CN. Si ya era dudosa, el documento no vuelve
    # a multiplicarla por PD; únicamente LGD * CD.
    datos["EL_normal_crc"] = np.where(
        es_2026,
        datos["PD"] * datos["LGD"] * datos["CN_2025_crc"],
        0.0,
    )
    datos["EL_dudosa_crc"] = np.where(
        es_2026, datos["LGD"] * datos["CD_2025_crc"], 0.0
    )

    # Para 2027 y 2028 seguimos la fórmula literal de Santiago/BUST:
    # PD del año * LGD del año * EAD al cierre del año anterior. Esa fórmula
    # no vuelve a separar créditos dudosos; esto se documenta como limitación.
    datos["EL_futura_crc"] = np.where(
        ~es_2026,
        datos["PD"] * datos["LGD"] * datos["EAD_ano_anterior_crc"],
        0.0,
    )
    datos["EL_crc"] = (
        datos["EL_normal_crc"] + datos["EL_dudosa_crc"]
        + datos["EL_futura_crc"]
    )

    # El programa proyecta saldos, pero no el número futuro de operaciones.
    # Para la PE promedio por operación tomamos un supuesto muy explícito:
    # el tamaño promedio de un crédito se mantiene constante por segmento,
    # así que el número esperado crece en la misma proporción que el saldo.
    # No redondeamos porque es una estimación de cantidad, no un censo real.
    datos["N_proyectadas"] = np.where(
        datos["saldo_total_2025_crc"] > 0,
        datos["N_total_2025"] * datos["EAD_proyectada_mn"]
        / datos["saldo_total_2025_crc"],
        0.0,
    )
    datos["EL_por_operacion_crc"] = np.where(
        datos["N_proyectadas"] > 0,
        datos["EL_crc"] / datos["N_proyectadas"],
        np.nan,
    )
    datos["EL_sobre_EAD_pct"] = np.where(
        datos["EAD_ano_anterior_crc"] > 0,
        100 * datos["EL_crc"] / datos["EAD_ano_anterior_crc"],
        0.0,
    )

    # Como PD y LGD están entre 0 y 1, la pérdida no debería superar la
    # exposición usada en la fórmula ni resultar negativa.
    tolerancia = 1.0
    if (datos["EL_crc"] < -tolerancia).any() or (
        datos["EL_crc"] > datos["EAD_ano_anterior_crc"] + tolerancia
    ).any():
        raise ValueError("Hay pérdidas esperadas negativas o superiores a la EAD.")

    # Al terminar, dejamos las filas ordenadas para que los gráficos y tablas
    # sean fáciles de leer y no dependan del orden accidental de los merges.
    return datos.sort_values(claves_completas).reset_index(drop=True)


def incorporar_informacion_financiera(datos_el, financieros):
    """Añade patrimonio y estimaciones registradas del cierre de 2025.

    El patrimonio se mantiene fijo como referencia para los tres años:
    NO estamos proyectando patrimonio ni haciendo un modelo de solvencia.
    """
    f = financieros.loc[
        pd.to_datetime(financieros["fecha"]).eq(pd.Timestamp("2025-12-31")),
        ["entidad", "patrimonio_crc", "estimacion_registrada_crc",
         "cartera_principal_crc"],
    ].copy()
    if f["entidad"].duplicated().any():
        raise ValueError("La base financiera repite entidades para diciembre de 2025.")

    resultado = datos_el.merge(f, on="entidad", how="left", validate="many_to_one")
    if resultado[["patrimonio_crc", "estimacion_registrada_crc"]].isna().any().any():
        raise ValueError("Faltan datos financieros para alguna entidad.")
    if (resultado["patrimonio_crc"] <= 0).any():
        raise ValueError("Hay patrimonios no positivos; revisar EL/patrimonio.")

    # Dividir la EL entre patrimonio nos permite comparar instituciones de
    # tamaños distintos. A nivel de cartera es la pérdida de ESA cartera frente
    # al patrimonio completo de su entidad: no es un patrimonio por segmento.
    resultado["EL_sobre_patrimonio_pct"] = (
        100 * resultado["EL_crc"] / resultado["patrimonio_crc"]
    )
    return resultado


def calcular_detalle_operaciones_2026(inventario, datos_el):
    """Desagrega la pérdida esperada de 2026 a operaciones observadas en 2025.

    En 2026 esta desagregación es directa porque sí tenemos estados y saldos
    de cada operación en el inventario inicial. NO inventamos operaciones
    futuras para 2027 y 2028. La PD y LGD son las tasas del segmento: no son
    modelos individuales de comportamiento del deudor.
    """
    claves = ["entidad", "categorias_bust"]
    tasas = datos_el.loc[
        datos_el["ano"].eq(2026),
        claves + ["escenario", "PD", "LGD"],
    ].copy()

    # El cruce con los dos escenarios genera dos filas por operación inicial.
    detalle = inventario[
        claves + ["operacion_id", "estado_t0", "saldo_crc"]
    ].merge(tasas, on=claves, how="left", validate="many_to_many")
    if detalle[["PD", "LGD"]].isna().any().any():
        raise ValueError("Faltan tasas para algunas operaciones en 2026.")
    detalle["EL_operacion_crc"] = np.where(
        detalle["estado_t0"].eq("Normal"),
        detalle["saldo_crc"] * detalle["PD"] * detalle["LGD"],
        detalle["saldo_crc"] * detalle["LGD"],
    )

    # Verificamos que la suma del detalle produzca exactamente el cálculo
    # agregado del segmento. Si no, puede haber un merge duplicado o una tasa
    # asignada incorrectamente y no conviene seguir al dashboard.
    agrupado = detalle.groupby(claves + ["escenario"], as_index=False)[
        "EL_operacion_crc"
    ].sum()
    referencia = datos_el.loc[datos_el["ano"].eq(2026),
                              claves + ["escenario", "EL_crc"]]
    revision = referencia.merge(agrupado, on=claves + ["escenario"],
                                how="left", validate="one_to_one")
    if revision["EL_operacion_crc"].isna().any() or not np.allclose(
        revision["EL_crc"], revision["EL_operacion_crc"], atol=1, rtol=1e-12
    ):
        raise ValueError("El detalle de operaciones no concilia con EL 2026.")
    return detalle


# Estos objetos quedan disponibles para las siguientes secciones
# No imprimimos las miles de filas.
saldos_iniciales_2025, inventario_el_2025 = construir_saldos_iniciales_pe(
    df_operaciones_crediticias, df_tipo_cambio
)
resultados_perdida_esperada = calcular_perdida_esperada(
    insumos_secciones_4_5, saldos_iniciales_2025
)
resultados_perdida_esperada = incorporar_informacion_financiera(
    resultados_perdida_esperada, df_financieros_entidad
)
pe_operaciones_2026 = calcular_detalle_operaciones_2026(
    inventario_el_2025, resultados_perdida_esperada
)

# %%

""" Escenarios y sensibilidad """

"""
La comparación Base vs. Adverso se calcula a partir de las mismas pérdidas
que acabamos de obtener. No ajustamos ni recalibramos PD y PC aquí, porque
hacerlo ya sería cambiar los modelos de nuestros compañeros.
Para sensibilidad elegimos un solo supuesto: un RECORTE ADICIONAL a las
recuperaciones netas, por encima del ajuste macroeconómico ya incluido.
Es decir, recorte=0 reproduce exactamente el modelo recibido.
"""


def calcular_sensibilidad_recuperaciones(datos_el, recorte_pct=0):
    """Recalcula LGL, LGD y EL si las recuperaciones netas bajan cierto %.

    Ejemplo: 20 significa que multiplicamos las recuperaciones originales
    por 0.80. NO es una caída de 20 puntos porcentuales de LGL y NO volvemos
    a descontar por segunda vez los mismos gastos o la tasa de cura.
    """
    recorte_pct = float(recorte_pct)
    if not (0 <= recorte_pct <= 100):
        raise ValueError("El recorte de recuperación debe estar entre 0 y 100 %.")
    d = datos_el.copy()

    # En la sección 3 ya se limitó la recuperación al saldo de la cartera,
    # y esa es la cantidad que vamos a estresar. clip evita resultados fuera
    # del intervalo [0, saldo] si apareciera alguna diferencia numérica.
    recuperacion_base = d["recuperacion_cartera_crc"].clip(lower=0)
    if (recuperacion_base > d["saldo_cartera_crc"] + 1).any():
        raise ValueError("Las recuperaciones superan los saldos de cartera.")
    d["recuperacion_sensible_crc"] = recuperacion_base * (1 - recorte_pct / 100)

    # Calculamos de nuevo la proporción no recuperable. En una cartera sin
    # recuperaciones, la LGL se mantiene en 1: no inventamos cobros.
    d["LGL_sensible"] = np.where(
        d["saldo_cartera_crc"] > 0,
        1 - d["recuperacion_sensible_crc"] / d["saldo_cartera_crc"],
        1.0,
    )
    d["LGL_sensible"] = d["LGL_sensible"].clip(lower=0, upper=1)
    d["LGD_sensible"] = d["LGL_sensible"] * (1 - d["PC"])

    # Reutilizamos las dos fórmulas, sin modificar las exposiciones ni las PD.
    es_2026 = d["ano"].eq(2026)
    d["EL_sensible_crc"] = np.where(
        es_2026,
        d["LGD_sensible"] * (
            d["PD"] * d["CN_2025_crc"] + d["CD_2025_crc"]
        ),
        d["PD"] * d["LGD_sensible"] * d["EAD_ano_anterior_crc"],
    )
    d["incremento_EL_crc"] = d["EL_sensible_crc"] - d["EL_crc"]
    d["incremento_EL_pct"] = np.where(
        d["EL_crc"] > 0, 100 * d["incremento_EL_crc"] / d["EL_crc"], np.nan
    )

    # A recorte cero deben coincidir todas las LGD y EL calculadas antes.
    # También esperamos que recortar recuperaciones no reduzca las pérdidas.
    if recorte_pct == 0 and (
        not np.allclose(d["LGD_sensible"], d["LGD"], rtol=1e-10, atol=1e-12)
        or not np.allclose(d["EL_sensible_crc"], d["EL_crc"], rtol=1e-10, atol=1)
    ):
        raise ValueError("La sensibilidad 0 % no reproduce el modelo original.")
    if (d["incremento_EL_crc"] < -1).any() or (
        d["EL_sensible_crc"] > d["EAD_ano_anterior_crc"] + 1
    ).any():
        raise ValueError("La sensibilidad produjo una pérdida fuera de límites.")
    return d


def resumir_el(datos, columnas, columna_pe="EL_crc"):
    """Suma pérdidas y exposición sin promediar tasas simples por cartera.

    groupby separa los grupos pedidos y agg suma las magnitudes monetarias.
    Dividir los totales de EL/EAD produce un indicador ponderado por exposición;
    es más representativo que calcular un promedio simple de porcentajes.
    """
    if not columnas:
        raise ValueError("Es necesario indicar al menos una columna de agrupación.")
    resumen = datos.groupby(columnas, as_index=False).agg(
        EL_crc=(columna_pe, "sum"),
        EAD_base_crc=("EAD_ano_anterior_crc", "sum"),
        N_proyectadas=("N_proyectadas", "sum"),
    )
    resumen["EL_EAD_pct"] = np.where(
        resumen["EAD_base_crc"] > 0,
        100 * resumen["EL_crc"] / resumen["EAD_base_crc"],
        0.0,
    )
    return resumen


def comparar_escenarios_el(datos):
    """Compara las pérdidas de Base y Adverso, manteniendo el mismo segmento."""
    claves = ["ano", "entidad", "categorias_bust"]
    base = datos.loc[datos["escenario"].eq("Base"),
                     claves + ["EL_crc"]].rename(columns={"EL_crc": "EL_Base_crc"})
    adverso = datos.loc[datos["escenario"].eq("Adverso"),
                        claves + ["EL_crc"]].rename(columns={"EL_crc": "EL_Adverso_crc"})
    tabla = base.merge(adverso, on=claves, how="outer", validate="one_to_one")
    if tabla[["EL_Base_crc", "EL_Adverso_crc"]].isna().any().any():
        raise ValueError("Falta Base o Adverso para comparar algún segmento.")
    tabla["Diferencia_crc"] = tabla["EL_Adverso_crc"] - tabla["EL_Base_crc"]
    tabla["Diferencia_pct"] = np.where(
        tabla["EL_Base_crc"] > 0,
        100 * tabla["Diferencia_crc"] / tabla["EL_Base_crc"],
        np.nan,
    )
    # Un adverso menor que el base no se corrige ni se oculta: lo dejamos en
    # los resultados para revisarlo y justificarlo con las trayectorias macro.
    return tabla.sort_values(claves).reset_index(drop=True)


# Los cuadros se preparan aquí, antes de dibujar, para que cualquier persona
# pueda auditarlos también fuera del dashboard.
pe_por_entidad = resumir_el(
    resultados_perdida_esperada, ["escenario", "ano", "entidad"]
)
pe_por_cartera = resumir_el(
    resultados_perdida_esperada, ["escenario", "ano", "categorias_bust"]
)
pe_por_segmento = resumir_el(
    resultados_perdida_esperada,
    ["escenario", "ano", "entidad", "categorias_bust"],
)
pe_consolidada = resumir_el(
    resultados_perdida_esperada, ["escenario", "ano"]
)
comparacion_base_adverso = comparar_escenarios_el(resultados_perdida_esperada)
sensibilidad_sin_ajuste = calcular_sensibilidad_recuperaciones(
    resultados_perdida_esperada, recorte_pct=0
)

# Comprobación de suma: agregar las 21 carteras debe dar exactamente el mismo
# total del conjunto A-E para cada año y escenario.
revision_agregacion = pe_por_segmento.groupby(
    ["escenario", "ano"], as_index=False
)["EL_crc"].sum().merge(
    pe_consolidada[["escenario", "ano", "EL_crc"]],
    on=["escenario", "ano"], validate="one_to_one", suffixes=("_detalle", "_total")
)
if not np.allclose(
    revision_agregacion["EL_crc_detalle"], revision_agregacion["EL_crc_total"],
    rtol=1e-12, atol=1
):
    raise ValueError("La suma de las carteras no coincide con el conjunto A-E.")

# El cuadro macro resume cada año mediante la media de los cuatro trimestres.
# Lo utilizamos para explicar las trayectorias de los escenarios, pero sin
# afirmar que sea una relación causal demostrada para todas las variables.
macro_para_dashboard = (
    df_macro_proyectado.groupby(["escenario", "ano"], as_index=False)
    [["pib_interanual_pct", "desempleo_pct", "tasa_mn_pct", "tasa_me_pct",
      "tc_crc_usd", "indice_inmuebles", "indice_vehiculos", "indice_titulos"]]
    .mean()
)

# %%

""" Datos históricos y auditoría que alimentan el dashboard """

"""
El histórico que mostramos no es otra simulación. Sale de las observaciones
mensuales que ya estaban en la base de operaciones. Las filtramos con el
mismo universo BUST que usamos en los cálculos de la PE: dejamos fuera las
operaciones back-to-back y las etiquetadas como Otro.
"""


def preparar_cartera_historica_dashboard(operaciones, tipos_cambio):
    """Resume saldos y cantidades mensuales de 2019 a 2025 en colones.

    Mantenemos una fila por fecha, entidad y cartera BUST. El saldo es un
    inventario en un momento concreto; sumar distintos meses produciría una
    cifra engañosa, porque estaríamos contando varias veces el mismo crédito.
    """
    campos = ["fecha", "operacion_id", "entidad", "categorias_bust",
              "moneda", "back_to_back", "saldo_principal_mo"]
    datos = operaciones[campos].copy()
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    datos = datos.loc[
        datos["fecha"].between("2019-01-01", "2025-12-31")
        & datos["back_to_back"].eq(0)
        & datos["categorias_bust"].ne("Otro")
    ].copy()
    if datos.duplicated(["fecha", "operacion_id"]).any():
        raise ValueError("El histórico tiene operaciones repetidas en un mismo mes.")

    fx = tipos_cambio[["fecha", "moneda", "crc_por_unidad"]].copy()
    fx["fecha"] = pd.to_datetime(fx["fecha"])
    if fx.duplicated(["fecha", "moneda"]).any():
        raise ValueError("Se repiten fechas y monedas en la tabla de tipo de cambio.")

    # La unión por fecha Y moneda evita aplicar un tipo de cambio de otro mes.
    datos = datos.merge(fx, on=["fecha", "moneda"], how="left",
                        validate="many_to_one")
    if datos["crc_por_unidad"].isna().any():
        raise ValueError("Falta tipo de cambio en una observación histórica.")
    datos["saldo_crc"] = (datos["saldo_principal_mo"]
                          * datos["crc_por_unidad"])
    if not np.isfinite(datos["saldo_crc"]).all() or (datos["saldo_crc"] < 0).any():
        raise ValueError("Hay saldos históricos negativos o no numéricos.")

    # groupby agrupa las observaciones que comparten estas tres columnas.
    # agg permite pedir una suma de saldo y un conteo distinto de operaciones.
    mensual = datos.groupby(
        ["fecha", "entidad", "categorias_bust"], as_index=False
    ).agg(saldo_crc=("saldo_crc", "sum"),
          operaciones=("operacion_id", "nunique"))
    mensual["ano"] = mensual["fecha"].dt.year
    mensual = mensual.sort_values(["fecha", "entidad", "categorias_bust"])
    return mensual.reset_index(drop=True)


def resumir_pd_pc_historicas_dashboard(tabla):
    """Agrupa las tasas de cohorte por sus conteos, nunca por media simple.

    Por ejemplo, si una cartera tiene 3 incumplimientos entre 100 normales y
    otra tiene 1 entre 10, la tasa conjunta es 4/110, no el promedio de 3 y 10 %.
    Un denominador sin observaciones devuelve NaN, no una tasa observada 0 %.
    """
    t = tabla.copy()
    necesarios = ["cohorte", "entidad", "categorias_bust", "pd_A", "pd_B",
                  "pd_C", "pd_D", "pc_A", "pc_B", "pc_E"]
    faltantes = [c for c in necesarios if c not in t.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas históricas de PD/PC: {faltantes}")
    t["defectos"] = t[["pd_B", "pd_C", "pd_D"]].sum(axis=1)
    t["curas"] = t[["pc_B", "pc_E"]].sum(axis=1)
    return t[["cohorte", "entidad", "categorias_bust", "pd_A", "pc_A",
              "defectos", "curas"]].copy()


def construir_controles_dashboard(perdidas, saldos, inventario,
                                  operaciones_2026, revision):
    """Devuelve verificaciones trazables, junto con su magnitud observada.

    No pintamos un mensaje fijo de 'todo bien': cada estado se vuelve a
    calcular con las tablas del proyecto. Si algo cambia en el repositorio,
    el control reflejará el resultado y permitirá ver qué se alteró.
    """
    controles = []

    def agregar(nombre, correcto, detalle):
        controles.append({"Control": nombre,
                          "Estado": "Correcto" if correcto else "Revisar",
                          "Detalle": detalle})

    claves = ["entidad", "categorias_bust", "escenario", "ano"]
    duplicados = int(perdidas.duplicated(claves).sum())
    agregar("Unicidad entidad-cartera-escenario-año", duplicados == 0,
            f"{duplicados} combinaciones duplicadas; {len(perdidas)} registros")

    dif_inventario = abs(saldos["saldo_total_2025_crc"].sum()
                         - inventario["saldo_crc"].sum())
    agregar("Inventario inicial CN + CD", dif_inventario <= 1,
            f"Diferencia de {dif_inventario:.6f} CRC")

    por_segmento = perdidas.loc[perdidas["ano"].eq(2026),
        ["entidad", "categorias_bust", "escenario", "EL_crc"]]
    detalle = operaciones_2026.groupby(
        ["entidad", "categorias_bust", "escenario"], as_index=False
    )["EL_operacion_crc"].sum()
    conc = por_segmento.merge(detalle, on=["entidad", "categorias_bust",
                                            "escenario"], how="outer",
                              validate="one_to_one")
    dif_2026 = (conc["EL_crc"] - conc["EL_operacion_crc"]).abs()
    mayor_2026 = float(dif_2026.max()) if dif_2026.notna().all() else np.inf
    agregar("Detalle de operaciones de 2026", mayor_2026 <= 1,
            f"Mayor diferencia: {mayor_2026:.6f} CRC")

    desvio = (revision["EL_crc_detalle"] - revision["EL_crc_total"]).abs()
    agregar("Consolidación A-E por escenario y año", desvio.max() <= 1,
            f"Diferencia máxima: {desvio.max():.6f} CRC")

    fuera = (perdidas["EL_crc"] < -1) | (
        perdidas["EL_crc"] > perdidas["EAD_ano_anterior_crc"] + 1)
    agregar("Límite 0 ≤ PE ≤ exposición", not fuera.any(),
            f"{int(fuera.sum())} resultados fuera de rango")

    tasas = perdidas[["PD", "PC", "LGL_cartera", "LGD"]]
    invalidos = int((~tasas.ge(0).all(axis=1) | ~tasas.le(1).all(axis=1)
                    | tasas.isna().any(axis=1)).sum())
    agregar("Tasas entre 0 % y 100 %", invalidos == 0,
            f"{invalidos} filas con tasas inválidas")

    cero = calcular_sensibilidad_recuperaciones(perdidas, 0)
    mayor_cero = (cero["EL_sensible_crc"] - perdidas["EL_crc"]).abs().max()
    agregar("Sensibilidad del 0 % = modelo original", mayor_cero <= 1,
            f"Diferencia máxima: {mayor_cero:.6f} CRC")
    return pd.DataFrame(controles)


# Se calculan una única vez al cargar app.py. Mover filtros no vuelve a leer
# cientos de miles de observaciones: Shiny solo filtra estas tablas resumidas.
cartera_historica_dashboard = preparar_cartera_historica_dashboard(
    df_operaciones_crediticias, df_tipo_cambio
)
tasas_historicas_dashboard = resumir_pd_pc_historicas_dashboard(pd_pc_historicas)
controles_dashboard = construir_controles_dashboard(
    resultados_perdida_esperada, saldos_iniciales_2025,
    inventario_el_2025, pe_operaciones_2026, revision_agregacion
)

# %%

""" Dashboard interactivo con Shiny para Python """

"""
Usamos Shiny porque permite trabajar con filtros y resultados reactivos,
bastante parecido a Shiny de R. La aplicación se ejecuta, desde la carpeta
del repositorio, con:
.\.venv\Scripts\shiny.exe run --reload --launch-browser app.py
Para instalar las librerías adicionales una sola vez en esa misma carpeta:
.\.venv\Scripts\python.exe -m pip install shiny matplotlib
Si ejecutamos python app.py sin Shiny instalado, las secciones 4 y 5 igual
se calculan y se pueden consultar; únicamente no se abre la interfaz.
"""

def formatear_crc(valor):
    """Da un formato corto a los montos: mil millones o millones de colones."""
    if not np.isfinite(valor):
        return "No disponible"
    if abs(valor) >= 1e9:
        return f"₡{valor / 1e9:,.2f} mil millones"
    if abs(valor) >= 1e6:
        return f"₡{valor / 1e6:,.2f} millones"
    return f"₡{valor:,.0f}"


def filtrar_resultados(datos, entidad="Todas", cartera="Todas",
                      ano="2026", escenario="Ambos"):
    """Filtra sin modificar la tabla original; Todos/Ambos incluyen todo."""
    d = datos
    if entidad != "Todas":
        d = d.loc[d["entidad"].eq(entidad)]
    if cartera != "Todas":
        d = d.loc[d["categorias_bust"].eq(cartera)]
    if ano != "Todos":
        d = d.loc[d["ano"].eq(int(ano))]
    if escenario != "Ambos":
        d = d.loc[d["escenario"].eq(escenario)]
    return d.copy()


def obtener_indicadores(datos):
    """Calcula indicadores agregados sin sumar dos veces patrimonios."""
    pe = datos["EL_crc"].sum()
    ead = datos["EAD_ano_anterior_crc"].sum()
    n = datos["N_proyectadas"].sum()

    # Una entidad puede aparecer en muchas carteras y años. Por eso usamos
    # drop_duplicates antes de sumar el patrimonio de referencia 2025.
    patrimonio = datos[["entidad", "patrimonio_crc"]].drop_duplicates(
        "entidad"
    )["patrimonio_crc"].sum()
    return {
        "pe": pe,
        "ead": ead,
        "pe_ead": 100 * pe / ead if ead > 0 else np.nan,
        "patrimonio": patrimonio,
        "pe_patrimonio": 100 * pe / patrimonio if patrimonio > 0 else np.nan,
        "n": n,
        "pe_por_operacion": pe / n if n > 0 else np.nan,
    }


# Para evitar instalar Shiny solamente para consultar EL en Python, creamos
# la interfaz si la biblioteca está instalada. En VS Code instalaremos Shiny
# y entonces este bloque quedará activo sin cambiar el archivo.
if importlib.util.find_spec("shiny") is not None:
    from shiny import App, ui, render, reactive
    import matplotlib.pyplot as plt

    # Dos colores consistentes: azul para Base y naranja para Adverso.
    # Las figuras utilizan millones de colones para no llenar los ejes de ceros.
    COLORES = {"Base": "#2765b0", "Adverso": "#db7844"}
    COLOR_TEXTO = "#23364a"
    carteras_dashboard = sorted(str(x) for x in resultados_perdida_esperada["categorias_bust"].unique())
    entidades_dashboard = sorted(str(x) for x in resultados_perdida_esperada["entidad"].unique())

    # CSS es solo presentación: mejora la legibilidad y no interviene en la
    # lógica de las pérdidas ni modifica ninguna de las bases de datos.
    estilos = ui.tags.style("""
        body { background: #f5f7fb; color: #23364a; }
        .bslib-card { border: 1px solid #e2e8f2; border-radius: 12px; }
        .card-header { font-weight: 650; }
        .bslib-value-box { border-radius: 12px; }
        .explicacion { color: #506278; font-size: .89rem; line-height: 1.55; }
        .titulo-seccion { font-weight: 700; margin-top: 12px; }
        .pie-grafico { color: #718096; font-size: .8rem; }
    """)

    # Separamos los controles (izquierda) de los resultados. Esto permite
    # cambiar escenario, año o segmento sin volver a ejecutar todos los modelos
    # históricos de crédito.
    app_ui = ui.page_sidebar(
        ui.sidebar(
            ui.h4("Filtros del análisis"),
            ui.input_select("f_ano", "Año de proyección",
                            choices=["2026", "2027", "2028", "Todos"],
                            selected="2026"),
            ui.input_select("f_escenario", "Escenarios disponibles",
                            choices=["Ambos", "Base", "Adverso"],
                            selected="Base"),
            ui.input_select("f_entidad", "Entidad financiera",
                            choices=["Todas"] + entidades_dashboard,
                            selected="Todas"),
            ui.input_selectize("f_cartera", "Cartera BUST",
                               choices=["Todas"] + carteras_dashboard,
                               selected="Todas"),
            ui.hr(),
            ui.h5("Sensibilidad"),
            ui.input_slider("f_recorte", "Recorte ADICIONAL a recuperaciones",
                            min=0, max=40, value=0, step=5, post=" %"),
            ui.p("El 0 % deja el modelo original sin cambios. Este control "
                 "modifica el análisis de sensibilidad, no los resultados base.",
                 class_="explicacion"),
            ui.hr(),
            ui.p("Datos sintéticos. Corte: diciembre de 2025. "
                 "Proyecciones: 2026–2028.", class_="explicacion"),
            width=310,
        ),
        estilos,
        ui.div(
            ui.h2("Prueba de tensión | Riesgo de crédito"),
            ui.p("Pérdidas esperadas por entidad y cartera BUST, escenarios "
                 "macroeconómicos y sensibilidad de recuperaciones.",
                 class_="explicacion"),
        ),
        ui.navset_tab(
            ui.nav_panel(
                "Resumen",
                ui.layout_columns(
                    ui.value_box("Pérdida esperada", ui.output_text("kpi_pe"),
                                 theme="bg-blue", fill=False),
                    ui.value_box("PE / exposición utilizada", ui.output_text("kpi_ratio"),
                                 theme="bg-teal", fill=False),
                    ui.value_box("PE / patrimonio 2025", ui.output_text("kpi_patrimonio"),
                                 theme="bg-orange", fill=False),
                    col_widths=[4, 4, 4],
                ),
                ui.p("Con 'Ambos' o 'Todos' se suman pérdidas de varios escenarios "
                     "o años. Para interpretar un único año/escenario, filtralo "
                     "en la barra lateral. La exposición usada en la razón EL/EAD "
                     "es la del inicio de cada horizonte.", class_="explicacion"),
                ui.layout_columns(
                    ui.card(ui.card_header("Base frente a Adverso"),
                            ui.output_plot("grafico_escenarios", height="340px"),
                            ui.p("Compara siempre los dos escenarios aunque el filtro "
                                 "principal muestre solo uno.", class_="pie-grafico")),
                    ui.card(ui.card_header("Carteras que más contribuyen"),
                            ui.output_plot("grafico_carteras", height="340px")),
                    col_widths=[6, 6],
                ),
                ui.card(ui.card_header("Resumen consolidado por año y escenario"),
                        ui.output_data_frame("tabla_consolidado")),
            ),
            ui.nav_panel(
                "Cartera histórica",
                ui.p("Los saldos son inventarios mensuales observados entre "
                     "2019 y 2025, en millones de colones. Los filtros de "
                     "entidad y cartera se aplican aquí; los escenarios son "
                     "solo para proyecciones.", class_="explicacion"),
                ui.layout_columns(
                    ui.card(ui.card_header("Exposición mensual 2019–2025"),
                            ui.output_plot("grafico_historico_saldo", height="345px")),
                    ui.card(ui.card_header("Composición al cierre histórico"),
                            ui.input_select("f_ano_hist", "Año de cierre observado",
                                            choices=[str(y) for y in range(2019, 2026)],
                                            selected="2025"),
                            ui.output_plot("grafico_historico_composicion",
                                           height="305px")),
                    col_widths=[6, 6],
                ),
                ui.card(ui.card_header("Saldos y operaciones del cierre histórico seleccionado"),
                        ui.output_data_frame("tabla_historico_cartera")),
                ui.card(ui.card_header("PD y PC históricas: cohortes de 12 meses"),
                        ui.output_plot("grafico_tasas_historicas", height="325px"),
                        ui.output_data_frame("tabla_tasas_historicas"),
                        ui.p("La cohorte 2024 se observa hasta diciembre de 2025. "
                             "Se agrupan los numeradores y denominadores reales; "
                             "una tasa sin denominador aparece vacía.",
                             class_="pie-grafico")),
            ),
            ui.nav_panel(
                "Indicadores de riesgo",
                ui.p("PD, PC y LGD se muestran como promedios ponderados por "
                     "exposición. El gráfico de tasas compara siempre Base y "
                     "Adverso para los años seleccionados, aunque el filtro "
                     "lateral esté en un único escenario. El cociente PE/EAD "
                     "se obtiene de los importes totales.",
                     class_="explicacion"),
                ui.layout_columns(
                    ui.card(ui.card_header("Comparación de tasas: Base frente a Adverso"),
                            ui.input_select("f_metrica", "Tasa para comparar",
                                            choices={"PD": "PD - incumplimiento",
                                                     "PC": "PC - cura",
                                                     "LGD": "LGD - severidad",
                                                     "LGL_cartera": "LGL - pérdida sin cura"},
                                            selected="LGD"),
                            ui.output_plot("grafico_tasas_proyectadas", height="340px"),
                            ui.p("Cada barra corresponde a un escenario y año. La longitud "
                                 "indica la tasa seleccionada, en porcentaje.",
                                 class_="pie-grafico")),
                    ui.card(ui.card_header("Exposición frente a pérdida esperada"),
                            ui.output_plot("grafico_ead_el", height="340px")),
                    col_widths=[6, 6],
                ),
                ui.card(ui.card_header("Indicadores agregados por año y escenario"),
                        ui.output_data_frame("tabla_indicadores_agrupados")),
                ui.card(ui.card_header("Detalle técnico: PD, PC, LGL, LGD, EAD y PE"),
                        ui.output_data_frame("tabla_indicadores_detalle"),
                        ui.p("Para 2026, PE distingue el saldo normal y el dudoso. "
                             "Para 2027–2028, la EAD es la exposición del año previo. "
                             "El detalle conserva las cifras que justifican "
                             "cada resultado.", class_="pie-grafico")),
            ),
            ui.nav_panel(
                "Entidades y carteras",
                ui.layout_columns(
                    ui.card(ui.card_header("Pérdida esperada por entidad"),
                            ui.output_plot("grafico_entidades", height="330px")),
                    ui.card(ui.card_header("Contribución por cartera BUST"),
                            ui.output_plot("grafico_participaciones", height="330px")),
                    col_widths=[6, 6],
                ),
                ui.card(ui.card_header("Detalle por entidad y cartera"),
                        ui.output_data_frame("tabla_detalle"),
                        ui.p("Los montos se muestran en millones de CRC. La razón "
                             "PE/patrimonio compara la PE del segmento con el "
                             "patrimonio completo de su entidad en 2025.",
                             class_="pie-grafico")),
                ui.card(ui.card_header("Diez operaciones con mayor EL en 2026"),
                        ui.output_data_frame("tabla_operaciones"),
                        ui.p("Solo 2026: el inventario de créditos individuales "
                             "existe al corte 2025. No se inventan IDs para 2027–2028. "
                             "El filtro de año no cambia esta tabla.", class_="pie-grafico")),
            ),
            ui.nav_panel(
    "Escenarios",
    ui.card(
        ui.card_header("Lectura de resultados y señales atípicas"),
        ui.div(
    ui.output_text("texto_interpretacion"),
    style="white-space: pre-wrap; font-family: inherit;"
),
        ui.p(
            "Esta lectura resume relaciones del escenario y no "
            "prueba causalidad estadística.",
            class_="explicacion"
        ),
    ),
    ui.card(
        ui.card_header("Diferencia entre Adverso y Base"),
        ui.output_data_frame("tabla_comparacion")
    ),
 
                ui.card(ui.card_header("Trayectorias macroeconómicas suministradas"),
                        ui.input_select(
                            "f_macro", "Indicador macroeconómico",
                            choices={
                                "tc_crc_usd": "Tipo de cambio CRC/USD",
                                "tasa_mn_pct": "Tasa en moneda nacional (%)",
                                "desempleo_pct": "Desempleo (%)",
                                "pib_interanual_pct": "PIB interanual (%)",
                                "indice_inmuebles": "Índice de inmuebles",
                                "indice_vehiculos": "Índice de vehículos",
                                "indice_titulos": "Índice de títulos",
                            }, selected="tc_crc_usd"
                        ),
                        ui.output_plot("grafico_macro", height="330px"),
                        ui.p("Las proyecciones de PD usan el tipo de cambio, "
                             "las de PC usan la tasa en MN y la LGD utiliza "
                             "índices de colaterales. PIB y tasas también "
                             "influyen en el crecimiento proyectado de cartera; "
                             "no son coeficientes individuales de PD/PC.",
                             class_="explicacion")),
            ),
            ui.nav_panel(
                "Sensibilidad",
                ui.p("Se reduce la recuperación neta que ya tenía incluido "
                     "el modelo del escenario. No cambiamos PD, PC, exposición "
                     "ni tasas de descuento: aislamos solamente este supuesto.",
                     class_="explicacion"),
                ui.layout_columns(
                    ui.value_box("PE con recorte seleccionado",
                                 ui.output_text("kpi_pe_sensible"),
                                 theme="bg-blue", fill=False),
                    ui.value_box("Cambio respecto al modelo original",
                                 ui.output_text("kpi_incremento"),
                                 theme="bg-orange", fill=False),
                    col_widths=[6, 6],
                ),
                ui.card(ui.card_header("Curva de sensibilidad (0 % a 40 %)"),
                        ui.output_plot("grafico_sensibilidad", height="355px")),
                ui.card(ui.card_header("Detalle del supuesto seleccionado"),
                        ui.output_data_frame("tabla_sensibilidad")),
            ),
            ui.nav_panel(
                "Descargas",
                ui.p("Podés exportar los resultados calculados a CSV para "
                     "revisarlos en Excel o utilizarlos en el informe. Los "
                     "archivos respetan los filtros que corresponden a cada tabla.",
                     class_="explicacion"),
                ui.layout_columns(
                    ui.card(ui.card_header("Resultados proyectados"),
                            ui.download_button("descargar_proyecciones",
                                               "Descargar PE, PD, PC, LGD y EAD")),
                    ui.card(ui.card_header("Resumen por segmento"),
                            ui.download_button("descargar_resumen",
                                               "Descargar resumen filtrado")),
                    col_widths=[6, 6],
                ),
                ui.layout_columns(
                    ui.card(ui.card_header("Histórico mensual de cartera"),
                            ui.download_button("descargar_historico",
                                               "Descargar saldos 2019–2025")),
                    ui.card(ui.card_header("PD y PC históricas"),
                            ui.download_button("descargar_tasas_historicas",
                                               "Descargar cohortes históricas")),
                    col_widths=[6, 6],
                ),
                ui.layout_columns(
                    ui.card(ui.card_header("Sensibilidad"),
                            ui.download_button("descargar_sensibilidad",
                                               "Descargar sensibilidad seleccionada")),
                    ui.card(ui.card_header("Operaciones de 2026"),
                            ui.download_button("descargar_operaciones",
                                               "Descargar detalle individual")),
                    col_widths=[6, 6],
                ),
            ),
            ui.nav_panel(
                "Metodología",
                ui.markdown("""
### Qué está calculando esta aplicación

- **2026:** `PE = PD × LGD × CN_2025 + LGD × CD_2025`.
  Los créditos ya dudosos al cierre de 2025 no se multiplican nuevamente por PD.
- **2027 y 2028:** `PE_t = PD_t × LGD_t × EAD_(t−1)`.
  Se usan las exposiciones proyectadas a diciembre del año previo.
- **LGD:** `LGL × (1 − PC)`, tal como quedó implementado en la sección 3.
- **LGL:** `1 − (recuperaciones netas descontadas / saldo de cartera)`.
  Los valores de garantía y las recuperaciones ya incorporan los supuestos
  macroeconómicos de cada escenario, gastos de venta y plazos de realización.
- **Sensibilidad:** `Recuperación ajustada = Recuperación original ×
  (1 − recorte/100)`. El control 0 % reproduce la PE original.

### Supuestos que debemos documentar

1. Los saldos de 2025 se separan con la clasificación BUST ya programada.
   Se excluyen back-to-back, categoría `Otro` y cancelaciones elegibles.
2. Todos los importes se convierten a colones al tipo de cambio del corte.
3. La **LGL es por segmento**, porque se supone que las garantías del conjunto
   de créditos son representativas de las garantías de los incumplidos.
4. En operaciones sin recuperación por garantías, la porción no recuperable se
   considera pérdida total. No se estiman cobros extrajudiciales adicionales.
5. El **patrimonio de 2025 permanece fijo** como base comparativa. No proyectamos
   patrimonio ni calculamos solvencia regulatoria.
6. Para informar PE por operación de años proyectados se aproxima la cantidad
   futura de créditos suponiendo **saldo promedio constante por segmento**.
   Los conteos proyectados son valores esperados, no operaciones observadas.
7. Las fórmulas de 2027–2028 no separan explícitamente la EAD dudosa; se sigue
   el tratamiento del documento del curso, no una transición de estados nueva.
8. No se suman los importes de bienes adjudicados pendientes por separado a
   la PE, pues la sección de recuperaciones ya los trata de forma diferenciada.
9. Una variación de sensibilidad es **un estrés adicional** sobre recuperaciones
   netas existentes; no sustituye el escenario adverso ni vuelve a cobrar gastos.
10. La PE es una estimación del ejercicio sintético, no una pérdida realizada ni
    una provisión contable que deba ser idéntica a la estimación registrada.

### Fuentes

- Reglamento de cálculo de carteras BUST.
- Bases sintéticas y funciones desarrolladas en `app.py`.

### Lectura de las tablas históricas y los controles

- **Histórico de cartera (2019–2025):** saldos mensuales por entidad y BUST;
  se convierten a colones con el tipo de cambio del mes. Son cortes separados:
  no se suman los inventarios de distintos meses para obtener una cartera total.
- **PD y PC históricas:** proporción de eventos de las cohortes iniciales
  seguidas por 12 meses; se agregan eventos y exposiciones en número de
  operaciones, no se promedian las tasas de cada segmento.
- **PD, PC y LGD proyectadas agregadas:** promedios ponderados por la exposición
  usada en el cálculo. Son indicadores descriptivos, no sustituyen la fórmula
  especial de pérdida esperada de 2026.
- **Celdas sin datos:** cuando no existe denominador para estimar una tasa,
  dejamos el valor vacío, en vez de inventar cero. Las tablas de detalle permiten identificar esos casos.
- **Exportación:** archivos CSV con resultados generados a partir de las bases
  y filtros de la aplicación. Ninguna tabla se llena manualmente.
                """),
            ),
            id="pestanas",
        ),
        title="Mini proyecto de crédito | Riesgos en entidades financieras I",
        fillable=False,
    )

    def server(input, output, session):
        """Conecta los controles del usuario con datos, indicadores y gráficos."""

        @reactive.calc
        def datos_filtrados():
            # Shiny recuerda qué inputs leímos. Si cambia uno, solamente
            # recalcula las tablas y figuras que dependen de estos filtros.
            return filtrar_resultados(
                resultados_perdida_esperada,
                entidad=input.f_entidad(), cartera=input.f_cartera(),
                ano=input.f_ano(), escenario=input.f_escenario(),
            )

        @reactive.calc
        def datos_ambos_escenarios():
            # Para comparar Base y Adverso ignoramos solo el filtro escenario.
            # Se conservan entidad, cartera y año para una comparación justa.
            return filtrar_resultados(
                resultados_perdida_esperada,
                entidad=input.f_entidad(), cartera=input.f_cartera(),
                ano=input.f_ano(), escenario="Ambos",
            )

        @reactive.calc
        def datos_sensibles():
            # Este cálculo es liviano porque usa los 126 registros ya estimados.
            return calcular_sensibilidad_recuperaciones(
                datos_filtrados(), recorte_pct=input.f_recorte()
            )

        @reactive.calc
        def historico_filtrado():
            # Las cifras observadas no cambian por el escenario macro futuro.
            h = cartera_historica_dashboard
            if input.f_entidad() != "Todas":
                h = h.loc[h["entidad"].eq(input.f_entidad())]
            if input.f_cartera() != "Todas":
                h = h.loc[h["categorias_bust"].eq(input.f_cartera())]
            return h.copy()

        @reactive.calc
        def historico_tasas_filtrado():
            # Cada fila mantiene los conteos reales de una cohorte por segmento.
            d = tasas_historicas_dashboard
            if input.f_entidad() != "Todas":
                d = d.loc[d["entidad"].eq(input.f_entidad())]
            if input.f_cartera() != "Todas":
                d = d.loc[d["categorias_bust"].eq(input.f_cartera())]
            return d.copy()

        @reactive.calc
        def tasas_por_cohorte():
            # Recalcular los cocientes tras SUMAR numeradores y denominadores,
            # en vez de promediar porcentajes con tamaños de muestra distintos.
            d = historico_tasas_filtrado().groupby("cohorte", as_index=False).agg(
                normales=("pd_A", "sum"),
                dudosas=("pc_A", "sum"),
                incumplimientos=("defectos", "sum"),
                curas=("curas", "sum"),
            )
            d["Cierre"] = d["cohorte"] + 1
            d["PD (%)"] = np.where(d["normales"] > 0,
                                    100 * d["incumplimientos"] / d["normales"],
                                    np.nan)
            d["PC (%)"] = np.where(d["dudosas"] > 0,
                                    100 * d["curas"] / d["dudosas"], np.nan)
            return d

        @reactive.calc
        def indicadores_ponderados():
            # Las tasas de diferentes segmentos no se pueden sumar. Usamos
            # exposiciones como pesos, con una salida separada por año y escenario.
            d = datos_filtrados()
            filas = []
            for (ano, escenario), g in d.groupby(["ano", "escenario"]):
                w = g["EAD_ano_anterior_crc"].to_numpy(dtype=float)
                wtotal = w.sum()
                fila = {"Año": ano, "Escenario": escenario,
                        "EAD (millones CRC)": wtotal / 1e6,
                        "PE (millones CRC)": g["EL_crc"].sum() / 1e6,
                        "PE/EAD (%)": (100 * g["EL_crc"].sum() / wtotal
                                       if wtotal > 0 else np.nan)}
                for columna in ["PD", "PC", "LGL_cartera", "LGD"]:
                    fila[columna] = (100 * np.average(g[columna], weights=w)
                                     if wtotal > 0 else np.nan)
                filas.append(fila)
            # Con una selección sin observaciones conservamos todas las
            # columnas para que los gráficos y DataGrids no den KeyError.
            columnas = ["Año", "Escenario", "EAD (millones CRC)",
                        "PE (millones CRC)", "PE/EAD (%)", "PD", "PC",
                        "LGL_cartera", "LGD"]
            return pd.DataFrame(filas, columns=columnas)

        def descargar_csv(tabla):
            # Con punto y coma evitamos chocar con la coma decimal que utiliza
            # normalmente Excel en español. utf-8-sig agrega una marca BOM,
            # útil para que Excel reconozca correctamente las tildes y la ñ.
            return tabla.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")

        @render.text
        def kpi_pe():
            return formatear_crc(obtener_indicadores(datos_filtrados())["pe"])

        @render.text
        def kpi_ratio():
            valor = obtener_indicadores(datos_filtrados())["pe_ead"]
            return f"{valor:.2f} %" if np.isfinite(valor) else "Sin datos"

        @render.text
        def kpi_patrimonio():
            valor = obtener_indicadores(datos_filtrados())["pe_patrimonio"]
            return f"{valor:.2f} %" if np.isfinite(valor) else "Sin datos"

        @render.text
        def kpi_pe_sensible():
            return formatear_crc(datos_sensibles()["EL_sensible_crc"].sum())

        @render.text
        def kpi_incremento():
            d = datos_sensibles()
            base = d["EL_crc"].sum()
            aumento = d["incremento_EL_crc"].sum()
            porcentaje = 100 * aumento / base if base > 0 else 0
            return f"{formatear_crc(aumento)} (+{porcentaje:.2f} %)"

        @render.plot
        def grafico_escenarios():
            d = resumir_el(datos_ambos_escenarios(), ["ano", "escenario"])
            tabla = d.pivot(index="ano", columns="escenario", values="EL_crc")
            fig, ax = plt.subplots(figsize=(8.0, 4.0))
            anos = tabla.index.to_numpy()
            x = np.arange(len(anos))
            for desplazamiento, escenario in [(-0.19, "Base"), (0.19, "Adverso")]:
                valores = tabla.get(escenario, pd.Series(0, index=tabla.index))
                ax.bar(x + desplazamiento, valores.to_numpy() / 1e6,
                       width=0.36, label=escenario, color=COLORES[escenario])
            ax.set_xticks(x, anos)
            ax.set_ylabel("Pérdida esperada (millones de ₡)")
            ax.legend(frameon=False)
            ax.grid(axis="y", alpha=0.17)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            return fig

        @render.plot
        def grafico_carteras():
            d = resumir_el(datos_filtrados(), ["categorias_bust"])
            d = d.sort_values("EL_crc", ascending=True)
            fig, ax = plt.subplots(figsize=(8.3, 4.3))
            ax.barh(d["categorias_bust"], d["EL_crc"] / 1e6,
                    color="#4379b7")
            ax.set_xlabel("Pérdida esperada (millones de ₡)")
            ax.grid(axis="x", alpha=0.17)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", labelsize=8)
            fig.tight_layout()
            return fig

        @render.plot
        def grafico_entidades():
            d = resumir_el(datos_filtrados(), ["entidad"])
            d = d.sort_values("entidad")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(d["entidad"], d["EL_crc"] / 1e6, color="#467bb6")
            ax.set_xlabel("Entidad")
            ax.set_ylabel("Pérdida esperada (millones de ₡)")
            ax.grid(axis="y", alpha=0.17)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            return fig

        @render.plot
        def grafico_participaciones():
            d = resumir_el(datos_filtrados(), ["categorias_bust"])
            d = d.sort_values("EL_crc", ascending=False)
            total = d["EL_crc"].sum()
            porcentajes = 100 * d["EL_crc"] / total if total > 0 else d["EL_crc"] * 0
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(d["categorias_bust"], porcentajes, color="#5e8eb8")
            ax.invert_yaxis()
            ax.set_xlabel("Participación en EL (%)")
            ax.set_xlim(left=0)
            ax.grid(axis="x", alpha=0.17)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", labelsize=8)
            fig.tight_layout()
            return fig

        @render.text
        def texto_interpretacion():
            # No basta con decir "adverso es peor". Aquí comparamos montos
            # agregados de los mismos segmentos y comentamos los casos donde
            # el resultado no sigue esa expectativa (sin modificar los datos).
            d = datos_ambos_escenarios()
            if d.empty:
                return "No hay operaciones para esta combinación de filtros."
            anuales = resumir_el(d, ["ano", "escenario"])
            texto = []
            for ano, g in anuales.groupby("ano"):
                valores = g.set_index("escenario")["EL_crc"]
                if not {"Base", "Adverso"}.issubset(valores.index):
                    continue
                base = float(valores["Base"])
                adverso = float(valores["Adverso"])
                dif = 100 * (adverso / base - 1) if base > 0 else np.nan
                texto.append(
                    f"{ano}: EL Base {formatear_crc(base)}; EL Adverso "
                    f"{formatear_crc(adverso)}; variación {dif:+.2f} %."
                )
            dif_segmentos = comparar_escenarios_el(d)
            excepciones = dif_segmentos.loc[dif_segmentos["Diferencia_crc"] < -1]
            if len(excepciones):
                texto.append(
                    f"Atención: {len(excepciones)} segmento(s) tienen menor PE "
                    "en Adverso que en Base; revisar exposición proyectada, "
                    "PD y LGD antes de interpretarlos."
                )
            else:
                texto.append(
                    "En el universo seleccionado, el escenario adverso no "
                    "produce una PE menor que el base en ningún segmento."
                )
            texto.append(
                "Mecanismos del modelo: tipo de cambio → PD; tasa en MN → "
                "PC; índices de activos → recuperaciones/LGL; PIB y tasas → "
                "crecimiento de saldos. No hay coeficientes directos de "
                "desempleo o inflación en PD y PC."
            )
            return "\n".join(texto)

        @render.plot
        def grafico_macro():
            variable = input.f_macro()
            d = macro_para_dashboard
            fig, ax = plt.subplots(figsize=(9, 3.8))
            for escenario in ["Base", "Adverso"]:
                s = d.loc[d["escenario"].eq(escenario)].sort_values("ano")
                ax.plot(s["ano"], s[variable], marker="o", linewidth=2.5,
                        color=COLORES[escenario], label=escenario)
            ax.set_xticks([2026, 2027, 2028])
            ax.set_xlabel("Año")
            ax.set_ylabel(variable.replace("_", " "))
            ax.grid(alpha=0.18)
            ax.legend(frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            return fig

        @render.plot
        def grafico_sensibilidad():
            # Armamos una pequeña curva variando el supuesto desde 0 a 40%.
            # Se evalúa el mismo universo que seleccionó la persona.
            d = datos_filtrados()
            fig, ax = plt.subplots(figsize=(9, 4))
            for escenario, grupo in d.groupby("escenario"):
                x, y = [], []
                for recorte in range(0, 41, 5):
                    s = calcular_sensibilidad_recuperaciones(grupo, recorte)
                    x.append(recorte)
                    y.append(s["EL_sensible_crc"].sum() / 1e6)
                ax.plot(x, y, marker="o", linewidth=2,
                        label=escenario, color=COLORES.get(escenario, "#415a77"))
            ax.axvline(input.f_recorte(), color="#707b8a", linestyle="--",
                       linewidth=1.4, label="Recorte seleccionado")
            ax.set_xlabel("Recorte adicional de recuperaciones (%)")
            ax.set_ylabel("Pérdida esperada (millones de ₡)")
            ax.grid(alpha=0.16)
            ax.legend(frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            return fig

        @render.data_frame
        def tabla_consolidado():
            d = resumir_el(datos_ambos_escenarios(), ["ano", "escenario"])
            d["EL (millones CRC)"] = (d["EL_crc"] / 1e6).round(2)
            d["EAD base (millones CRC)"] = (d["EAD_base_crc"] / 1e6).round(2)
            d["EL/EAD (%)"] = d["EL_EAD_pct"].round(2)
            return render.DataGrid(d[["ano", "escenario", "EL (millones CRC)",
                                       "EAD base (millones CRC)", "EL/EAD (%)"]],
                                   filters=True, width="100%")

        @render.data_frame
        def tabla_detalle():
            d = datos_filtrados().copy()
            d["PE (millones CRC)"] = (d["EL_crc"] / 1e6).round(2)
            d["EAD base (millones CRC)"] = (
                d["EAD_ano_anterior_crc"] / 1e6
            ).round(2)
            d["PE/EAD (%)"] = d["EL_sobre_EAD_pct"].round(2)
            d["PE/patrimonio 2025 (%)"] = d["EL_sobre_patrimonio_pct"].round(2)
            d["PE por operación (CRC)"] = d["EL_por_operacion_crc"].round(2)
            cols = ["entidad", "categorias_bust", "escenario", "ano",
                    "PE (millones CRC)", "EAD base (millones CRC)",
                    "PE/EAD (%)", "PE/patrimonio 2025 (%)", "PE por operación (CRC)"]
            return render.DataGrid(d[cols].sort_values("PE (millones CRC)",
                                                       ascending=False),
                                   filters=True, width="100%", height="420px")

        @render.data_frame
        def tabla_operaciones():
            d = pe_operaciones_2026.copy()
            if input.f_entidad() != "Todas":
                d = d.loc[d["entidad"].eq(input.f_entidad())]
            if input.f_cartera() != "Todas":
                d = d.loc[d["categorias_bust"].eq(input.f_cartera())]
            if input.f_escenario() != "Ambos":
                d = d.loc[d["escenario"].eq(input.f_escenario())]
            d = d.nlargest(10, "EL_operacion_crc").copy()
            d["Saldo (millones CRC)"] = (d["saldo_crc"] / 1e6).round(2)
            d["PE (millones CRC)"] = (d["EL_operacion_crc"] / 1e6).round(2)
            return render.DataGrid(
                d[["operacion_id", "entidad", "categorias_bust", "estado_t0",
                   "escenario", "Saldo (millones CRC)", "PE (millones CRC)"]],
                width="100%", filters=True,
            )

        @render.data_frame
        def tabla_comparacion():
            d = comparacion_base_adverso.copy()
            if input.f_entidad() != "Todas":
                d = d.loc[d["entidad"].eq(input.f_entidad())]
            if input.f_cartera() != "Todas":
                d = d.loc[d["categorias_bust"].eq(input.f_cartera())]
            if input.f_ano() != "Todos":
                d = d.loc[d["ano"].eq(int(input.f_ano()))]
            d["PE Base (millones CRC)"] = (d["EL_Base_crc"] / 1e6).round(2)
            d["PE Adverso (millones CRC)"] = (d["EL_Adverso_crc"] / 1e6).round(2)
            d["Diferencia (millones CRC)"] = (d["Diferencia_crc"] / 1e6).round(2)
            d["Diferencia (%)"] = d["Diferencia_pct"].round(2)
            return render.DataGrid(d[["ano", "entidad", "categorias_bust",
                                       "PE Base (millones CRC)",
                                       "PE Adverso (millones CRC)",
                                       "Diferencia (millones CRC)", "Diferencia (%)"]],
                                   filters=True, width="100%", height="420px")

        @render.data_frame
        def tabla_sensibilidad():
            d = datos_sensibles().copy()
            d["PE original (millones CRC)"] = (d["EL_crc"] / 1e6).round(2)
            d["PE ajustada (millones CRC)"] = (d["EL_sensible_crc"] / 1e6).round(2)
            d["Aumento (millones CRC)"] = (d["incremento_EL_crc"] / 1e6).round(2)
            d["Aumento (%)"] = d["incremento_EL_pct"].round(2)
            return render.DataGrid(
                d[["entidad", "categorias_bust", "escenario", "ano",
                   "PE original (millones CRC)", "PE ajustada (millones CRC)",
                   "Aumento (millones CRC)", "Aumento (%)"]],
                filters=True, width="100%", height="420px",
            )

        @render.plot
        def grafico_historico_saldo():
            d = historico_filtrado().groupby("fecha", as_index=False)["saldo_crc"].sum()
            fig, ax = plt.subplots(figsize=(8.5, 3.8))
            ax.plot(d["fecha"], d["saldo_crc"] / 1e6,
                    color="#2765b0", linewidth=2)
            ax.set_ylabel("Saldo al cierre mensual (millones CRC)")
            ax.set_xlabel("Fecha de observación")
            ax.grid(alpha=0.17)
            ax.spines[["top", "right"]].set_visible(False)
            fig.autofmt_xdate()
            fig.tight_layout()
            return fig

        @render.plot
        def grafico_historico_composicion():
            d = historico_filtrado()
            d = d.loc[(d["ano"].eq(int(input.f_ano_hist())))
                      & d["fecha"].dt.month.eq(12)]
            g = d.groupby("categorias_bust")["saldo_crc"].sum().sort_values()
            fig, ax = plt.subplots(figsize=(8.5, 3.8))
            if not g.empty and g.sum() > 0:
                ax.barh(g.index, 100 * g / g.sum(), color="#5e8eb8")
            else:
                ax.text(0.5, 0.5, "Sin observaciones en el corte", ha="center",
                        va="center", transform=ax.transAxes)
            ax.set_xlabel("Participación del saldo de diciembre (%)")
            ax.tick_params(axis="y", labelsize=8)
            ax.set_xlim(left=0)
            ax.grid(axis="x", alpha=0.17)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            return fig

        @render.plot
        def grafico_tasas_historicas():
            d = tasas_por_cohorte()
            fig, ax = plt.subplots(figsize=(8.5, 3.7))
            for nombre, color in [("PD (%)", "#2765b0"),
                                  ("PC (%)", "#db7844")]:
                ax.plot(d["Cierre"], d[nombre], marker="o", label=nombre,
                        color=color, linewidth=2)
            ax.set_xticks(list(range(2020, 2026)))
            ax.set_ylabel("Tasa histórica por número de operaciones (%)")
            ax.set_xlabel("Año de cierre del seguimiento de 12 meses")
            ax.legend(frameon=False)
            ax.grid(alpha=0.17)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            return fig

        @render.data_frame
        def tabla_historico_cartera():
            d = historico_filtrado()
            d = d.loc[d["fecha"].dt.month.eq(12)
                      & d["ano"].eq(int(input.f_ano_hist()))]
            d = d[["ano", "entidad", "categorias_bust", "operaciones",
                   "saldo_crc"]].copy()
            d["Saldo (millones CRC)"] = (d["saldo_crc"] / 1e6).round(2)
            d = d.drop(columns="saldo_crc").sort_values(
                ["ano", "entidad", "categorias_bust"])
            return render.DataGrid(d, filters=True, width="100%", height="350px")

        @render.data_frame
        def tabla_tasas_historicas():
            d = tasas_por_cohorte().copy()
            for c in ["PD (%)", "PC (%)"]:
                d[c] = d[c].round(3)
            d = d.rename(columns={"cohorte": "Inicio", "normales": "Normales iniciales",
                                  "dudosas": "Dudosos iniciales",
                                  "incumplimientos": "Nuevos incumplimientos",
                                  "curas": "Curas / cancelaciones elegibles"})
            return render.DataGrid(d[["Inicio", "Cierre", "Normales iniciales",
                                      "Dudosos iniciales", "Nuevos incumplimientos",
                                      "Curas / cancelaciones elegibles", "PD (%)", "PC (%)"]],
                                   filters=True, width="100%")

        @render.plot
        def grafico_tasas_proyectadas():
            # Siempre comparamos Base y Adverso, incluso si el usuario dejó
            # seleccionado solo uno de ellos en el filtro lateral.
            # Así evitamos que aparezca un único punto perdido en el gráfico.
            d = datos_ambos_escenarios()
            columna = input.f_metrica()
            nombres = {
                "PD": "Probabilidad de incumplimiento (PD)",
                "PC": "Tasa de cura (PC)",
                "LGD": "Severidad de pérdida (LGD)",
                "LGL_cartera": "Pérdida sin ajuste por cura (LGL)",
            }

            # Agrupamos por año y escenario. Como las tasas no se suman,
            # calculamos el promedio usando la exposición como peso.
            # np.average permite que una cartera con exposición mayor tenga
            # más peso, en lugar de contar todas las carteras por igual.
            filas = []
            for (ano, escenario), g in d.groupby(["ano", "escenario"]):
                pesos = g["EAD_ano_anterior_crc"].to_numpy(dtype=float)
                tasas = g[columna].to_numpy(dtype=float)
                validos = np.isfinite(pesos) & np.isfinite(tasas) & (pesos > 0)
                if validos.any():
                    tasa = 100 * np.average(tasas[validos], weights=pesos[validos])
                    filas.append((int(ano), str(escenario), tasa))

            fig, ax = plt.subplots(figsize=(8.5, 3.8))
            if filas:
                # Las barras horizontales son más fáciles de leer que una
                # línea con un solo punto cuando filtramos un único año.
                filas.sort(key=lambda r: (r[0], r[1] != "Base"))
                etiquetas = [f"{ano} · {escenario}" for ano, escenario, _ in filas]
                valores = [valor for _, _, valor in filas]
                colores = [COLORES[escenario] for _, escenario, _ in filas]
                posiciones = np.arange(len(filas))

                ax.barh(posiciones, valores, color=colores, height=0.65)
                ax.set_yticks(posiciones, labels=etiquetas)
                ax.invert_yaxis()
                limite = min(100, max(5, max(valores) * 1.22))
                ax.set_xlim(0, limite)
                # Escribimos el porcentaje sobre cada barra para no tener que
                # estimarlo visualmente a partir del eje horizontal.
                for y, valor in zip(posiciones, valores):
                    ax.text(valor + limite * 0.015, y, f"{valor:.2f} %",
                            va="center", ha="left", fontsize=10, color=COLOR_TEXTO)
            else:
                ax.text(0.5, 0.5, "No hay datos para esta selección",
                        ha="center", va="center", transform=ax.transAxes)

            ax.set_xlabel(f"{nombres.get(columna, columna)} ponderada por EAD (%)")
            ax.grid(axis="x", alpha=0.14)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            fig.tight_layout()
            return fig

        @render.plot
        def grafico_ead_el():
            d = indicadores_ponderados()
            fig, ax = plt.subplots(figsize=(8.5, 3.8))
            if not d.empty:
                # Dibujamos por cada escenario seleccionado sin sumar años:
                # así el gráfico muestra importes comparables por horizonte.
                g = d.sort_values(["Año", "Escenario"])
                etiquetas = [f"{a} {e[0]}" for a, e in
                             zip(g["Año"], g["Escenario"])]
                x = np.arange(len(g))
                ax.bar(x - 0.18, g["EAD (millones CRC)"], width=0.36,
                       label="EAD", color="#8ca5c2")
                ax.bar(x + 0.18, g["PE (millones CRC)"], width=0.36,
                       label="PE", color="#db7844")
                ax.set_xticks(x, etiquetas)
            ax.set_ylabel("Importe (millones CRC)")
            ax.set_xlabel("Año y escenario (B: Base; A: Adverso)")
            ax.grid(axis="y", alpha=0.17)
            if not d.empty:
                ax.legend(frameon=False)
            else:
                ax.text(0.5, 0.5, "No hay datos para esta selección",
                        ha="center", va="center", transform=ax.transAxes)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            return fig

        @render.data_frame
        def tabla_indicadores_agrupados():
            d = indicadores_ponderados().copy()
            for c in ["PD", "PC", "LGL_cartera", "LGD", "PE/EAD (%)"]:
                d[c] = d[c].round(3)
            for c in ["EAD (millones CRC)", "PE (millones CRC)"]:
                d[c] = d[c].round(2)
            d = d.rename(columns={"PD": "PD ponderada (%)", "PC": "PC ponderada (%)",
                                  "LGL_cartera": "LGL ponderada (%)",
                                  "LGD": "LGD ponderada (%)"})
            return render.DataGrid(d, filters=True, width="100%")

        @render.data_frame
        def tabla_indicadores_detalle():
            d = datos_filtrados().copy()
            for c in ["PD", "PC", "LGL_cartera", "LGD"]:
                d[c] = (100 * d[c]).round(3)
            for c in ["EAD_ano_anterior_crc", "EAD_proyectada_mn",
                      "CN_2025_crc", "CD_2025_crc", "recuperacion_cartera_crc",
                      "EL_normal_crc", "EL_dudosa_crc", "EL_futura_crc", "EL_crc"]:
                d[c] = (d[c] / 1e6).round(2)
            d = d[["entidad", "categorias_bust", "escenario", "ano", "PD", "PC",
                   "LGL_cartera", "LGD", "CN_2025_crc", "CD_2025_crc",
                   "EAD_ano_anterior_crc", "EAD_proyectada_mn",
                   "recuperacion_cartera_crc", "EL_normal_crc",
                   "EL_dudosa_crc", "EL_futura_crc", "EL_crc"]]
            return render.DataGrid(d, filters=True, width="100%", height="425px")

        @render.download(filename="proyecciones_PE_PD_PC_LGD_EAD.csv")
        def descargar_proyecciones():
            # Exportamos datos calculados sin redondear, para que otra persona
            # pueda rehacer las cuentas a partir del CSV.
            yield descargar_csv(datos_filtrados())

        @render.download(filename="resumen_PE_por_segmento.csv")
        def descargar_resumen():
            d = datos_filtrados()
            if d.empty:
                yield descargar_csv(pd.DataFrame())
            else:
                yield descargar_csv(resumir_el(d, ["escenario", "ano",
                                                   "entidad", "categorias_bust"]))

        @render.download(filename="historico_cartera_2019_2025.csv")
        def descargar_historico():
            yield descargar_csv(historico_filtrado())

        @render.download(filename="PD_PC_historicas_cohortes.csv")
        def descargar_tasas_historicas():
            yield descargar_csv(tasas_por_cohorte())

        @render.download(filename="sensibilidad_recuperaciones.csv")
        def descargar_sensibilidad():
            yield descargar_csv(datos_sensibles())

        @render.download(filename="operaciones_PE_2026.csv")
        def descargar_operaciones():
            d = pe_operaciones_2026
            if input.f_entidad() != "Todas":
                d = d.loc[d["entidad"].eq(input.f_entidad())]
            if input.f_cartera() != "Todas":
                d = d.loc[d["categorias_bust"].eq(input.f_cartera())]
            if input.f_escenario() != "Ambos":
                d = d.loc[d["escenario"].eq(input.f_escenario())]
            yield descargar_csv(d)

    app = App(app_ui, server)

else:
    # Esta rama permite ejecutar únicamente el procesamiento estadístico si
    # todavía no se instaló Shiny. Para abrir el dashboard sí hace falta.
    app = None
    if __name__ == "__main__":
        print("Shiny no está instalado. Para abrir el dashboard instalá shiny y matplotlib.")


# Si abrís el archivo con python app.py, obtendrás un resumen de resultados;
# si lo abrís con shiny run app.py, iniciás la aplicación web.
if __name__ == "__main__":
    print("\nPérdida esperada consolidada (millones de colones):")
    print(pe_consolidada.assign(
        EL_millones=lambda d: (d["EL_crc"] / 1e6).round(2)
    )[["escenario", "ano", "EL_millones", "EL_EAD_pct"]].to_string(index=False))
    print("\nValidación: inventario, operaciones 2026 y total A-E conciliados.")
    if app is not None:
        print("Para abrir Shiny: shiny run --launch-browser app.py")

if __name__ == "__main__":

    ruta_app = Path.cwd() / "app.py"

    subprocess.Popen([
        sys.executable,
        "-m",
        "shiny",
        "run",
        "--launch-browser",
        str(ruta_app)
    ])



