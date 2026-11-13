# -*- coding: utf-8 -*-

# %%

# Se agregan las librerías correspondientes
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

"""
Dashboard de riesgo de crédito

Aquí se juntan todas las funciones que se usan para la que la aplicación
se ejecute correctamente, junto con los gráficos y demás elementos.
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
    Prepara el inventario de operaciones crediticias para una fecha de evaluación.

    Parámetros
    ----------
    df_operaciones : DataFrame
        Base histórica de operaciones crediticias.
    fecha_corte : str
        Fecha de evaluación en formato YYYY-MM-DD.
    excluir_btb : bool
        Indica si se excluyen las operaciones back-to-back.
    excluir_otro : bool
        Indica si se excluyen las operaciones con categoría Otro.

    Retorna
    -------
    DataFrame
        Inventario de operaciones en la fecha indicada.
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


# Documentar
def diagnosticar_salidas(
    df_operaciones,
    inventario_inicial,
    inventario_final,
    fecha_inicial,
    fecha_final,
):
    """
    Diagnostica operaciones del inventario inicial ausentes en el
    inventario final.

    No clasifica incumplimientos ni curas.
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


# Documentar
def resumir_modificaciones_crediticias(
    df_operaciones, fecha_corte, meses_ventana=12
):
    """
    Resume las modificaciones crediticias por operación utilizando información
    disponible hasta fecha_corte.

    Retorna una fila por operación modificada dentro de la ventana seleccionada.
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


# Documentar
def construir_indicadores_crediticios(
    df_operaciones, fecha_corte, meses_modificaciones=12
):
    """
    Construye indicadores crediticios por operación
    para una fecha de evaluación.

    No asigna todavía el estado BUST definitivo.
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


# Documentar
def clasificar_inventario_inicial_crediticio(
    df_operaciones, fecha_corte, meses_ventana=12
):
    """
    Clasifica operaciones en T0 para PD y PC.

    Excluye back-to-back, cartera Otro y cancelaciones ocurridas en T0 o antes.

    La clasificación no utiliza información posterior a la fecha de corte.
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


# Documentar
def evaluar_operaciones_t12(
    df_operaciones, inventario_t0, fecha_t0, meses_horizonte=12
):
    """
    Evalúa el estado observado de las operaciones del inventario inicial
    al cierre del horizonte.

    Conserva una fila por operación inicial.
    No aplica todavía el ajuste de modificaciones.
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


# Documentar
def evaluar_crecimiento_modificaciones(
    df_operaciones, seguimiento_t12, fecha_t0
):
    """
    Evalúa las modificaciones crediticias en la ventana
    de 12 meses antes y 12 meses después de T0.

    Para operaciones con modificaciones:
    - Cuenta las modificaciones.
    - Obtiene la mora previa a la primera modificación.
    - Obtiene el saldo previo a la última modificación.
    - Compara ese saldo con el saldo en T+12.
    - Aplica la regla de crecimiento superior al 2%.

    Conserva los casos sin información suficiente
    como no evaluables.
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


# Documentar
def clasificar_operaciones_t12(evaluacion_mods, fecha_t12, revision_ref=None):
    """
    Consolida la clasificación crediticia en T+12.

    Estados:
    - Normal
    - Dudoso
    - Pendiente
    - Cancelado

    Aplica:
    1. Deterioro básico.
    2. Evaluación de modificaciones.
    3. Tratamiento separado de salidas.
    4. Identificación de casos no evaluables.
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


# Documentar
def calcular_pd_pc_historicas(
    clasificacion, agrupar_por=("entidad", "categorias_bust")
):
    """
    Calcula PD y PC históricas por número de operaciones.

    Las operaciones pendientes permanecen en el
    denominador y se reportan separadamente.
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


# Documentar
def proyectar_pd_pc(
    pd_pc_app,
    df_macro_historico,
    df_macro_proyectado,
    df_operaciones_crediticias,
    df_tipo_cambio,
):
    """PD: tipo de cambio (+). PC: tasa MN (-).

    Regresion logistica binomial con interceptos entidad-BUST,
    restriccion de signo y penalizacion 5*beta_macro**2.
    Replica el ajuste usado en la comparacion temporal de modelos.
    """
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
    from scipy.special import expit

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


# Documentar
def seleccionar_garantias_admisibles(df_garantias_operaciones):

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


# Documentar
def valorar_garantias_2025(
    garantias_admisibles, df_operaciones_crediticias, df_tipo_cambio
):

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


# Documentar
def calcular_recuperaciones_historicas(
    df_operaciones_bienes_realizables, df_flujos_recuperacion
):

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


# Documentar
def proyectar_recuperaciones_garantias(
    garantias_valoradas,
    resumen_recuperaciones,
    df_macro_historico,
    df_macro_proyectado,
):

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


# Documentar
def descontar_recuperaciones(
    recuperaciones_proyectadas, df_operaciones_crediticias, df_tipo_cambio
):

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


# Documentar
def calcular_cobertura_recuperable(recuperaciones_vp, garantias_valoradas):

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


# Documentar
def calcular_lgl_cartera(
    cobertura_recuperable, df_operaciones_crediticias, df_tipo_cambio
):

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


# Documentar
def valorar_bienes_pendientes(
    df_operaciones_bienes_realizables,
    resumen_recuperaciones,
    df_macro_historico,
    df_macro_proyectado,
):

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


# Documentar
def descontar_bienes_pendientes(bienes_pendientes, df_operaciones_crediticias):

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


# Documentar
def actualizar_avaluos_terrenos(garantias_admisibles, df_macro_historico):

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


# Documentar
def preparar_insumos_secciones_4_5(proyecciones_pd_pc, lgl_cartera):

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

# 

# %%

"""" Escenarios y sensibilidad """

# 


# %%














