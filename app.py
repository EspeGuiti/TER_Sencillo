import streamlit as st
import pandas as pd
import numpy as np

# =========================
# Configuración de página
# =========================
st.set_page_config(page_title="Calculadora TER + AI", layout="wide")
st.markdown("# 📊 Calculadora de TER — Cartera I vs Cartera II (Asesoramiento Independiente)")

# =========================
# Estado de sesión
# =========================
if "cartera_I" not in st.session_state:
    st.session_state.cartera_I = None  # {"table": df, "ter": float}
if "cartera_II" not in st.session_state:
    st.session_state.cartera_II = None
if "incidencias" not in st.session_state:
    st.session_state.incidencias = []

# =========================
# Helpers
# =========================
def _to_float_percent_like(x):
    """Convierte '1,23%' o '1.23' a float (1.23)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace("%", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def _clean_master(dfm):
    """Limpia columnas críticas del maestro."""
    df = dfm.copy()
    # Ongoing Charge a float (no %)
    if "Ongoing Charge" in df.columns:
        df["Ongoing Charge"] = df["Ongoing Charge"].apply(_to_float_percent_like)
    # Normalizar Transferable a Yes/No strings
    if "Transferable" in df.columns:
        def norm_tf(v):
            if pd.isna(v): return ""
            s = str(v).strip().lower()
            if s in {"yes", "y", "true", "1"}: return "Yes"
            if s in {"no", "n", "false", "0"}: return "No"
            return str(v)  # dejar tal cual si trae otro valor
        df["Transferable"] = df["Transferable"].apply(norm_tf)
    return df

def _clean_weights(dfw):
    """Limpia pesos de la cartera."""
    df = dfw.copy()
    if "Peso %" not in df.columns:
        raise ValueError("El Excel de cartera debe incluir la columna 'Peso %'.")
    df["Peso %"] = df["Peso %"].apply(_to_float_percent_like)
    return df

def _format_eu_number(x, decimals=4):
    if pd.isna(x):
        return x
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def pretty_table(df_in: pd.DataFrame) -> pd.DataFrame:
    """Reordena y renombra columnas para la visualización."""
    tbl = df_in.copy()
    # Unificar "Transferable" -> "Traspasable" para mostrar
    if "Traspasable" not in tbl.columns and "Transferable" in tbl.columns:
        tbl.rename(columns={"Transferable": "Traspasable"}, inplace=True)

    desired = [
        "Family Name",
        "Type of Share", "Currency", "Hedged", "MiFID FH", "Min. Initial",
        "ISIN", "Prospectus AF", "Traspasable",
        "Ongoing Charge", "Weight %"
    ]
    existing = [c for c in desired if c in tbl.columns]
    rest = [c for c in tbl.columns if c not in existing]
    return tbl[existing + rest]

def calcular_ter(df_rows: pd.DataFrame) -> float:
    """TER ponderado: sum(OC * w/100) / (sum(w)/100)."""
    if df_rows.empty:
        return None
    if "Ongoing Charge" not in df_rows.columns or "Weight %" not in df_rows.columns:
        return None
    total_w = df_rows["Weight %"].sum()
    if total_w <= 0:
        return None
    weighted = (df_rows["Ongoing Charge"] * (df_rows["Weight %"] / 100.0)).sum()
    return weighted / (total_w / 100.0)

def merge_cartera_con_maestro(df_master: pd.DataFrame, df_weights: pd.DataFrame):
    """
    Devuelve:
      - df_merged por ISIN con detalles del maestro (si ISIN duplica en maestro, se toma el de menor OC)
      - incidencias por ISIN no encontrado
    """
    # Si el maestro tiene múltiples filas por mismo ISIN, tomamos la de menor Ongoing Charge
    master_isin = df_master.sort_values("Ongoing Charge", na_position="last").drop_duplicates("ISIN", keep="first")
    merged = pd.merge(df_weights, master_isin, on="ISIN", how="left", validate="one_to_one")
    incidencias = []
    if merged["Family Name"].isnull().any():
        faltan = merged[merged["Family Name"].isnull()]["ISIN"].tolist()
        incidencias.append(("ISIN no encontrado en maestro", ", ".join(map(str, faltan))))
        merged = merged[~merged["Family Name"].isnull()].copy()

    # Normalizamos nombre de columna de peso para trabajar internamente como "Weight %"
    merged.rename(columns={"Peso %": "Weight %"}, inplace=True)
    return merged, incidencias

def convertir_a_AI(df_master: pd.DataFrame, df_cartera_I: pd.DataFrame):
    """
    Para cada fila de Cartera I (que ya tiene datos completos del maestro),
    busca en el maestro una clase con:
      - mismo Family Name
      - mismo Type of Share, Currency, Hedged
      - Prospectus AF que contenga 'AI' (case-insensitive)
      - Transferable == 'Yes'
    De entre las candidatas, elige la de menor Ongoing Charge.
    Mantiene el peso.
    Devuelve (df_cartera_II, incidencias)
    """
    results = []
    incidencias = []
    for _, row in df_cartera_I.iterrows():
        fam = row.get("Family Name")
        tos = row.get("Type of Share")
        cur = row.get("Currency")
        hed = row.get("Hedged")
        w   = row.get("Weight %", 0.0)

        # Filtro base
        candidates = df_master[
            (df_master["Family Name"] == fam) &
            (df_master["Type of Share"] == tos) &
            (df_master["Currency"] == cur) &
            (df_master["Hedged"] == hed)
        ].copy()

        if "Prospectus AF" in candidates.columns:
            candidates["_has_ai"] = candidates["Prospectus AF"].astype(str).str.contains("AI", case=False, na=False)
        else:
            candidates["_has_ai"] = False

        if "Transferable" in candidates.columns:
            candidates["_tf_yes"] = (candidates["Transferable"] == "Yes")
        else:
            candidates["_tf_yes"] = False

        candidates = candidates[candidates["_has_ai"] & candidates["_tf_yes"]].copy()

        if candidates.empty:
            incidencias.append((str(fam), "No hay clase AI + Transferable=Yes manteniendo Currency/Hedged/Type of Share"))
            continue

        # Elegimos la de menor Ongoing Charge
        candidates = candidates.sort_values("Ongoing Charge", na_position="last")
        best = candidates.iloc[0]

        out = {
            "Family Name":   best.get("Family Name"),
            "Type of Share": best.get("Type of Share"),
            "Currency":      best.get("Currency"),
            "Hedged":        best.get("Hedged"),
            "MiFID FH":      best.get("MiFID FH"),
            "Min. Initial":  best.get("Min. Initial"),
            "ISIN":          best.get("ISIN"),
            "Prospectus AF": best.get("Prospectus AF"),
            "Transferable":  "Yes",  # Confirmado por filtro; lo mantenemos explícitamente como "Yes"
            "Ongoing Charge": best.get("Ongoing Charge"),
            "Weight %":      float(w) if pd.notnull(w) else 0.0
        }
        results.append(out)

    df_ii = pd.DataFrame(results)
    return df_ii, incidencias

def mostrar_tabla_con_formato(df_in, title):
    st.markdown(f"#### {title}")
    df_show = pretty_table(df_in).copy()
    # Formateos numéricos (estilo europeo)
    for col in ["Ongoing Charge", "Weight %"]:
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(lambda x: _format_eu_number(x, 4) if pd.notnull(x) else x)
    st.dataframe(df_show, use_container_width=True)


# =========================
# 1) Subida de archivos
# =========================
st.subheader("Paso 1: Subir ficheros")
master_file = st.file_uploader("📥 Sube el Excel MAESTRO (todas las clases)", type=["xlsx"], key="master")
weights_file = st.file_uploader("📥 Sube el Excel de CARTERA (columnas 'ISIN' y 'Peso %')", type=["xlsx"], key="weights")

if not master_file or not weights_file:
    st.info("Sube ambos ficheros para continuar.")
    st.stop()

# =========================
# 2) Carga y validaciones
# =========================

# ✅ Leer el Excel maestro EXACTAMENTE como en tu app original
df_master_raw = pd.read_excel(master_file, skiprows=2)

# Validar columnas requeridas como en tu app original
required_cols = [
    "Family Name","Type of Share","Currency","Hedged",
    "MiFID FH","Min. Initial","Ongoing Charge","ISIN","Prospectus AF"
]
missing = [c for c in required_cols if c not in df_master_raw.columns]
if missing:
    st.error(f"Faltan columnas en el maestro: {missing}")
    st.stop()

# Detectar 'Transferable' como en la app original
has_transferable = "Transferable" in df_master_raw.columns
st.success("Fichero maestro importado correctamente.")

# ✅ Limpieza de 'Ongoing Charge' EXACTA a la app original
df_master_raw["Ongoing Charge"] = (
    df_master_raw["Ongoing Charge"].astype(str)
        .str.replace("%","", regex=False)
        .str.replace(",",".", regex=False)
        .astype(float)
)

# Mantener la variable df_master para el resto de tu lógica
df_master = df_master_raw

required_cols = [
    "Family Name","Type of Share","Currency","Hedged",
    "MiFID FH","Min. Initial","Ongoing Charge","ISIN","Prospectus AF"
]
missing = [c for c in required_cols if c not in df_master_raw.columns]
if missing:
    st.error(f"Faltan columnas en el maestro: {missing}")
    st.stop()

has_transferable = "Transferable" in df_master_raw.columns
if not has_transferable:
    st.warning("El maestro no tiene columna 'Transferable'. Se asumirá vacío y no se podrá convertir a AI correctamente.")

df_master = _clean_master(df_master_raw)

try:
    df_weights_raw = pd.read_excel(weights_file)
except Exception as e:
    st.error(f"No se pudo leer la cartera: {e}")
    st.stop()

if "ISIN" not in df_weights_raw.columns or "Peso %" not in df_weights_raw.columns:
    st.error("El Excel de cartera debe incluir las columnas: 'ISIN' y 'Peso %'.")
    st.stop()

df_weights = _clean_weights(df_weights_raw)
df_weights = df_weights.groupby("ISIN", as_index=False)["Peso %"].sum()  # agrupar duplicados

# =========================
# 3) Cartera I (original) + TER
# =========================
st.subheader("Paso 2: Calcular Cartera I (original)")

df_I_raw, incidencias_merge = merge_cartera_con_maestro(df_master, df_weights)

# Mostrar suma de pesos y advertencia si no es 100
peso_total_I = df_I_raw["Weight %"].sum() if "Weight %" in df_I_raw.columns else df_I_raw["Peso %"].sum()
st.write(f"**Suma de pesos cartera (I):** {peso_total_I:.2f}%")
if abs(peso_total_I - 100.0) > 1e-6:
    st.warning("La suma de pesos no es 100%. Corrige tu Excel de cartera.")

# TER y tabla
ter_I = calcular_ter(df_I_raw.rename(columns={"Peso %": "Weight %"}))
st.session_state.cartera_I = {"table": df_I_raw.rename(columns={"Peso %": "Weight %"}), "ter": ter_I}

mostrar_tabla_con_formato(st.session_state.cartera_I["table"], "Tabla Cartera I (original)")
if st.session_state.cartera_I["ter"] is not None:
    st.metric("📊 TER Cartera I", f"{st.session_state.cartera_I['ter']:.2%}".replace(".", ","))

# Incidencias de merge
incidencias = list(incidencias_merge)

# =========================
# 4) Convertir a Cartera II (AI)
# =========================
st.subheader("Paso 3: Convertir a Cartera de Asesoramiento Independiente (Cartera II)")
st.caption("Se mantiene Type of Share, Currency, Hedged; se exige Prospectus AF con 'AI' y Transferable = 'Yes'; se elige la clase con menor Ongoing Charge.")

if st.button("🔁 Convertir a cartera Asesoramiento Independiente"):
    df_II, incid_AI = convertir_a_AI(df_master, st.session_state.cartera_I["table"])
    st.session_state.cartera_II = {"table": df_II, "ter": calcular_ter(df_II)}
    incidencias.extend(incid_AI)

# Mostrar Cartera II si existe
if st.session_state.cartera_II and not st.session_state.cartera_II["table"].empty:
    mostrar_tabla_con_formato(st.session_state.cartera_II["table"], "Tabla Cartera II (AI)")
    if st.session_state.cartera_II["ter"] is not None:
        st.metric("📊 TER Cartera II (AI)", f"{st.session_state.cartera_II['ter']:.2%}".replace(".", ","))

# =========================
# 5) Comparación I vs II
# =========================
st.subheader("Paso 4: Comparar Cartera I vs Cartera II")
if st.session_state.cartera_I and st.session_state.cartera_I["ter"] is not None and \
   st.session_state.cartera_II and st.session_state.cartera_II["ter"] is not None:

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Cartera I")
        st.metric("TER medio ponderado", f"{st.session_state.cartera_I['ter']:.2%}")
        st.dataframe(pretty_table(st.session_state.cartera_I['table']), use_container_width=True)

    with c2:
        st.markdown("#### Cartera II (AI)")
        st.metric("TER medio ponderado", f"{st.session_state.cartera_II['ter']:.2%}")
        st.dataframe(pretty_table(st.session_state.cartera_II['table']), use_container_width=True)

    diff = st.session_state.cartera_II["ter"] - st.session_state.cartera_I["ter"]
    st.markdown("---")
    st.subheader("Diferencia de TER (II − I)")
    st.metric("Diferencia", f"{diff:.2%}")

# =========================
# 6) Incidencias
# =========================
if incidencias:
    st.subheader("⚠️ Incidencias detectadas")
    for fam, msg in incidencias:
        st.error(f"{fam}: {msg}")

