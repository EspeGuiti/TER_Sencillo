import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata

# =========================
# Configuración de página
# =========================
st.set_page_config(page_title="Calculadora TER + AI", layout="wide")
st.markdown("# 📊 Calculadora de TER — Cartera I vs Cartera II (Asesoramiento Independiente)")

# =========================
# Estado de sesión
# =========================
if "cartera_I" not in st.session_state:
    st.session_state.cartera_I = None                 # {"table": df, "ter": float}
if "cartera_I_raw" not in st.session_state:
    st.session_state.cartera_I_raw = None             # df mergeado por ISIN (para convertir)
if "cartera_II" not in st.session_state:
    st.session_state.cartera_II = None                # {"table": df, "ter": float}
if "incidencias" not in st.session_state:
    st.session_state.incidencias = []

# =========================
# Helpers
# =========================
def _to_float_percent_like(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("%","").replace(",",".")
    try: return float(s)
    except: return np.nan

def _to_float_eu_money(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int,float,np.number)):
        return float(x) if float(x) > 0 else np.nan
    s = str(x).strip()
    if s == "": return np.nan
    s = s.replace("€","").replace("EUR","").replace(" ","")
    s = s.replace(".","").replace(",",".")  # 1.234,56 -> 1234.56
    try:
        v = float(s)
        return v if v > 0 else np.nan
    except:
        return np.nan

def _norm_txt(s):
    """Normaliza texto para buscar encabezados de forma robusta."""
    if pd.isna(s): return ""
    s = str(s).replace("\xa0"," ")
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii","ignore").decode("ascii")
    s = re.sub(r"\s+"," ",s).strip().lower()
    return s

def _find_header_cell(df_any, targets):
    """Busca la primera celda cuyo _norm_txt esté en targets."""
    for r in range(df_any.shape[0]):
        for c in range(df_any.shape[1]):
            if _norm_txt(df_any.iat[r,c]) in targets:
                return r,c
    return None,None

def _has_code(s: str, code: str) -> bool:
    if pd.isna(s): return False
    tokens = re.split(r'[^A-Za-z0-9]+', str(s).upper())
    return code.upper() in tokens

def _fmt_ratio_eu_percent(x, decimals=2):
    if x is None: return "-"
    return f"{x:.{decimals}%}".replace(".", ",")

def _format_eu_number(x, decimals=2):
    if pd.isna(x): return ""
    s = f"{float(x):,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def _clean_master(dfm):
    df = dfm.copy()
    if "Ongoing Charge" in df.columns:
        df["Ongoing Charge"] = df["Ongoing Charge"].apply(_to_float_percent_like)
    if "Transferable" in df.columns:
        def norm_tf(v):
            if pd.isna(v): return ""
            s = str(v).strip().lower()
            if s in {"yes","y","true","1"}: return "Yes"
            if s in {"no","n","false","0"}:  return "No"
            return str(v)  # dejar blancos u otros valores tal cual
        df["Transferable"] = df["Transferable"].apply(norm_tf)
    return df

def pretty_table(df_in: pd.DataFrame) -> pd.DataFrame:
    """Columnas a mostrar (con Name, no Family Name)."""
    tbl = df_in.copy()
    if "MiFID FH" in tbl.columns and "MIFID FH" not in tbl.columns:
        tbl.rename(columns={"MiFID FH":"MIFID FH"}, inplace=True)

    wanted = [
        "ISIN","Name","Type of Share","Currency","Hedged","Ongoing Charge",
        "Min. Initial","MIFID FH","MiFID EMT","Prospectus AF","Soft Close",
        "Subscription Fee","Redemption Fee","VALOR ACTUAL (EUR)","Weight %"
    ]
    for c in wanted:
        if c not in tbl.columns: tbl[c] = np.nan
    return tbl[wanted]

def calcular_ter_por_valor(df):
    """TER ponderado por Weight % (calculado por valor). Ignora filas sin OC."""
    if "Weight %" not in df.columns or "Ongoing Charge" not in df.columns: return None
    w = df["Weight %"].astype(float)
    oc = df["Ongoing Charge"].astype(float)
    mask = w.notna() & oc.notna()
    if not mask.any(): return None
    return np.nansum(w[mask]*oc[mask]) / w[mask].sum()

def recalcular_pesos_por_valor(df, valor_col="VALOR ACTUAL (EUR)"):
    """Normaliza Weight % a partir del valor en EUR (solo positivos)."""
    df2 = df.copy()
    vals = df2[valor_col].apply(_to_float_eu_money)
    df2[valor_col] = vals
    df2 = df2[vals.notna()]
    total = df2[valor_col].sum()
    if total and total > 0:
        df2["Weight %"] = df2[valor_col] / total * 100.0
    else:
        df2["Weight %"] = 0.0
    return df2
def recalcular_pesos_por_valor_respetando_oc(df, valor_col="VALOR ACTUAL (EUR)", oc_col="Ongoing Charge"):
    """
    Recalcula 'Weight %' usando SOLO los fondos con Ongoing Charge (no NaN).
    - Los fondos SIN OC se mantienen en la tabla pero con Weight % = 0.
    - Los pesos de los fondos con OC se normalizan para sumar 100.
    """
    df2 = df.copy()
    df2[valor_col] = df2[valor_col].apply(_to_float_eu_money)
    # máscara de elegibles para TER
    elig = df2[oc_col].astype(float).notna() if oc_col in df2.columns else pd.Series(False, index=df2.index)

    total_ok = df2.loc[elig, valor_col].sum()
    df2["Weight %"] = 0.0
    if total_ok and total_ok > 0:
        df2.loc[elig, "Weight %"] = (df2.loc[elig, valor_col] / total_ok) * 100.0
    return df2

# =========================
# 1) Subida de archivos
# =========================
st.subheader("Paso 1: Subir ficheros")
master_file  = st.file_uploader("📥 Excel AllFunds (Share Class Tool, completo)", type=["xlsx"], key="master")
cartera_file = st.file_uploader("📥 Excel de Landing de Carteras, cartera cliente. Debe incluir columnas ISIN y Valos Actual (EUR)", type=["xlsx"], key="weights")
if not master_file or not cartera_file:
    st.info("Sube ambos ficheros para continuar.")
    st.stop()

# =========================
# 2) Carga y validaciones
# =========================
# Maestro
df_master_raw = pd.read_excel(master_file, skiprows=2)
required_cols = ["Family Name","Type of Share","Currency","Hedged","MiFID FH","Min. Initial","Ongoing Charge","ISIN","Prospectus AF"]
missing = [c for c in required_cols if c not in df_master_raw.columns]
if missing:
    st.error(f"Faltan columnas en el maestro: {missing}")
    st.stop()
df_master_raw["Ongoing Charge"] = (
    df_master_raw["Ongoing Charge"].astype(str).str.replace("%","",regex=False).str.replace(",",".",regex=False).astype(float)
)
df_master = _clean_master(df_master_raw)

# Cartera (sin asumir cabeceras)
try:
    df_any = pd.read_excel(cartera_file, header=None)
except Exception as e:
    st.error(f"No se pudo leer la cartera: {e}")
    st.stop()

targets_isin  = {"isin"}
targets_valor = {"valor actual (eur)","valor actual(eur)","valor actual eur","valor actual €","valor actual€","valor actual"}
r_isin,  c_isin  = _find_header_cell(df_any, targets_isin)
r_valor, c_valor = _find_header_cell(df_any, targets_valor)
if r_isin is None or r_valor is None:
    st.error("EL excel adjunto de cartera no se ha podido leer")
    st.stop()

col_isin   = df_any.iloc[r_isin+1:,  c_isin].reset_index(drop=True)
col_valor  = df_any.iloc[r_valor+1:, c_valor].reset_index(drop=True)

# eliminar total final (si última fila no tiene ISIN o contiene 'total')
while len(col_isin)>0 and (pd.isna(col_isin.iloc[-1]) or _norm_txt(col_isin.iloc[-1]) in {"total","totales"}):
    col_isin  = col_isin.iloc[:-1]
    col_valor = col_valor.iloc[:-1]

min_len = min(len(col_isin), len(col_valor))
col_isin, col_valor = col_isin.iloc[:min_len], col_valor.iloc[:min_len]

df_cartera_raw = pd.DataFrame({"ISIN": col_isin, "VALOR ACTUAL (EUR)": col_valor}).dropna(how="all")
df_cartera_raw = df_cartera_raw[df_cartera_raw["ISIN"].notna()].copy()
df_cartera_raw["ISIN"] = df_cartera_raw["ISIN"].astype(str).str.strip().str.upper()
df_cartera_raw["VALOR ACTUAL (EUR)"] = df_cartera_raw["VALOR ACTUAL (EUR)"].apply(_to_float_eu_money)
df_cartera_raw = df_cartera_raw[df_cartera_raw["VALOR ACTUAL (EUR)"].notna()]

# si hay duplicados de ISIN, sumar valor
df_cartera_raw = df_cartera_raw.groupby("ISIN", as_index=False)["VALOR ACTUAL (EUR)"].sum()

# =========================
# 3) Cartera I: match por ISIN y limpieza
# =========================
st.subheader("Paso 2: Calcular Cartera I (original)")

# Merge por ISIN y DESCARTAR los que no están en AllFunds
df_master_idx = df_master.copy()
df_master_idx["ISIN"] = df_master_idx["ISIN"].astype(str).str.strip().str.upper()
df_I_raw = pd.merge(df_cartera_raw, df_master_idx, how="inner", on="ISIN")   # solo los que están en AllFunds

# Guardamos versión RAW (para conversión) con Name/Family/ToS/Currency/Hedged/OC, etc.
st.session_state.cartera_I_raw = df_I_raw.copy()

# Recalcular pesos por valor (solo los que están en AllFunds)
df_I = recalcular_pesos_por_valor_respetando_oc(df_I_raw, valor_col="VALOR ACTUAL (EUR)")
ter_I = calcular_ter_por_valor(df_I)
st.session_state.cartera_I = {"table": df_I, "ter": ter_I}

# Mostrar Cartera I
st.markdown(f"**Fondos válidos en AllFunds (Cartera I):** {len(df_I)}")
def mostrar_tabla_con_formato(df_in, title):
    st.markdown(f"#### {title}")
    df_show = pretty_table(df_in).copy()
    # formato
    if "Ongoing Charge" in df_show.columns:
        df_show["Ongoing Charge"] = df_show["Ongoing Charge"].apply(lambda v: _format_eu_number(v,4))
    if "Weight %" in df_show.columns:
        df_show["Weight %"] = df_show["Weight %"].apply(lambda v: _format_eu_number(v,2))
    if "VALOR ACTUAL (EUR)" in df_show.columns:
        df_show["VALOR ACTUAL (EUR)"] = df_show["VALOR ACTUAL (EUR)"].apply(lambda v: _format_eu_number(v,2))
    st.dataframe(df_show, use_container_width=True)

mostrar_tabla_con_formato(df_I, "Tabla Cartera I (solo ISIN encontrados en AllFunds, pesos por valor)")
st.metric("📊 TER Cartera I (real por valor)", _fmt_ratio_eu_percent(ter_I, 2) if ter_I is not None else "-")

# =========================
# 4) Conversión a Cartera II (AI)
# =========================
st.subheader("Paso 3: Convertir a Cartera de Asesoramiento Independiente (Cartera II)")
st.caption("Se busca clase apta para Asesoramiento Indendiente, manteniendo mismas características de divisa, cobertura, reparto")

def convertir_a_AI(df_master: pd.DataFrame, df_cartera_I_filtrada: pd.DataFrame):
    """
    Convierte Cartera I a clases aptas para Asesoramiento Independiente, manteniendo mismas características.
    Orden de búsqueda:
      1) AI + Transferable == 'Yes'
      2) T  + Transferable == 'Yes' y MiFID FH 'clean'
      3) Cualquier clase del MISMO Family Name cuyo Name contenga 'Cartera'
    Copia el VALOR ACTUAL (EUR) del fondo original y recalcula pesos por valor solo con los transformados.
    """
    clean_set = {"clean", "clean institucional", "clean institutional"}
    out_rows = []
    incidencias = []

    def _name_has_cartera(row):
        for col in ("Name", "Share Class Name", "Fund Name"):
            v = row.get(col, None)
            if pd.notna(v) and "cartera" in str(v).lower():
                return True
        return False

    for _, row in df_cartera_I_filtrada.iterrows():
        fam = row.get("Family Name")
        tos = row.get("Type of Share")
        cur = row.get("Currency")
        hed = row.get("Hedged")
        valor = row.get("VALOR ACTUAL (EUR)")

        # Subconjunto por mismas características
        subset = df_master[
            (df_master["Family Name"] == fam) &
            (df_master["Type of Share"] == tos) &
            (df_master["Currency"] == cur) &
            (df_master["Hedged"] == hed)
        ]

        chosen = None

        # 1) AI + Transferable Yes
        ai = subset[subset["Prospectus AF"].apply(lambda x: _has_code(x, "AI"))]
        if "Transferable" in ai.columns:
            ai = ai[ai["Transferable"].fillna("").astype(str).str.strip().str.lower() == "yes"]
        if not ai.empty:
            chosen = ai

        # 2) T + Transferable Yes + MiFID FH clean
        if chosen is None or chosen.empty:
            t = subset[subset["Prospectus AF"].apply(lambda x: _has_code(x, "T"))]
            if "Transferable" in t.columns:
                t = t[t["Transferable"].fillna("").astype(str).str.strip().str.lower() == "yes"]
            if "MiFID FH" in t.columns:
                t = t[t["MiFID FH"].fillna("").astype(str).str.lower().isin(clean_set)]
            if not t.empty:
                chosen = t

        # 3) Name que contenga 'Cartera' dentro del MISMO Family Name
        if chosen is None or chosen.empty:
            fam_pool = df_master[df_master["Family Name"] == fam].copy()
            cartera_pool = fam_pool[fam_pool.apply(_name_has_cartera, axis=1)]
            if not cartera_pool.empty:
                chosen = cartera_pool

        # Si no hay clase apta
        if chosen is None or chosen.empty:
            incidencias.append(
                (row.get("Name", "(sin nombre)"),
                 "Para este fondo no se ha encontrado una clase que cumpla los criterios de Asesoramiento Independiente.")
            )
            continue

        # Elegir la de menor Ongoing Charge
        best = chosen.sort_values("Ongoing Charge", na_position="last").iloc[0]

        # 🔎 Validación adicional: Min. Initial superior a 10 millones (heurística por longitud de texto)
        min_initial = str(best.get("Min. Initial", "")).strip()
        if len(min_initial) > 11:
            nombre_best = (
                best.get("Name")
                or best.get("Share Class Name")
                or best.get("Fund Name")
                or best.get("Family Name")
                or "(sin nombre)"
            )
            incidencias.append(
                (nombre_best, f"El mínimo inicial de contratación del fondo '{nombre_best}' es de '{min_initial}', consultar.")
            )

        out_rows.append({
            "ISIN": best.get("ISIN", ""),
            "Name": best.get("Name") or best.get("Share Class Name") or best.get("Fund Name") or best.get("Family Name"),
            "Type of Share": best.get("Type of Share", ""),
            "Currency": best.get("Currency", ""),
            "Hedged": best.get("Hedged", ""),
            "Ongoing Charge": best.get("Ongoing Charge", np.nan),
            "Min. Initial": best.get("Min. Initial", ""),
            "MiFID FH": best.get("MiFID FH", ""),
            "MiFID EMT": best.get("MiFID EMT") or best.get("MIFID EMT", ""),
            "Prospectus AF": best.get("Prospectus AF", ""),
            "Soft Close": best.get("Soft Close", ""),
            "Subscription Fee": best.get("Subscription Fee", ""),
            "Redemption Fee": best.get("Redemption Fee", ""),
            "VALOR ACTUAL (EUR)": valor
        })

    df_out = pd.DataFrame(out_rows)
    if df_out.empty:
        return df_out, incidencias

    # Recalcular pesos por valor SOLO con los que tienen OC; el resto queda con peso 0
    df_out = recalcular_pesos_por_valor_respetando_oc(df_out, valor_col="VALOR ACTUAL (EUR)")
    return df_out, incidencias

if st.button("🔁 Convertir a cartera Asesoramiento Independiente"):
    df_II, incid_AI = convertir_a_AI(df_master, df_I)  # df_I = solo ISIN en AllFunds
    st.session_state.cartera_II = {"table": df_II, "ter": calcular_ter_por_valor(df_II)}
    st.session_state.incidencias = st.session_state.get("incidencias", []) + incid_AI

# Mostrar Cartera II si existe
if st.session_state.cartera_II and st.session_state.cartera_II["table"] is not None:
    dfII = st.session_state.cartera_II["table"]
    if not dfII.empty:
        mostrar_tabla_con_formato(dfII, "Tabla Cartera II (AI, solo transformados, pesos por valor)")
        st.metric("📊 TER Cartera II (AI)", _fmt_ratio_eu_percent(st.session_state.cartera_II["ter"], 2) if st.session_state.cartera_II["ter"] is not None else "-")
    else:
        st.info("No se pudieron transformar fondos a AI con los criterios dados.")

# =========================
# 5) Comparación: SOLO fondos transformados
# =========================
st.subheader("Paso 4: Comparar Cartera I vs Cartera II (solo fondos transformados)")
if st.session_state.cartera_II and st.session_state.cartera_II["table"] is not None and not st.session_state.cartera_II["table"].empty:
    dfII = st.session_state.cartera_II["table"].copy()
    # Tomamos los Family Name de los fondos transformados
    fams_II = set(
        pd.merge(dfII[["ISIN"]], df_master[["ISIN","Family Name"]].assign(ISIN=lambda s: s["ISIN"].astype(str).str.upper()),
                 how="left", on="ISIN")["Family Name"].dropna().astype(str)
    )

    # Filtramos Cartera I RAW (ya match ISIN en maestro) a esos Family Name
    fams_master = df_master[["ISIN","Family Name"]].copy()
    fams_master["ISIN"] = fams_master["ISIN"].astype(str).str.upper()
    dfI_raw = st.session_state.cartera_I_raw.copy()
    dfI_raw["ISIN"] = dfI_raw["ISIN"].astype(str).str.upper()
    dfI_raw = pd.merge(dfI_raw, fams_master, on="ISIN", how="left", suffixes=("","_fam"))
    dfI_sub = dfI_raw[dfI_raw["Family Name_fam"].astype(str).isin(fams_II)].copy()

    # Recalcular pesos y TER en el subset de I y usar II tal cual
    dfI_sub = recalcular_pesos_por_valor_respetando_oc(dfI_sub, valor_col="VALOR ACTUAL (EUR)")
    ter_I_sub = calcular_ter_por_valor(dfI_sub)
    ter_II_sub = st.session_state.cartera_II["ter"]  # ya viene ponderado por valor en el subset transformado

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Cartera I (solo fondos con transformación AI)")
        mostrar_tabla_con_formato(dfI_sub, "Tabla Cartera I (subset comparable)")
        st.metric("TER medio ponderado", _fmt_ratio_eu_percent(ter_I_sub, 2) if ter_I_sub is not None else "-")
    with c2:
        st.markdown("#### Cartera II (AI)")
        mostrar_tabla_con_formato(dfII, "Tabla Cartera II (AI)")
        st.metric("TER medio ponderado", _fmt_ratio_eu_percent(ter_II_sub, 2) if ter_II_sub is not None else "-")

    if ter_I_sub is not None and ter_II_sub is not None:
        st.markdown("---")
        st.subheader("Diferencia de TER (II − I) en fondos transformados")
        st.metric("Diferencia", _fmt_ratio_eu_percent(ter_II_sub - ter_I_sub, 2))
else:
    st.info("Primero convierte a Cartera II para ver la comparativa.")

# =========================
# 6) Incidencias
# =========================
if st.session_state.get("incidencias"):
    st.subheader("⚠️ Incidencias detectadas")
    for fam, msg in st.session_state.incidencias:
        st.error(f"{fam}: {msg}")


