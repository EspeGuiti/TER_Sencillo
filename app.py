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
    st.session_state.cartera_I = None  # {"table": df, "ter": float}
if "cartera_I_raw" not in st.session_state:
    st.session_state.cartera_I_raw = None  # df mergeado completo (para convertir a AI)
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

def _to_float_eu_money(x):
    """Convierte strings de dinero en formato EU a float (>0) o NaN."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x) if float(x) > 0 else np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace("€", "").replace("EUR", "").replace(" ", "")
    s = s.replace(".", "").replace(",", ".")  # 1.234,56 -> 1234.56
    try:
        v = float(s)
        return v if v > 0 else np.nan
    except Exception:
        return np.nan

def _clean_master(dfm):
    """Limpia columnas críticas del maestro."""
    df = dfm.copy()
    if "Ongoing Charge" in df.columns:
        df["Ongoing Charge"] = df["Ongoing Charge"].apply(_to_float_percent_like)
    # Normalizar Transferable -> deja blancos como "", respeta Yes/No si vienen
    if "Transferable" in df.columns:
        def norm_tf(v):
            if pd.isna(v): return ""
            s = str(v).strip().lower()
            if s in {"yes", "y", "true", "1"}: return "Yes"
            if s in {"no", "n", "false", "0"}: return "No"
            return str(v)  # dejar tal cual (incluye blanco)
        df["Transferable"] = df["Transferable"].apply(norm_tf)
    return df

def _format_eu_number(x, decimals=4):
    if pd.isna(x):
        return x
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def pretty_table(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve columnas principales + 'Weight %' + 'VALOR ACTUAL (EUR)' si existe.
    """
    tbl = df_in.copy()

    if "MiFID FH" in tbl.columns and "MIFID FH" not in tbl.columns:
        tbl.rename(columns={"MiFID FH": "MIFID FH"}, inplace=True)

    wanted = [
        "ISIN",
        "Name",
        "Type of Share",
        "Currency",
        "Hedged",
        "Ongoing Charge",
        "Min. Initial",
        "MIFID FH",
        "MiFID EMT",
        "Prospectus AF",
        "Soft Close",
        "Subscription Fee",
        "Redemption Fee",
        "VALOR ACTUAL (EUR)",   # visible
        "Weight %",             # pesos calculados por valor
    ]
    for c in wanted:
        if c not in tbl.columns:
            tbl[c] = np.nan

    return tbl[wanted]

def _recalcular_por_valor_y_agrupar_por_nombre(df, *, valor_col="VALOR ACTUAL (EUR)", nombre_col="Name", require_oc=True):
    """
    Devuelve (ter_ponderado, df_out)
    - Filtra a filas elegibles: valor>0 y (si require_oc) Ongoing Charge notna
    - Agrupa por 'nombre_col' (o Family Name si no existe), suma VALOR y calcula Weight % = valor / total * 100
    - TER ponderado por esos Weight %
    """
    if valor_col not in df.columns:
        return None, df.head(0).copy()

    vals = df[valor_col].apply(_to_float_eu_money)
    df2 = df.assign(**{valor_col: vals})

    elig = df2[valor_col].notna()
    if require_oc:
        elig = elig & df2["Ongoing Charge"].astype(float).notna()

    df_e = df2.loc[elig].copy()
    if df_e.empty:
        return None, df_e

    # Nombre preferente
    if nombre_col not in df_e.columns:
        if "Family Name" in df_e.columns:
            nombre_col = "Family Name"
        else:
            nombre_col = None

    if nombre_col is not None:
        agg = df_e.groupby(nombre_col, dropna=False).agg({
            valor_col: "sum",
            "Ongoing Charge": "mean"
        }).reset_index()
    else:
        agg = df_e.copy()

    total_val = agg[valor_col].sum()
    if total_val <= 0:
        return None, agg.head(0).copy()

    agg["Weight %"] = (agg[valor_col] / total_val) * 100.0

    # TER ponderado por valor
    ter = np.nansum(agg["Ongoing Charge"].astype(float) * agg["Weight %"]) / agg["Weight %"].sum()

    # Orden y columnas
    cols = [c for c in [nombre_col, valor_col, "Weight %", "Ongoing Charge"] if c is not None]
    agg = agg[cols].sort_values(by="Weight %", ascending=False)

    return ter, agg

def _has_code(s: str, code: str) -> bool:
    """Comprueba si 'code' aparece como token en 'Prospectus AF'."""
    if pd.isna(s):
        return False
    tokens = re.split(r'[^A-Za-z0-9]+', str(s).upper())
    return code.upper() in tokens

def _fmt_ratio_eu_percent(x, decimals=2):
    """Formatea un ratio (p.ej. 0.0123) como % europeo '1,23%'."""
    if x is None:
        return "-"
    return f"{x:.{decimals}%}".replace(".", ",")

def _norm_txt(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower()

def _find_header_cell(df_any, targets):
    """
    Busca la primera celda cuyo texto normalizado coincide exactamente con alguno de targets.
    targets: conjunto/lista de strings normalizados (lowercase, sin acentos).
    Devuelve (row_idx, col_idx) o (None, None) si no encuentra.
    """
    for r in range(df_any.shape[0]):
        for c in range(df_any.shape[1]):
            if _norm_txt(df_any.iat[r, c]) in targets:
                return r, c
    return None, None

def mostrar_tabla_con_formato(df_in, title):
    st.markdown(f"#### {title}")
    df_show = pretty_table(df_in).copy()

    # Formateo europeo para columnas clave
    def _fmt_eu(v, dec):
        if pd.isna(v):
            return ""
        try:
            x = float(str(v).replace("%", "").replace(",", "."))
        except Exception:
            return str(v)
        s = f"{x:,.{dec}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")

    if "Ongoing Charge" in df_show.columns:
        df_show["Ongoing Charge"] = df_show["Ongoing Charge"].apply(lambda v: _fmt_eu(v, 4)).astype(str)

    if "Weight %" in df_show.columns:
        df_show["Weight %"] = df_show["Weight %"].apply(lambda v: _fmt_eu(v, 2)).astype(str)

    if "VALOR ACTUAL (EUR)" in df_show.columns:
        df_show["VALOR ACTUAL (EUR)"] = df_show["VALOR ACTUAL (EUR)"].apply(lambda v: _fmt_eu(v, 2)).astype(str)

    st.dataframe(df_show, use_container_width=True)

# =========================
# 1) Subida de archivos
# =========================
st.subheader("Paso 1: Subir ficheros")
master_file = st.file_uploader(
    "📥 Sube el Excel Completo de AllFunds Share Class Tool (con todas las clases)",
    type=["xlsx"], key="master"
)
weights_file = st.file_uploader(
    "📥 Sube el Excel de CARTERA (columnas 'ISIN' y 'VALOR ACTUAL (EUR)')",
    type=["xlsx"], key="weights"
)

if not master_file or not weights_file:
    st.info("Sube ambos ficheros para continuar.")
    st.stop()

# =========================
# 2) Carga y validaciones
# =========================

# Maestro
df_master_raw = pd.read_excel(master_file, skiprows=2)

required_cols = [
    "Family Name","Type of Share","Currency","Hedged",
    "MiFID FH","Min. Initial","Ongoing Charge","ISIN","Prospectus AF"
]
missing = [c for c in required_cols if c not in df_master_raw.columns]
if missing:
    st.error(f"Faltan columnas en el maestro: {missing}")
    st.stop()

st.success("Fichero maestro importado correctamente.")

# Limpieza Ongoing Charge (como tenías)
df_master_raw["Ongoing Charge"] = (
    df_master_raw["Ongoing Charge"].astype(str)
        .str.replace("%","", regex=False)
        .str.replace(",",".", regex=False)
        .astype(float)
)
df_master = _clean_master(df_master_raw)

if "Transferable" not in df_master_raw.columns:
    st.warning("El maestro no tiene columna 'Transferable'. Los blancos se mantendrán como '' y solo filtrará 'Yes' cuando exista.")

# --- Cargar Excel de CARTERA en crudo, sin asumir cabeceras ---
try:
    df_any = pd.read_excel(weights_file, header=None)
except Exception as e:
    st.error(f"No se pudo leer la cartera: {e}")
    st.stop()

# Solo buscamos ISIN y VALOR ACTUAL (EUR)
targets_isin  = {"isin"}
targets_valor = {"valor actual (eur)"}  # nombre exacto confirmado

r_isin,  c_isin  = _find_header_cell(df_any, targets_isin)
r_valor, c_valor = _find_header_cell(df_any, targets_valor)

if r_isin is None or r_valor is None:
    st.error("No se han encontrado los encabezados 'ISIN' y/o 'VALOR ACTUAL (EUR)'.")
    st.stop()

# Leemos desde la fila siguiente al encabezado
col_isin_vals  = df_any.iloc[r_isin+1:,  c_isin].reset_index(drop=True)
col_valor_vals = df_any.iloc[r_valor+1:, c_valor].reset_index(drop=True)

# Quitar filas totalmente vacías al final (ambas columnas vacías)
while len(col_isin_vals) > 0 and pd.isna(col_isin_vals.iloc[-1]) and pd.isna(col_valor_vals.iloc[-1]):
    col_isin_vals  = col_isin_vals.iloc[:-1]
    col_valor_vals = col_valor_vals.iloc[:-1]

# Si la última fila de VALOR ACTUAL (EUR) es el total (suele venir sin ISIN), descártala
if len(col_valor_vals) > 0 and (len(col_isin_vals) == 0 or pd.isna(col_isin_vals.iloc[-1])):
    col_valor_vals = col_valor_vals.iloc[:-1]
    if len(col_isin_vals) > len(col_valor_vals):
        col_isin_vals = col_isin_vals.iloc[:len(col_valor_vals)]

# Alinear longitudes por seguridad
min_len = min(len(col_isin_vals), len(col_valor_vals))
col_isin_vals  = col_isin_vals.iloc[:min_len]
col_valor_vals = col_valor_vals.iloc[:min_len]

# Construir DF de cartera con las dos columnas
df_weights_raw = pd.DataFrame({
    "ISIN": col_isin_vals,
    "VALOR ACTUAL (EUR)": col_valor_vals
})

# Limpieza básica
df_weights_raw.dropna(how="all", inplace=True)
df_weights_raw = df_weights_raw[df_weights_raw["ISIN"].notna()].copy()

# Normaliza VALOR ACTUAL (EUR) al formato numérico y elimina no positivos
df_weights_raw["VALOR ACTUAL (EUR)"] = df_weights_raw["VALOR ACTUAL (EUR)"].apply(_to_float_eu_money)
df_weights_raw = df_weights_raw[df_weights_raw["VALOR ACTUAL (EUR)"].notna()]

# Consolidar duplicados por ISIN: SUMA del valor actual (los pesos originales ya no se usan)
df_weights_raw["ISIN"] = df_weights_raw["ISIN"].astype(str).str.strip().str.upper()
df_weights = df_weights_raw.groupby("ISIN", as_index=False)["VALOR ACTUAL (EUR)"].sum()

def merge_cartera_con_maestro(df_master, df_weights):
    """
    Une la cartera con el maestro por ISIN.
    Devuelve (df_merged, incidencias).
    """
    df_master = df_master.copy()
    df_weights = df_weights.copy()
    df_master["ISIN"] = df_master["ISIN"].astype(str).str.strip().str.upper()
    df_weights["ISIN"] = df_weights["ISIN"].astype(str).str.strip().str.upper()

    df_merged = pd.merge(df_weights, df_master, how="left", on="ISIN", suffixes=('', '_m'))

    incidencias = []
    for _, row in df_merged.iterrows():
        if pd.isna(row.get("Family Name", None)):
            incidencias.append((row["ISIN"], "ISIN no encontrado en el maestro"))

    return df_merged, incidencias

# =========================
# 3) Cartera I (original) + TER REAL por VALOR ACTUAL
# =========================
st.subheader("Paso 2: Calcular Cartera I (original)")

df_I_raw, incidencias_merge = merge_cartera_con_maestro(df_master, df_weights)

# Guardamos la mergeada cruda (con Family/Share/Currency/Hedged/Valor) para convertir a AI
st.session_state.cartera_I_raw = df_I_raw.copy()

# TER real y tabla: SOLO fondos con OC y ponderados por VALOR ACTUAL (EUR), agrupando por Name
ter_I, tabla_I = _recalcular_por_valor_y_agrupar_por_nombre(
    df_I_raw, valor_col="VALOR ACTUAL (EUR)", nombre_col="Name", require_oc=True
)
st.session_state.cartera_I = {"table": tabla_I, "ter": ter_I}

# Mostrar tabla I
mostrar_df = st.session_state.cartera_I["table"].copy()
st.markdown(f"**Fondos usados (Cartera I) para TER real:** {len(mostrar_df)}")
mostrar_tabla_con_formato(mostrar_df, "Tabla Cartera I (filtrada y ponderada por VALOR ACTUAL)")

if st.session_state.cartera_I["ter"] is not None:
    st.metric("📊 TER Cartera I (real por valor)", _fmt_ratio_eu_percent(st.session_state.cartera_I["ter"], 2))
else:
    st.warning("No se pudo calcular el TER de Cartera I (no hay Ongoing Charge y/o VALOR ACTUAL (EUR) válido).")

# Incidencias de merge
incidencias = list(incidencias_merge)

# =========================
# 5) Convertir a Cartera II (AI)
# =========================
def convertir_a_AI(df_master: pd.DataFrame, df_cartera_I: pd.DataFrame):
    """
    Convierte Cartera I a clases AI/T (Clean).
    Reglas:
      - Mantener Type of Share, Currency, Hedged
      - Prioridad: AI + Transferable=='Yes'; si no, T + Transferable=='Yes' + MiFID FH clean
      - Elegir menor Ongoing Charge
    Devuelve (df_AI_filtrado_agrupado, incidencias)
    """
    results = []
    incidencias = []

    clean_set = {"clean", "clean institucional", "clean institutional"}
    incidencias_fees = []
    incidencias_soft = []

    for _, row in df_cartera_I.iterrows():
        fam = row.get("Family Name")
        tos = row.get("Type of Share")
        cur = row.get("Currency")
        hed = row.get("Hedged")
        valor_actual = row.get("VALOR ACTUAL (EUR)")

        subset = df_master[
            (df_master["Family Name"] == fam) &
            (df_master["Type of Share"] == tos) &
            (df_master["Currency"] == cur) &
            (df_master["Hedged"] == hed)
        ]

        # 1) AI + Transferable==Yes (respetando blancos "")
        ai_match = subset[subset["Prospectus AF"].apply(lambda x: _has_code(x, "AI"))]
        if "Transferable" in ai_match.columns:
            ai_match = ai_match.copy()
            ai_match["Transferable"] = ai_match["Transferable"].fillna("").astype(str).str.strip()
            ai_match_yes = ai_match[ai_match["Transferable"].str.lower() == "yes"]
        else:
            ai_match_yes = ai_match

        chosen = None

        if not ai_match_yes.empty:
            chosen = ai_match_yes
        else:
            # 2) T + Transferable==Yes + MiFID FH clean
            t_match = subset[subset["Prospectus AF"].apply(lambda x: _has_code(x, "T"))]
            if "Transferable" in t_match.columns:
                t_match = t_match.copy()
                t_match["Transferable"] = t_match["Transferable"].fillna("").astype(str).str.strip()
                t_match_yes = t_match[t_match["Transferable"].str.lower() == "yes"]
            else:
                t_match_yes = t_match

            if "MiFID FH" in t_match_yes.columns:
                t_match_yes = t_match_yes.copy()
                t_match_yes["MiFID FH"] = t_match_yes["MiFID FH"].fillna("").astype(str)
                t_match_yes = t_match_yes[t_match_yes["MiFID FH"].str.lower().isin(clean_set)]

            if not t_match_yes.empty:
                chosen = t_match_yes

        if chosen is None or chosen.empty:
            incidencias.append(
                (str(fam),
                 "Sin clase AI ni T (Clean) transferible con misma (Type of Share/Currency/Hedged)")
            )
            results.append({
                "Family Name":   fam,
                "Name":          "",
                "Type of Share": "",
                "Currency":      "",
                "Hedged":        "",
                "MiFID FH":      "",
                "MiFID EMT":     "",
                "Min. Initial":  "",
                "ISIN":          "",
                "Prospectus AF": "",
                "Transferable":  "",
                "Ongoing Charge": np.nan,
                "Soft Close":    "",
                "Subscription Fee": "",
                "Redemption Fee": "",
                "VALOR ACTUAL (EUR)": _to_float_eu_money(valor_actual),
            })
            continue

        # Elegir la de menor Ongoing Charge
        chosen = chosen.sort_values("Ongoing Charge", na_position="last")
        best = chosen.iloc[0]

        # Aviso si 'Transferable' viene en blanco
        if "Transferable" in chosen.columns:
            tf_value = str(best.get("Transferable", "")).strip()
            if tf_value == "":
                incidencias.append((str(fam), "El campo 'Transferable' viene EN BLANCO en la clase seleccionada."))

        def fee_to_float(v):
            if pd.isna(v): return 0.0
            s = str(v).replace("%", "").replace(",", ".")
            try: return float(s)
            except Exception: return 0.0

        sub_fee = best.get("Subscription Fee", 0)
        red_fee = best.get("Redemption Fee", 0)
        soft_close = str(best.get("Soft Close", "")).strip().lower()
        name_val = (
            best.get("Name") or best.get("Share Class Name")
            or best.get("Fund Name") or best.get("Family Name")
        )
        emt_val  = best.get("MiFID EMT") or best.get("MIFID EMT")

        if fee_to_float(sub_fee) > 0:
            incidencias_fees.append((name_val, f"Subscription Fee es {sub_fee}"))
        if fee_to_float(red_fee) > 0:
            incidencias_fees.append((name_val, f"Redemption Fee es {red_fee}"))
        if soft_close == "yes":
            incidencias_soft.append((name_val, "Soft Close está marcado como 'Yes'"))

        results.append({
            "Family Name":    best.get("Family Name", ""),
            "Name":           name_val,
            "Type of Share":  best.get("Type of Share", ""),
            "Currency":       best.get("Currency", ""),
            "Hedged":         best.get("Hedged", ""),
            "MiFID FH":       best.get("MiFID FH", ""),
            "MiFID EMT":      emt_val,
            "Min. Initial":   best.get("Min. Initial", ""),
            "ISIN":           best.get("ISIN", ""),
            "Prospectus AF":  best.get("Prospectus AF", ""),
            "Transferable":   best.get("Transferable", ""),
            "Ongoing Charge": best.get("Ongoing Charge", np.nan),
            "Soft Close":     best.get("Soft Close", ""),
            "Subscription Fee": best.get("Subscription Fee", ""),
            "Redemption Fee":   best.get("Redemption Fee", ""),
            "VALOR ACTUAL (EUR)": _to_float_eu_money(valor_actual),
        })

    df_result = pd.DataFrame(results)

    # Filtrado final: solo con OC y VALOR válido; agrupado por nombre; pesos por valor
    ter_AI, df_AI = _recalcular_por_valor_y_agrupar_por_nombre(
        df_result, valor_col="VALOR ACTUAL (EUR)", nombre_col="Name", require_oc=True
    )

    incidencias_finales = [
        (fam, msg)
        for fam, msg in (incidencias + incidencias_fees + incidencias_soft)
        if not msg.startswith("Sin clase AI ni T (Clean) transferible")
    ]

    return df_AI, incidencias_finales

st.subheader("Paso 3: Convertir a Cartera de Asesoramiento Independiente (Cartera II)")
st.caption(
    "Se mantiene Type of Share, Currency, Hedged; Transferable = 'Yes'. "
    "Prioridad: 'AI' en Prospectus AF; si no hay, 'T' con MiFID FH = Clean/Clean Institucional. "
    "Siempre se elige la menor Ongoing Charge. Ponderación por VALOR ACTUAL (EUR)."
)

# Botón de conversión
if st.button("🔁 Convertir a cartera Asesoramiento Independiente"):
    df_II, incid_AI = convertir_a_AI(df_master, st.session_state.cartera_I_raw)
    # df_II ya viene filtrado y ponderado; TER se puede recomputar por si acaso
    if not df_II.empty:
        ter_II = np.nansum(df_II["Weight %"].astype(float) * df_II["Ongoing Charge"].astype(float)) / df_II["Weight %"].sum()
    else:
        ter_II = None
    st.session_state.cartera_II = {"table": df_II, "ter": ter_II}
    incidencias = st.session_state.get("incidencias", []) + incid_AI
    st.session_state.incidencias = incidencias

# Mostrar Cartera II si existe
if st.session_state.cartera_II and st.session_state.cartera_II["table"] is not None:
    if not st.session_state.cartera_II["table"].empty:
        mostrar_tabla_con_formato(st.session_state.cartera_II["table"], "Tabla Cartera II (AI)")
        if st.session_state.cartera_II["ter"] is not None:
            st.metric("📊 TER Cartera II (AI)", _fmt_ratio_eu_percent(st.session_state.cartera_II["ter"], 2))
    else:
        st.info("No hay fondos con Ongoing Charge y/o VALOR ACTUAL (EUR) válido en Cartera II.")

# =========================
# 6) Comparación I vs II
# =========================
st.subheader("Paso 4: Comparar Cartera I vs Cartera II")
if (
    st.session_state.cartera_I and st.session_state.cartera_I["ter"] is not None and
    st.session_state.cartera_II and st.session_state.cartera_II["ter"] is not None
):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Cartera I (filtrada)")
        st.metric("TER medio ponderado", _fmt_ratio_eu_percent(st.session_state.cartera_I["ter"], 2))
        mostrar_tabla_con_formato(st.session_state.cartera_I["table"], "Tabla Cartera I")

    with c2:
        st.markdown("#### Cartera II (AI)")
        st.metric("TER medio ponderado", _fmt_ratio_eu_percent(st.session_state.cartera_II["ter"], 2))
        mostrar_tabla_con_formato(st.session_state.cartera_II["table"], "Tabla Cartera II (AI)")

    diff = st.session_state.cartera_II["ter"] - st.session_state.cartera_I["ter"]
    st.markdown("---")
    st.subheader("Diferencia de TER (II − I)")
    st.metric("Diferencia", _fmt_ratio_eu_percent(diff, 2))

# =========================
# 7) Incidencias
# =========================
if st.session_state.get("incidencias"):
    st.subheader("⚠️ Incidencias detectadas")
    for fam, msg in st.session_state.incidencias:
        st.error(f"{fam}: {msg}")

