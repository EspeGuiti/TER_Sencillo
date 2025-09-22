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
    """
    Devuelve SOLO las columnas pedidas del maestro + 'Weight %'.
    'MIFID FH' se muestra con ese nombre (mapeando desde 'MiFID FH' del maestro).
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
        "Weight %",  # ⬅️ añadido (y se mantiene en todas las tablas)
    ]

    for c in wanted:
        if c not in tbl.columns:
            tbl[c] = np.nan

    return tbl[wanted]
def _to_float_eu_money(x):
    """Convierte strings de dinero en formato EU a float. Admite números ya float."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x) if float(x) > 0 else np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace("€", "").replace("EUR", "").replace(" ", "")
    # quita separador miles europeo y deja '.' decimal
    # ojo: si llega como 1.234,56 -> 1234.56
    s = s.replace(".", "").replace(",", ".")
    try:
        v = float(s)
        return v if v > 0 else np.nan
    except Exception:
        return np.nan


def _recalcular_por_valor_y_agrupar_por_nombre(df, *, valor_col="VALOR ACTUAL (EUR)", nombre_col="Name", require_oc=True):
    """
    Devuelve (ter_ponderado, df_out)
    - Filtra a filas elegibles: valor>0 y (si require_oc) Ongoing Charge notna
    - Agrupa por 'nombre_col', suma VALOR y calcula Weight % = valor / total * 100
    - TER ponderado por esos Weight %
    """
    if valor_col not in df.columns:
        return None, df.head(0).copy()

    # Normaliza valor actual
    vals = df[valor_col].apply(_to_float_eu_money)
    df2 = df.assign(**{valor_col: vals})

    elig = df2[valor_col].notna()
    if require_oc:
        elig = elig & df2["Ongoing Charge"].astype(float).notna()

    df_e = df2.loc[elig].copy()
    if df_e.empty:
        return None, df_e

    # Si no existe 'Name', usamos 'Family Name'
    if nombre_col not in df_e.columns:
        if "Family Name" in df_e.columns:
            nombre_col = "Family Name"
        else:
            # último recurso: mantiene filas tal cual
            nombre_col = None

    if nombre_col is not None:
        # agrupamos por nombre y dejamos un OC representativo (si hay varios, media simple del OC)
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

    # Orden estético
    cols = [c for c in [nombre_col, valor_col, "Weight %", "Ongoing Charge"] if c is not None]
    agg = agg[cols].sort_values(by="Weight %", ascending=False)

    return ter, agg

def convertir_a_AI(df_master: pd.DataFrame, df_cartera_I: pd.DataFrame):
    """
    Genera Cartera II con el MISMO ORDEN y MISMO Nº de filas que Cartera I.
    Regla:
      1) Buscar clase con Prospectus AF que contenga 'AI' y Transferable == 'Yes'
      2) Si no hay, buscar clase con 'T' y Transferable == 'Yes' y MiFID FH 'clean'
      3) Elegir siempre la de menor Ongoing Charge
    Además:
      - Si la clase elegida tiene 'Transferable' en blanco -> añadir incidencia (pero dejarlo en blanco).
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
        w   = row.get("Weight %", 0.0)

        # Filtrar mismas características básicas
        subset = df_master[
            (df_master["Family Name"] == fam) &
            (df_master["Type of Share"] == tos) &
            (df_master["Currency"] == cur) &
            (df_master["Hedged"] == hed)
        ]

        # --- 1) AI + Transferable == Yes (blancos se mantienen como "")
        ai_match = subset[subset["Prospectus AF"].apply(lambda x: _has_code(x, "AI"))]
        if "Transferable" in ai_match.columns:
            ai_match = ai_match.copy()
            ai_match["Transferable"] = ai_match["Transferable"].fillna("").astype(str).str.strip()
            ai_match_yes = ai_match[ai_match["Transferable"].str.lower() == "yes"]
        else:
            ai_match_yes = ai_match

        chosen = None
        match_type = ""

        if not ai_match_yes.empty:
            chosen = ai_match_yes
            match_type = "AI"
        else:
            # --- 2) T + Transferable == Yes + MiFID FH clean
            t_match = subset[subset["Prospectus AF"].apply(lambda x: _has_code(x, "T"))]
            if "Transferable" in t_match.columns:
                t_match = t_match.copy()
                t_match["Transferable"] = t_match["Transferable"].fillna("").astype(str).str.strip()
                t_match_yes = t_match[t_match["Transferable"].str.lower() == "yes"]
            else:
                t_match_yes = t_match

            if "MiFID FH" in t_match_yes.columns:
                # tolerante a NaN antes de lower()
                t_match_yes = t_match_yes.copy()
                t_match_yes["MiFID FH"] = t_match_yes["MiFID FH"].fillna("").astype(str)
                t_match_yes = t_match_yes[t_match_yes["MiFID FH"].str.lower().isin(clean_set)]

            if not t_match_yes.empty:
                chosen = t_match_yes
                match_type = "T Clean"

        found_cartera = chosen is not None and not chosen.empty

        if not found_cartera:
            incidencias.append(
                (str(fam),
                 "Sin clase AI ni T (Clean) transferible con misma (Type of Share/Currency/Hedged) ni clase 'cartera'")
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
                "Weight %":      float(w) if pd.notnull(w) else 0.0
            })
            continue

        # Elegir la de menor Ongoing Charge
        chosen = chosen.sort_values("Ongoing Charge", na_position="last")
        best = chosen.iloc[0]

        name_val = (
            best.get("Name")
            or best.get("Share Class Name")
            or best.get("Fund Name")
            or best.get("Family Name")
        )
        emt_val  = best.get("MiFID EMT") or best.get("MIFID EMT")

        # --- Avisos y normalizaciones auxiliares
        sub_fee = best.get("Subscription Fee", 0)
        red_fee = best.get("Redemption Fee", 0)
        soft_close = str(best.get("Soft Close", "")).strip().lower()

        # Aviso si 'Transferable' viene en blanco en la clase elegida
        if "Transferable" in chosen.columns:
            tf_value = str(best.get("Transferable", "")).strip()
            if tf_value == "":
                incidencias.append(
                    (str(fam), "El campo 'Transferable' viene EN BLANCO en la clase seleccionada.")
                )

        def fee_to_float(v):
            if pd.isna(v): return 0.0
            s = str(v).replace("%", "").replace(",", ".")
            try:
                return float(s)
            except Exception:
                return 0.0

        sub_fee_f = fee_to_float(sub_fee)
        red_fee_f = fee_to_float(red_fee)

        if sub_fee_f > 0:
            incidencias_fees.append((name_val, f"Subscription Fee es {sub_fee}"))
        if red_fee_f > 0:
            incidencias_fees.append((name_val, f"Redemption Fee es {red_fee}"))
        if soft_close == "yes":
            incidencias_soft.append((name_val, "Soft Close está marcado como 'Yes'"))

        # Fila resultado
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
            "Weight %":       float(w) if pd.notnull(w) else 0.0
        })

    # Salida
    df_result = pd.DataFrame(results)

    # Filtrar incidencias que no quieres mostrar
    incidencias_finales = [
        (fam, msg)
        for fam, msg in (incidencias + incidencias_fees + incidencias_soft)
        if not msg.startswith("Sin clase AI ni T (Clean) transferible")
    ]
    return df_result, incidencias_finales

def mostrar_tabla_con_formato(df_in, title):
    st.markdown(f"#### {title}")
    df_show = pretty_table(df_in).copy()

    # Formateo europeo -> convertir a TEXTO con coma decimal
    def _fmt_eu(v, dec):
        if pd.isna(v):
            return ""
        try:
            # aceptar '1,23', '1.23', '1,23%', etc.
            x = float(str(v).replace("%", "").replace(",", "."))
        except Exception:
            return str(v)
        s = f"{x:,.{dec}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")

    if "Ongoing Charge" in df_show.columns:
        df_show["Ongoing Charge"] = df_show["Ongoing Charge"] \
            .apply(lambda v: _fmt_eu(v, 4)) \
            .astype(str)

    if "Weight %" in df_show.columns:
        df_show["Weight %"] = df_show["Weight %"] \
            .apply(lambda v: _fmt_eu(v, 2)) \
            .astype(str)
        # Si quieres enseñar el símbolo % en tabla, usa:
        # df_show["Weight %"] = df_show["Weight %"].apply(lambda s: f"{s}%" if s else s)

    st.dataframe(df_show, use_container_width=True)

def _has_code(s: str, code: str) -> bool:
    """
    Devuelve True si 'code' aparece como token en 'Prospectus AF'.
    Ej: "I+GDC+AI+AP" → tokens ["I","GDC","AI","AP"]
    """
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
def _find_col(df, logical_name: str):
    """
    Devuelve el nombre REAL de la columna cuyo nombre lógico (sin espacios/case) coincide.
    Ej.: _find_col(df, 'transferable') -> 'Transferable' o 'Transferable ' si venía con espacios.
    """
    target = logical_name.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == target:
            return c
    return None

def _norm_str_blank(x):
    """Convierte NaN/None/espacios a cadena vacía; deja el resto tal cual."""
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s == "" else s

# =========================
# 1) Subida de archivos
# =========================
st.subheader("Paso 1: Subir ficheros")
master_file = st.file_uploader("📥 Sube el Excel Completo de AllFunds Share Class Tool (con todas las clases)", type=["xlsx"], key="master")
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

# --- Cargar Excel de CARTERA en crudo, sin asumir cabeceras ---
try:
    df_any = pd.read_excel(weights_file, header=None)
except Exception as e:
    st.error(f"No se pudo leer la cartera: {e}")
    st.stop()

# Objetivos normalizados (acepta variantes con/ sin acentos y mayúsculas)
targets_isin = {"isin"}
targets_weight = {"total inversion", "total inversion."}  # por si lleva punto u otra variante menor

r_isin, c_isin = _find_header_cell(df_any, targets_isin)
r_wgt,  c_wgt  = _find_header_cell(df_any, targets_weight)

if r_isin is None or r_wgt is None:
    st.error("No se han encontrado los encabezados 'ISIN' y/o 'TOTAL INVERSIÓN' en el fichero de cartera.")
    st.stop()

# Leemos hacia abajo desde la fila siguiente al encabezado, en esas columnas
col_isin_vals  = df_any.iloc[r_isin+1:, c_isin].reset_index(drop=True)
col_wgt_vals   = df_any.iloc[r_wgt+1:,  c_wgt].reset_index(drop=True)

# Emparejamos por índice (hasta la longitud máxima de ambas)
max_len = max(len(col_isin_vals), len(col_wgt_vals))
col_isin_vals = col_isin_vals.reindex(range(max_len))
col_wgt_vals  = col_wgt_vals.reindex(range(max_len))

df_weights_raw = pd.DataFrame({
    "ISIN": col_isin_vals,
    "Peso %": col_wgt_vals
})

# Quitar filas totalmente vacías
df_weights_raw.dropna(how="all", inplace=True)

# Excluir la fila de totales (100) que viene sin ISIN
# (y en general, cualquier fila sin ISIN no nos sirve)
df_weights_raw = df_weights_raw[df_weights_raw["ISIN"].notna()].copy()

# Normalizar pesos (acepta '16,61', '16.61', '16,61%')
df_weights = _clean_weights(df_weights_raw)

# Sumar duplicados por ISIN (y normalizar formato del ISIN por seguridad)
df_weights["ISIN"] = df_weights["ISIN"].astype(str).str.strip().str.upper()
df_weights = df_weights.groupby("ISIN", as_index=False)["Peso %"].sum()

def merge_cartera_con_maestro(df_master, df_weights):
    """
    Une la cartera (df_weights) con el maestro (df_master) por ISIN.
    Devuelve el DataFrame combinado y una lista de incidencias para ISINs no encontrados.
    """
    # Asegura que ISIN está en mayúsculas en ambos
    df_master = df_master.copy()
    df_weights = df_weights.copy()
    df_master["ISIN"] = df_master["ISIN"].astype(str).str.strip().str.upper()
    df_weights["ISIN"] = df_weights["ISIN"].astype(str).str.strip().str.upper()

    # Merge
    df_merged = pd.merge(df_weights, df_master, how="left", on="ISIN", suffixes=('', '_m'))

    # Genera incidencias para ISINs no encontrados
    incidencias = []
    for idx, row in df_merged.iterrows():
        if pd.isna(row.get("Family Name", None)):
            incidencias.append((row["ISIN"], "ISIN no encontrado en el maestro"))

    # Renombra 'Peso %' a 'Weight %' para consistencia interna
    if "Peso %" in df_merged.columns:
        df_merged = df_merged.rename(columns={"Peso %": "Weight %"})

    return df_merged, incidencias

# =========================
# 3) Cartera I (original) + TER
# =========================
st.subheader("Paso 2: Calcular Cartera I (original)")

df_I_raw, incidencias_merge = merge_cartera_con_maestro(df_master, df_weights)

# Mostrar suma de pesos y advertencia si no es 100
peso_total_I = df_I_raw["Weight %"].sum() if "Weight %" in df_I_raw.columns else df_I_raw["Peso %"].sum()
st.write(f"**Suma de pesos cartera (I):** {_format_eu_number(peso_total_I, 2)}%")
if abs(peso_total_I - 100.0) > 1e-6:
    st.warning("La suma de pesos no es 100%. Corrige tu Excel de cartera.")

def calcular_ter(df):
    """Calcula el TER medio ponderado."""
    if "Weight %" not in df.columns or "Ongoing Charge" not in df.columns:
        return None
    pesos = df["Weight %"].astype(float)
    ter = df["Ongoing Charge"].astype(float)
    if pesos.sum() == 0:
        return None
    return np.nansum(pesos * ter) / pesos.sum()

# =========================
# 4) TER y tabla (lógica final)
# =========================
ter_I = calcular_ter(df_I_raw.rename(columns={"Peso %": "Weight %"}))
st.session_state.cartera_I = {"table": df_I_raw.rename(columns={"Peso %": "Weight %"}), "ter": ter_I}

mostrar_tabla_con_formato(st.session_state.cartera_I["table"], "Tabla Cartera I (original)")
if st.session_state.cartera_I["ter"] is not None:
    st.metric("📊 TER Cartera I", _fmt_ratio_eu_percent(st.session_state.cartera_I["ter"], 2))

# Incidencias de merge
incidencias = list(incidencias_merge)

# =========================
# 5) Convertir a Cartera II (AI)
# =========================
st.subheader("Paso 3: Convertir a Cartera de Asesoramiento Independiente (Cartera II)")
st.caption(
    "Se mantiene Type of Share, Currency, Hedged; Transferable = 'Yes'. "
    "Prioridad: 'AI' en Prospectus AF; si no hay, 'T' con MiFID FH = Clean/Clean Institucional. "
    "Siempre se elige la menor Ongoing Charge."
)

if st.button("🔁 Convertir a cartera Asesoramiento Independiente"):
    df_II, incid_AI = convertir_a_AI(df_master, st.session_state.cartera_I['table'])
    st.session_state.cartera_II = {"table": df_II, "ter": calcular_ter(df_II)}
    incidencias.extend(incid_AI)

# Mostrar Cartera II si existe
if st.session_state.cartera_II and not st.session_state.cartera_II["table"].empty:
    mostrar_tabla_con_formato(st.session_state.cartera_II["table"], "Tabla Cartera II (AI)")
    if st.session_state.cartera_II["ter"] is not None:
        st.metric("📊 TER Cartera II (AI)", _fmt_ratio_eu_percent(st.session_state.cartera_II["ter"], 2))

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
        st.markdown("#### Cartera I")
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
if incidencias:
    st.subheader("⚠️ Incidencias detectadas")
    for fam, msg in incidencias:
        st.error(f"{fam}: {msg}")
