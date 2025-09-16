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

    # Normalizar mayúsculas del MiFID FH (en maestro suele ser 'MiFID FH')
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
        "Weight %",          # ⬅️ añadido (y se mantiene en todas las tablas)
    ]

    # Crear columnas vacías si faltan (por ejemplo, si 'MiFID EMT' no existe en tu maestro)
    for c in wanted:
        if c not in tbl.columns:
            tbl[c] = np.nan

    # Orden y solo las deseadas
    return tbl[wanted]

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
    Genera Cartera II con el MISMO ORDEN y MISMO Nº de filas que Cartera I.
    Reglas:
      - Matching por misma (Family Name, Type of Share, Currency, Hedged)
      - Transferable = 'Yes'
      - MiFID FH ∈ {Clean, Clean Institucional/Institutional}
      - Prioridad: Prospectus AF contiene 'AI'; si no, 'T'
      - Elegir menor Ongoing Charge
    Si no hay candidata, devuelve una fila placeholder:
      Name = Family Name de I, resto de campos en blanco, Weight % igual que I.
    """
    results = []
    incidencias = []

    has_ia_col = "Prospectus AF - Independent Advice (IA)*" in df_master.columns
    clean_set = {"clean", "clean institucional", "clean institutional"}

    for _, row in df_cartera_I.iterrows():
        fam = row.get("Family Name")
        tos = row.get("Type of Share")
        cur = row.get("Currency")
        hed = row.get("Hedged")
        w   = row.get("Weight %", 0.0)

        # 1) Candidatas base: misma familia y misma clase (ToS/Currency/Hedged)
        base = df_master[
            (df_master["Family Name"] == fam) &
            (df_master["Type of Share"] == tos) &
            (df_master["Currency"] == cur) &
            (df_master["Hedged"] == hed)
        ].copy()

        # 2) Transferible = Yes
        if "Transferable" in base.columns:
            base = base[base["Transferable"] == "Yes"].copy()
        else:
            base = base.iloc[0:0].copy()

        # 3) MiFID FH Clean
        if "MiFID FH" in base.columns:
            mifid_norm = base["MiFID FH"].astype(str).str.strip().str.lower()
            base = base[mifid_norm.isin(clean_set)].copy()
        else:
            base = base.iloc[0:0].copy()

        # 4) Prioridad AI (texto o flag IA si existe)
        if "Prospectus AF" in base.columns:
            ai_mask_text = base["Prospectus AF"].astype(str).apply(lambda s: _has_code(s, "AI"))
        else:
            ai_mask_text = pd.Series(False, index=base.index)

        if has_ia_col:
            ai_mask_flag = base["Prospectus AF - Independent Advice (IA)*"] \
                               .astype(str).str.strip().str.lower().isin({"yes","y","si","sí","true","1"})
        else:
            ai_mask_flag = pd.Series(False, index=base.index)

        chosen = base[ai_mask_text | ai_mask_flag].copy()

        # 5) Fallback T
        if chosen.empty and "Prospectus AF" in base.columns:
            t_mask = base["Prospectus AF"].astype(str).apply(lambda s: _has_code(s, "T"))
            chosen = base[t_mask].copy()

        # 6) Si no hay ninguna, devolver placeholder manteniendo el orden y el peso
        if chosen.empty:
            incidencias.append((str(fam), "Sin clase AI ni T (Clean) transferible con misma (Type of Share/Currency/Hedged)"))
            results.append({
                "Family Name":   fam,
                "Name":          fam,         # ← mostrar Family Name en columna Name
                "Type of Share": "",
                "Currency":      "",
                "Hedged":        "",
                "MiFID FH":      "",
                "MiFID EMT":     "",
                "Min. Initial":  "",
                "ISIN":          "",
                "Prospectus AF": "",
                "Transferable":  "",
                "Ongoing Charge": np.nan,     # ← NaN para que no falle el formateo/TER
                "Weight %":      float(w) if pd.notnull(w) else 0.0
            })
            continue

        # 7) Elegir menor Ongoing Charge
        chosen = chosen.sort_values("Ongoing Charge", na_position="last")
        best = chosen.iloc[0]

        # 8) Completar campos (con pequeños fallbacks)
        name_val = best.get("Name") or best.get("Share Class Name") or best.get("Fund Name") or best.get("Family Name")
        emt_val  = best.get("MiFID EMT") or best.get("MIFID EMT")

        results.append({
            "Family Name":   best.get("Family Name"),
            "Name":          name_val,
            "Type of Share": best.get("Type of Share"),
            "Currency":      best.get("Currency"),
            "Hedged":        best.get("Hedged"),
            "MiFID FH":      best.get("MiFID FH"),
            "MiFID EMT":     emt_val,
            "Min. Initial":  best.get("Min. Initial"),
            "ISIN":          best.get("ISIN"),
            "Prospectus AF": best.get("Prospectus AF"),
            "Transferable":  "Yes",
            "Ongoing Charge": best.get("Ongoing Charge"),
            "Weight %":      float(w) if pd.notnull(w) else 0.0
        })

    # IMPORTANTe: NO reordenar results -> respeta el orden de Cartera I
    return pd.DataFrame(results), incidencias
    
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
    """Formatea un ratio (p.ej. 0.0123) como % europeo '1,23%'. """
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
    Busca la primera celda cuyo texto normalizado está en 'targets'.
    Devuelve (row_idx, col_idx) o (None, None) si no encuentra.
    """
    for r in range(df_any.shape[0]):
        for c in range(df_any.shape[1]):
            if _norm_txt(df_any.iat[r, c]) in targets:
                return r, c
    return None, None

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

# --- Cargar el Excel de CARTERA (auto-descubrimiento ISIN / TOTAL INVERSIÓN) ---
try:
    df_any = pd.read_excel(weights_file, header=None)
except Exception as e:
    st.error(f"No se pudo leer la cartera: {e}")
    st.stop()

targets_isin   = {"isin"}
targets_weight = {"total inversion", "total inversion."}  # admite variantes normalizadas

r_isin, c_isin = _find_header_cell(df_any, targets_isin)
r_wgt,  c_wgt  = _find_header_cell(df_any, targets_weight)

if r_isin is None or r_wgt is None:
    st.error("No se han encontrado las cabeceras 'ISIN' y/o 'TOTAL INVERSIÓN'.")
    st.stop()

# Leer hacia abajo desde cada cabecera
col_isin_vals = df_any.iloc[r_isin+1:, c_isin].reset_index(drop=True)
col_wgt_vals  = df_any.iloc[r_wgt+1:,  c_wgt].reset_index(drop=True)

# Emparejar por índice y construir df_weights_raw
max_len = max(len(col_isin_vals), len(col_wgt_vals))
col_isin_vals = col_isin_vals.reindex(range(max_len))
col_wgt_vals  = col_wgt_vals.reindex(range(max_len))
df_weights_raw = pd.DataFrame({"ISIN": col_isin_vals, "Peso %": col_wgt_vals})

# Quitar filas vacías y excluir el total (100) sin ISIN
df_weights_raw.dropna(how="all", inplace=True)
df_weights_raw = df_weights_raw[df_weights_raw["ISIN"].notna()].copy()

# Normalizar y agrupar duplicados
df_weights = _clean_weights(df_weights_raw)
df_weights["ISIN"] = df_weights["ISIN"].astype(str).str.strip().str.upper()
df_weights = df_weights.groupby("ISIN", as_index=False)["Peso %"].sum()


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

# TER y tabla
ter_I = calcular_ter(df_I_raw.rename(columns={"Peso %": "Weight %"}))
st.session_state.cartera_I = {"table": df_I_raw.rename(columns={"Peso %": "Weight %"}), "ter": ter_I}

mostrar_tabla_con_formato(st.session_state.cartera_I["table"], "Tabla Cartera I (original)")
if st.session_state.cartera_I["ter"] is not None:
    st.metric("📊 TER Cartera I", _fmt_ratio_eu_percent(st.session_state.cartera_I["ter"], 2))

# Incidencias de merge
incidencias = list(incidencias_merge)

# =========================
# 4) Convertir a Cartera II (AI)
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
# 5) Comparación I vs II
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
# 6) Incidencias
# =========================
if incidencias:
    st.subheader("⚠️ Incidencias detectadas")
    for fam, msg in incidencias:
        st.error(f"{fam}: {msg}")
