import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from typing import Optional
from sam_clean_map import SAM_CLEAN_MAP

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
        "VALOR ACTUAL (EUR)","Weight %"
    ]
    for c in wanted:
        if c not in tbl.columns:
            tbl[c] = np.nan
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

def abrir_outlook_con_comparativa(destinatarios: Optional[str],
                                  asunto: str,
                                  dfI_sub: pd.DataFrame,
                                  dfII_sel: pd.DataFrame,
                                  ter_I_sub: float | None,
                                  ter_II_sel: float | None,
                                  adjuntar_excel: bool = True):
    """
    Abre Outlook con un nuevo correo (no lo envía).
    - Cuerpo: solo resumen de TER I, TER II y Diferencia.
    - Adjunto: Excel con HOJA 1 = Cartera I (subset) y HOJA 2 = Cartera II (subset),
               mismas columnas y orden que en la app (pretty_table),
               con % formateados (si hay xlsxwriter).
    """
    import os, tempfile
    try:
        import win32com.client as win32
    except Exception:
        st.error("No se pudo cargar la librería de Outlook (pywin32). Instala `pip install pywin32`.")
        return

    def _p(x):
        return _fmt_ratio_eu_percent(x, 2) if x is not None else "-"

    cuerpo = f"""Hola,

Adjunto la comparativa de la cartera actual.

- TER Cartera I (actual): {_p(ter_I_sub)}
- TER Cartera II (transformada): {_p(ter_II_sel)}
- Diferencia (II - I): {_p((ter_II_sel - ter_I_sub) if (ter_I_sub is not None and ter_II_sel is not None) else None)}

Saludos,
"""

    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    # ↓↓↓ deja el campo "Para" vacío si no se pasa nada
    if destinatarios and destinatarios.strip():
        mail.To = destinatarios.strip()
    mail.Subject = asunto
    mail.Body = cuerpo

    if adjuntar_excel:
        from io import BytesIO
        # 1) Tablas tal como se muestran en la app (mismo orden/columnas)
        dfI_x = pretty_table(dfI_sub).copy()
        dfII_x = pretty_table(dfII_sel).copy()

        # 2) Normalizar tipos numéricos para Excel
        def _num(s): return pd.to_numeric(s, errors="coerce")
        for dfx in (dfI_x, dfII_x):
            if "Ongoing Charge" in dfx.columns:
                dfx["Ongoing Charge"] = _num(dfx["Ongoing Charge"])
            if "Weight %" in dfx.columns:
                dfx["Weight %"] = _num(dfx["Weight %"]) / 100.0  # Excel espera ratio
            if "VALOR ACTUAL (EUR)" in dfx.columns:
                dfx["VALOR ACTUAL (EUR)"] = _num(dfx["VALOR ACTUAL (EUR)"])

        # 3) Motor Excel (xlsxwriter si hay, si no openpyxl)
        try:
            import xlsxwriter  # noqa
            _engine = "xlsxwriter"
        except Exception:
            _engine = "openpyxl"

        buf = BytesIO()
        with pd.ExcelWriter(buf, engine=_engine) as writer:
            dfI_x.to_excel(writer, index=False, sheet_name="Cartera I")
            dfII_x.to_excel(writer, index=False, sheet_name="Cartera II")

            if _engine == "xlsxwriter":
                book = writer.book
                fmt_pct2 = book.add_format({"num_format": "0.00%"})
                fmt_pct4 = book.add_format({"num_format": "0.0000%"})
                fmt_num  = book.add_format({"num_format": "#,##0.00"})

                def _apply(ws, df):
                    cols = {c:i for i,c in enumerate(df.columns)}
                    for c, idx in cols.items():
                        width = max(len(c), int(df[c].astype(str).map(len).fillna(0).max() if c in df else 10))
                        ws.set_column(idx, idx, min(width + 2, 60))
                    ws.freeze_panes(1, 0)
                    if "Ongoing Charge" in cols:     ws.set_column(cols["Ongoing Charge"], cols["Ongoing Charge"], None, fmt_pct4)
                    if "Weight %" in cols:           ws.set_column(cols["Weight %"],      cols["Weight %"],      None, fmt_pct2)
                    if "VALOR ACTUAL (EUR)" in cols: ws.set_column(cols["VALOR ACTUAL (EUR)"], cols["VALOR ACTUAL (EUR)"], None, fmt_num)

                _apply(writer.sheets["Cartera I"], dfI_x)
                _apply(writer.sheets["Cartera II"], dfII_x)

        tmp_path = os.path.join(tempfile.gettempdir(), "Comparativa_Carteras.xlsx")
        with open(tmp_path, "wb") as f:
            f.write(buf.getvalue())
        mail.Attachments.Add(tmp_path)

    mail.Display()
def _sam_lookup(fam: str, currency: str | None = None, hedged: str | None = None):
    """Busca entrada SAM por Family Name (y opcionalmente Currency/Hedged)."""
    if not fam:
        return None
    fam_norm = str(fam).strip()
    # 1) Coincidencia específica por Family+Currency+Hedged (si tu catálogo la trae)
    if currency is not None and hedged is not None:
        for rec in SAM_CLEAN_MAP:
            if (rec.get("Family Name", "").strip() == fam_norm and
                str(rec.get("Currency", "")).strip() == str(currency).strip() and
                str(rec.get("Hedged", "")).strip() == str(hedged).strip()):
                return rec
    # 2) Fallback por Family Name
    for rec in SAM_CLEAN_MAP:
        if rec.get("Family Name", "").strip() == fam_norm:
            return rec
    return None

def _build_row_from_sam(df_master: pd.DataFrame, sam_rec: dict, fallback_row: pd.Series | dict):
    """
    Crea la fila ‘clean SAM’ y rellena con maestro si está; si no, usa datos del fondo original (I).
    """
    isin_clean = sam_rec.get("ISIN", "")
    oc_clean   = sam_rec.get("Ongoing Charge", None)
    name_clean = sam_rec.get("Name", "")

    row_master = None

    if isin_clean:
        m = df_master[df_master["ISIN"].astype(str).str.upper() == str(isin_clean).upper()]
    
        # <- inicializa SIEMPRE un contenedor válido
        row_master = {}                     # ⬅️ queda dict vacío si no hay match
        if not m.empty:
            row_master = m.iloc[0]          # ⬅️ y si hay match, fila del maestro

    def _gv(src, key, default=""):
        # Si no hay fuente -> devuelve el valor por defecto
        if src is None:
            return default
        # Series de pandas o dict: usa .get de forma segura
        if isinstance(src, (pd.Series, dict)):
            return src.get(key, default)
        # Último recurso: intenta indexar; si falla, devuelve default
        try:
            return src[key]
        except Exception:
            return default

    Type_of_Share = _gv(row_master, "Type of Share", _gv(fallback_row, "Type of Share", ""))
    Currency      = _gv(row_master, "Currency",      _gv(fallback_row, "Currency", ""))
    Hedged        = _gv(row_master, "Hedged",        _gv(fallback_row, "Hedged", ""))

    return {
        "ISIN": isin_clean,
        "Name": name_clean,
        "Type of Share": Type_of_Share,
        "Currency": Currency,
        "Hedged": Hedged,
        "Ongoing Charge": oc_clean,
        "Min. Initial": _gv(row_master, "Min. Initial", _gv(fallback_row, "Min. Initial", "")),
        "MiFID FH": _gv(row_master, "MiFID FH", ""),
        "MiFID EMT": _gv(row_master, "MiFID EMT", _gv(row_master, "MIFID EMT", "")),
        "Prospectus AF": _gv(row_master, "Prospectus AF", ""),
        "Soft Close": _gv(row_master, "Soft Close", ""),
    }

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

    # Ongoing Charge -> mostrar como % (viene como ratio: 0.0123 -> 1,23%)
    if "Ongoing Charge" in df_show.columns:
        def _fmt_oc(v):
            if pd.isna(v): 
                return ""
            try:
                x = float(str(v).replace(",", "."))  # por si llega como texto
            except Exception:
                return ""
            return _fmt_ratio_eu_percent(x, 2)
        df_show["Ongoing Charge"] = df_in.get("Ongoing Charge").apply(_fmt_oc).astype(str)

    # Weight % -> mostrar como % con símbolo (origen ya es %: 12.34 -> 12,34%)
    if "Weight %" in df_show.columns:
        def _fmt_w(v):
            if pd.isna(v):
                return ""
            try:
                x = float(str(v).replace(",", ".")) / 100.0  # pasar a ratio
            except Exception:
                return ""
            return _fmt_ratio_eu_percent(x, 2)
        df_show["Weight %"] = df_in.get("Weight %").apply(_fmt_w).astype(str)

    # Valor actual en formato europeo con coma (sin %)
    if "VALOR ACTUAL (EUR)" in df_show.columns:
        def _fmt_val(v):
            if pd.isna(v): 
                return ""
            try:
                return _format_eu_number(float(v), 2)
            except Exception:
                return str(v)
        df_show["VALOR ACTUAL (EUR)"] = df_in.get("VALOR ACTUAL (EUR)").apply(_fmt_val).astype(str)

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
    reported_no_ai_names = set()  # para no repetir la misma incidencia por Name
    
    def _name_has_cartera(row):
        for col in ("Name", "Share Class Name", "Fund Name"):
            v = row.get(col, None)
            if pd.notna(v) and "cartera" in str(v).lower():
                return True
        return False
        # --- Helpers EMT ---
    def _emt_str(x):
        s = "" if pd.isna(x) else str(x)
        s = s.replace(" ", "").strip().upper()
        return s

    def _emt_is_blocked(x):
        # Bloquear 'NN' en las dos primeras posiciones
        s = _emt_str(x)
        return len(s) >= 2 and s[0] == "N" and s[1] == "N"

    def _filter_not_blocked(df):
        # Aplica el bloqueo 'NN' si la columna existe (MiFID EMT o MIFID EMT)
        cols = [c for c in ("MiFID EMT", "MIFID EMT") if c in df.columns]
        if not cols:
            return df
        return df[~df[cols[0]].apply(_emt_is_blocked)]

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
        # --- PASO 0: si el Family Name es SAM, forzar la clase clean oficial del catálogo ---
        sam_rec = _sam_lookup(fam, currency=cur, hedged=hed)
        if sam_rec is not None:
            best_dict = _build_row_from_sam(df_master, sam_rec, row)
    
            name_val = best_dict.get("Name") or best_dict.get("Family Name") or "(sin nombre)"
            emt_val  = best_dict.get("MiFID EMT") or best_dict.get("MIFID EMT")
    
            # Aviso EMT: si empieza por 'N' -> profesional
            try:
                s_emt = ("" if pd.isna(emt_val) else str(emt_val)).replace(" ", "").upper()
                if len(s_emt) >= 1 and s_emt[0] == "N":
                    incidencias.append((name_val, "Clase solo disponible para cliente profesional"))
            except Exception:
                pass
    
            # Min. Initial > 10M (heurística por longitud > 11)
            min_initial = str(best_dict.get("Min. Initial", "")).strip()
            if len(min_initial) > 11:
                incidencias.append((name_val, f"El mínimo inicial de contratación del fondo '{name_val}' es de '{min_initial}', consultar."))
    
            out_rows.append({
                "ISIN": best_dict.get("ISIN",""),
                "Name": name_val,
                "Type of Share": best_dict.get("Type of Share",""),
                "Currency": best_dict.get("Currency",""),
                "Hedged": best_dict.get("Hedged",""),
                "Ongoing Charge": best_dict.get("Ongoing Charge", np.nan),
                "Min. Initial": best_dict.get("Min. Initial",""),
                "MiFID FH": best_dict.get("MiFID FH",""),
                "MiFID EMT": emt_val,
                "Prospectus AF": best_dict.get("Prospectus AF",""),
                "Soft Close": best_dict.get("Soft Close",""),
                "VALOR ACTUAL (EUR)": valor
            })
            continue  # ¡Muy importante! ya no se ejecuta la búsqueda AI/T/Cartera para SAM
    

        # 1) AI + Transferable Yes
        ai = subset[subset["Prospectus AF"].apply(lambda x: _has_code(x, "AI"))]
        if "Transferable" in ai.columns:
            ai = ai[ai["Transferable"].fillna("").astype(str).str.strip().str.lower() == "yes"]
        ai = _filter_not_blocked(ai)  # <-- bloquea NN
        if not ai.empty:
            chosen = ai

        # 2) T + Transferable Yes + MiFID FH clean
        if chosen is None or chosen.empty:
            t = subset[subset["Prospectus AF"].apply(lambda x: _has_code(x, "T"))]
            if "Transferable" in t.columns:
                t = t[t["Transferable"].fillna("").astype(str).str.strip().str.lower() == "yes"]
            if "MiFID FH" in t.columns:
                t = t[t["MiFID FH"].fillna("").astype(str).str.lower().isin(clean_set)]
            t = _filter_not_blocked(t)  # <-- bloquea NN
            if not t.empty:
                chosen = t

        # 3) Name que contenga 'Cartera'
        if chosen is None or chosen.empty:
            fam_pool = df_master[df_master["Family Name"] == fam].copy()
            cartera_pool = fam_pool[fam_pool.apply(_name_has_cartera, axis=1)]
            cartera_pool = _filter_not_blocked(cartera_pool)  # <-- bloquea NN
            if not cartera_pool.empty:
                chosen = cartera_pool

        # Si no hay clase apta
        # Si no hay clase apta
        if chosen is None or chosen.empty:
            name_row = row.get("Name", "(sin nombre)")
            # solo reportar una vez por Name (p.ej. ETFs con múltiples líneas)
            key = str(name_row).strip().lower()
            if key not in reported_no_ai_names:
                incidencias.append(
                    (name_row, "Para este fondo no se ha encontrado una clase que cumpla los criterios de Asesoramiento Independiente.")
                )
                reported_no_ai_names.add(key)
            continue

        # Elegir la de menor Ongoing Charge
        best = chosen.sort_values("Ongoing Charge", na_position="last").iloc[0]
        
        # --- AVISOS Y VALIDACIONES ANTES DE AÑADIR LA FILA ---
        
        # Nombre y EMT de la clase elegida (para reutilizar)
        name_val = (
            best.get("Name")
            or best.get("Share Class Name")
            or best.get("Fund Name")
            or best.get("Family Name")
            or "(sin nombre)"
        )
        emt_val = best.get("MiFID EMT") or best.get("MIFID EMT")
        
        # 1) Aviso EMT: si la 1ª letra es 'N' -> solo profesional
        s_emt = _emt_str(emt_val)  # ← recuerda tener el helper añadido (paso A)
        if len(s_emt) >= 1 and s_emt[0] == "N":
            incidencias.append((name_val, "Clase solo disponible para cliente profesional"))
        
        # 2) Min. Initial > 10 millones (heurística por longitud)
        min_initial = str(best.get("Min. Initial", "")).strip()
        if len(min_initial) > 11:
            incidencias.append(
                (name_val, f"El mínimo inicial de contratación del fondo '{name_val}' es de '{min_initial}', consultar.")
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
# 5) Comparación: lista editable basada en TODA Cartera I
# =========================
st.subheader("Paso 4: Comparar Cartera I vs Cartera II (edita qué entra y si usar original)")

if (
    st.session_state.cartera_I
    and st.session_state.cartera_I["table"] is not None
    and not st.session_state.cartera_I["table"].empty
    and st.session_state.cartera_II
    and st.session_state.cartera_II["table"] is not None
):
    # --- Tablas base ---
    dfI_all = st.session_state.cartera_I["table"].copy()          # todos los válidos en AllFunds (por ISIN) con valor
    dfII_all = st.session_state.cartera_II["table"].copy()         # solo transformados AI (puede estar vacía)

    # Mapear ISIN -> Family Name (clave estable para emparejar)
    fam_map = df_master[["ISIN", "Family Name"]].copy()
    fam_map["ISIN"] = fam_map["ISIN"].astype(str).str.upper()

    # Normaliza I con Family Name
    dfI_all["ISIN"] = dfI_all["ISIN"].astype(str).str.upper()
    dfI_all = pd.merge(dfI_all, fam_map, on="ISIN", how="left", suffixes=("", "_fam"))
    if "Family Name" not in dfI_all.columns:
        for c in ("Family Name_fam", "Family Name_x", "Family Name_y"):
            if c in dfI_all.columns:
                dfI_all.rename(columns={c: "Family Name"}, inplace=True)
                break

    # Normaliza II con Family Name
    if not dfII_all.empty:
        dfII_all["ISIN"] = dfII_all["ISIN"].astype(str).str.upper()
        dfII_all = pd.merge(dfII_all, fam_map, on="ISIN", how="left", suffixes=("", "_fam"))
        if "Family Name" not in dfII_all.columns:
            for c in ("Family Name_fam", "Family Name_x", "Family Name_y"):
                if c in dfII_all.columns:
                    dfII_all.rename(columns={c: "Family Name"}, inplace=True)
                    break

    # --- Candidatos: UNO por Family Name (origen = I; representativo por Name de I) ---
    rep_I = (
        dfI_all.sort_values(["Family Name","Name"])
               .drop_duplicates(subset=["Family Name"], keep="first")
               [["Family Name","Name","VALOR ACTUAL (EUR)","Ongoing Charge"]]
               .reset_index(drop=True)
    )

    families_in_II = set(dfII_all["Family Name"]) if not dfII_all.empty else set()

    # DataFrame del selector (persistente)
    key_df = "selector_fondos_allI_df"
    if key_df not in st.session_state:
        sel = rep_I.copy()
        sel.insert(0, "Incluir", True)  # entra por defecto
        # si no existe AI, por defecto usar original; si existe AI, por defecto usar transformada
        sel.insert(1, "Usar versión original (I)", sel["Family Name"].apply(lambda f: f not in families_in_II))
        # informativo: ¿está en II?
        sel.insert(2, "Disponible en Cartera II (AI)", sel["Family Name"].apply(lambda f: f in families_in_II))
        st.session_state[key_df] = sel
    else:
        # Resincro por Family Name manteniendo elecciones previas
        prev = st.session_state[key_df][["Family Name","Incluir","Usar versión original (I)"]]
        sel = rep_I.merge(prev, on="Family Name", how="left")
        sel["Incluir"] = sel["Incluir"].fillna(True)
        sel["Usar versión original (I)"] = sel["Usar versión original (I)"].fillna(
            sel["Family Name"].apply(lambda f: f not in families_in_II)
        )
        sel.insert(2, "Disponible en Cartera II (AI)", sel["Family Name"].apply(lambda f: f in families_in_II))
        st.session_state[key_df] = sel

    # --- Botón / modo edición ---
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = False
    if st.button("🛠️ Editar cartera (incluir/excluir y elegir I/II)"):
        st.session_state.edit_mode = not st.session_state.edit_mode

    # Subconjunto de columnas para el editor (sin Valor / OC)
    editor_cols = ["Family Name", "Name", "Disponible en Cartera II (AI)", "Incluir", "Usar versión original (I)"]

    # Inicialización (primer render): Incluir = Disponible / Usar original = False
    if key_df not in st.session_state:
        base = rep_I.copy()
        base["Disponible en Cartera II (AI)"] = base["Family Name"].isin(families_in_II)
        st.session_state[key_df] = pd.DataFrame({
            "Family Name": base["Family Name"],
            "Name": base["Name"],
            "Disponible en Cartera II (AI)": base["Disponible en Cartera II (AI)"],
            "Incluir": base["Disponible en Cartera II (AI)"],  # ← igual que Disponible
            "Usar versión original (I)": False                 # ← vacío por defecto
        })
    else:
        # Re-sincronizar por Family Name manteniendo las elecciones previas
        prev = st.session_state[key_df][["Family Name","Incluir","Usar versión original (I)"]]
        base = rep_I.merge(prev, on="Family Name", how="left")
        base["Disponible en Cartera II (AI)"] = base["Family Name"].isin(families_in_II)
        # faltantes → defaults
        base["Incluir"] = base["Incluir"].fillna(base["Disponible en Cartera II (AI)"])
        base["Usar versión original (I)"] = base["Usar versión original (I)"].fillna(False)
        st.session_state[key_df] = base[["Family Name","Name","Disponible en Cartera II (AI)","Incluir","Usar versión original (I)"]]

    # Editor
    if st.session_state.edit_mode:
        st.caption("Marca **Incluir** para que el fondo entre en la comparativa. "
                   "Si incluyes uno **no disponible** en Cartera II (AI), se marcará automáticamente **Usar versión original (I)**.")
        edited = st.data_editor(
            st.session_state[key_df][editor_cols],
            use_container_width=True,
            hide_index=True,
            key="selector_editor_allI",
            disabled=["Family Name","Name","Disponible en Cartera II (AI)"],
            column_config={
                "Disponible en Cartera II (AI)": st.column_config.CheckboxColumn(disabled=True),
                "Incluir": st.column_config.CheckboxColumn(),
                "Usar versión original (I)": st.column_config.CheckboxColumn(),
            },
        )
        # Regla automática: Incluir==True y Disponible==False -> Usar versión original = True
        m = edited["Incluir"] & ~edited["Disponible en Cartera II (AI)"]
        edited.loc[m, "Usar versión original (I)"] = True

        if st.button("✅ Aplicar selección"):
            st.session_state[key_df] = edited[editor_cols].copy()

    # --- Selección final (familias incluidas y forzadas a original) ---
    sel_tbl = st.session_state[key_df]
    selected = sel_tbl[sel_tbl["Incluir"]].copy()
    if selected.empty:
        st.info("No hay fondos seleccionados para comparar.")
        st.stop()

    families_selected = set(selected["Family Name"])
    families_use_orig = set(
        selected.loc[
            selected["Usar versión original (I)"] | ~selected["Disponible en Cartera II (AI)"],
            "Family Name"
        ].tolist()
    )

    # --- Cartera II (derecha) según selección ---
    dfII_ai   = dfII_all[dfII_all["Family Name"].isin(families_selected - families_use_orig)].copy()
    dfI_orig  = dfI_all[dfI_all["Family Name"].isin(families_use_orig)].copy()
    if not dfI_orig.empty and "VALOR ACTUAL (EUR)" in dfI_orig.columns:
        dfI_orig = dfI_orig.sort_values("VALOR ACTUAL (EUR)", ascending=False)
    dfI_orig = dfI_orig.drop_duplicates(subset=["Family Name"], keep="first")

    common_cols = list(dfII_all.columns)
    dfI_as_II   = dfI_orig.reindex(columns=common_cols, fill_value=np.nan)

    dfII_sel = pd.concat([dfII_ai, dfI_as_II], ignore_index=True)   # ← II FINAL

    # --- Cartera I (izquierda) alineada 1–a–1 con dfII_sel FINAL ---
    pattern_corto = r"(SPB\s*RF\s*CORTO\s*PLAZO|Santander\s+Corto\s+Plazo)"
    rows_I = []
    cols_I = dfI_all.columns

    for _, rowII in dfII_sel.reset_index(drop=True).iterrows():
        picked = None

        # Regla especial: ES0174735037 -> buscar por Name en I
        if str(rowII.get("ISIN", "")).strip().upper() == "ES0174735037":
            if "Name" in dfI_all.columns:
                cand = dfI_all["Name"].astype(str).str.contains(pattern_corto, case=False, regex=True, na=False)
                cand = dfI_all[cand]
                if not cand.empty:
                    if "VALOR ACTUAL (EUR)" in cand.columns:
                        cand = cand.sort_values("VALOR ACTUAL (EUR)", ascending=False)
                    picked = cand.iloc[0]

        # Regla general: por Family Name, preferir misma Currency/Hedged que II
        if picked is None:
            fam = rowII.get("Family Name")
            if pd.notna(fam) and "Family Name" in dfI_all.columns:
                cand = dfI_all[dfI_all["Family Name"] == fam].copy()
                if not cand.empty:
                    if "Currency" in cand.columns and "Hedged" in cand.columns:
                        cand["_pref_cur"] = cand["Currency"].astype(str).eq(str(rowII.get("Currency",""))).astype(int)
                        cand["_pref_hed"] = cand["Hedged"].astype(str).eq(str(rowII.get("Hedged",""))).astype(int)
                        if "VALOR ACTUAL (EUR)" in cand.columns:
                            cand["_valor"] = pd.to_numeric(cand["VALOR ACTUAL (EUR)"], errors="coerce").fillna(0)
                            cand = cand.sort_values(["_pref_cur","_pref_hed","_valor"], ascending=[False, False, False])
                            cand = cand.drop(columns=["_valor"])
                        else:
                            cand = cand.sort_values(["_pref_cur","_pref_hed"], ascending=[False, False])
                        cand = cand.drop(columns=["_pref_cur","_pref_hed"])
                    picked = cand.iloc[0]

        rows_I.append(picked.reindex(cols_I) if picked is not None else pd.Series(index=cols_I, dtype="object"))

    dfI_sub = pd.DataFrame(rows_I, columns=cols_I)
    if "ISIN" in dfI_sub.columns:
        dfI_sub = dfI_sub[~(dfI_sub["ISIN"].isna() & dfI_sub.isna().all(axis=1))].copy()

    # --- Recalcular pesos por valor (respeta OC: los sin OC quedan con Weight % = 0) ---
    dfI_sub  = recalcular_pesos_por_valor_respetando_oc(dfI_sub,  valor_col="VALOR ACTUAL (EUR)")
    ter_I_sub = calcular_ter_por_valor(dfI_sub)

    dfII_sel  = recalcular_pesos_por_valor_respetando_oc(dfII_sel, valor_col="VALOR ACTUAL (EUR)")
    ter_II_sel = calcular_ter_por_valor(dfII_sel)

    # ---------- Presentación ----------
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Cartera I (fondos incluidos)")
        mostrar_tabla_con_formato(dfI_sub, "Tabla Cartera I (subset comparable)")
        st.metric("TER medio ponderado", _fmt_ratio_eu_percent(ter_I_sub, 2) if ter_I_sub is not None else "-")

    with c2:
        st.markdown("#### Cartera II (final)")
        mostrar_tabla_con_formato(dfII_sel, "Tabla Cartera II (final con I/II según selección)")
        st.metric("TER medio ponderado", _fmt_ratio_eu_percent(ter_II_sel, 2) if ter_II_sel is not None else "-")

    if ter_I_sub is not None and ter_II_sel is not None:
        st.markdown("---")
        st.subheader("Diferencia de TER (II − I) en fondos incluidos")
        st.metric("Diferencia", _fmt_ratio_eu_percent(ter_II_sel - ter_I_sub, 2))

    # Botón Outlook (dentro del IF: ya existen dfI_sub/dfII_sel)
    if st.button("📧 Abrir Outlook con comparativa"):
        abrir_outlook_con_comparativa(
            destinatarios="",  # o None
            asunto="Comparativa TER – Cartera I vs Cartera II (definitiva)",
            dfI_sub=dfI_sub,
            dfII_sel=dfII_sel,
            ter_I_sub=ter_I_sub,
            ter_II_sel=ter_II_sel,
            adjuntar_excel=True
        )

else:
    st.info("Primero calcula Cartera I y convierte a Cartera II para ver la comparativa.")

# =========================
# 6) Incidencias (deduplicadas por Name+mensaje)
# =========================
# Recoge incidencias que tengas en variables locales y en session_state
incidencias_total = []
if "incidencias" in st.session_state and st.session_state.incidencias:
    incidencias_total.extend(st.session_state.incidencias)
# Si en tu flujo mantienes una lista local llamada `incidencias`, añádela:
try:
    if incidencias:
        incidencias_total.extend(incidencias)
except NameError:
    pass

if incidencias_total:
    st.subheader("⚠️ Incidencias detectadas")
    vistos = set()  # (name_normalizado, mensaje)
    for name, msg in incidencias_total:
        key = (str(name).strip().lower(), str(msg).strip())
        if key in vistos:
            continue
            
        vistos.add(key)
        st.error(f"{name}: {msg}")
