def convertir_a_AI(df_master: pd.DataFrame, df_cartera_I: pd.DataFrame):
    """
    Mantiene Type of Share, Currency, Hedged.
    Requiere Transferable = 'Yes'.
    Prioridad:
      1) Clase cuyo Prospectus AF contenga 'AI'
      2) Si no hay AI, clase cuyo Prospectus AF contenga 'T'
    Siempre se elige la de menor Ongoing Charge.
    Devuelve (df_cartera_II, incidencias)
    """
    results = []
    incidencias = []

    # Columna opcional de Allfunds que a veces viene para IA
    has_ia_col = "Prospectus AF - Independent Advice (IA)*" in df_master.columns

    for _, row in df_cartera_I.iterrows():
        fam = row.get("Family Name")
        tos = row.get("Type of Share")
        cur = row.get("Currency")
        hed = row.get("Hedged")
        w   = row.get("Weight %", 0.0)

        # 1) Candidatas con misma combinación (familia, ToS, Currency, Hedged)
        candidates = df_master[
            (df_master["Family Name"] == fam) &
            (df_master["Type of Share"] == tos) &
            (df_master["Currency"] == cur) &
            (df_master["Hedged"] == hed)
        ].copy()

        if candidates.empty:
            incidencias.append((str(fam), "No hay filas en maestro con misma combinación (Type of Share/Currency/Hedged)"))
            continue

        # 2) Deben ser transferibles
        if "Transferable" in candidates.columns:
            candidates = candidates[candidates["Transferable"] == "Yes"].copy()
        else:
            candidates = candidates.iloc[0:0].copy()

        if candidates.empty:
            incidencias.append((str(fam), "Sin candidatas transferibles con misma (Type of Share/Currency/Hedged)"))
            continue

        # 3) Prioridad AI
        ai_mask_text = candidates.get("Prospectus AF", pd.Series(index=candidates.index, dtype=object)) \
                                  .astype(str).apply(lambda s: _has_code(s, "AI"))
        if has_ia_col:
            # si existe la columna específica de IA, la usamos como OR
            ai_mask_flag = candidates["Prospectus AF - Independent Advice (IA)*"].astype(str).str.strip().str.lower().isin(
                {"yes","y","si","sí","true","1"}
            )
        else:
            ai_mask_flag = pd.Series(False, index=candidates.index)

        ai_candidates = candidates[ai_mask_text | ai_mask_flag].copy()
        chosen = ai_candidates

        # 4) Fallback T (si no hay AI)
        if chosen.empty:
            t_mask = candidates.get("Prospectus AF", pd.Series(index=candidates.index, dtype=object)) \
                               .astype(str).apply(lambda s: _has_code(s, "T"))
            t_candidates = candidates[t_mask].copy()
            chosen = t_candidates

        if chosen.empty:
            incidencias.append((str(fam), "Sin clase AI ni T transferible con misma (Type of Share/Currency/Hedged)"))
            continue

        # 5) Elegir menor Ongoing Charge
        chosen = chosen.sort_values("Ongoing Charge", na_position="last")
        best = chosen.iloc[0]

        results.append({
            "Family Name":   best.get("Family Name"),
            "Type of Share": best.get("Type of Share"),
            "Currency":      best.get("Currency"),
            "Hedged":        best.get("Hedged"),
            "MiFID FH":      best.get("MiFID FH"),
            "Min. Initial":  best.get("Min. Initial"),
            "ISIN":          best.get("ISIN"),
            "Prospectus AF": best.get("Prospectus AF"),
            "Transferable":  "Yes",  # confirmado por filtro
            "Ongoing Charge": best.get("Ongoing Charge"),
            "Weight %":      float(w) if pd.notnull(w) else 0.0
        })

    df_ii = pd.DataFrame(results)
    return df_ii, incidencias
