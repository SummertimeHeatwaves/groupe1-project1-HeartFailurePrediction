def optimize_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimise l'utilisation mémoire d'un DataFrame en convertissant
    les types de données vers des types plus légers :
      - float64  →  float32
      - int64    →  int8 / int16 / int32  (selon la plage réelle)
      - int32    →  int8 / int16          (selon la plage réelle)
      - object   →  category              (si faible cardinalité)
    """
    df_opt = df.copy()
    mem_before = df_opt.memory_usage(deep=True).sum()
    changes = []

    for col in df_opt.columns:
        dtype = df_opt[col].dtype
        original_type = str(dtype)

        # float64 → float32
        if dtype == np.float64:
            df_opt[col] = df_opt[col].astype(np.float32)
            changes.append((col, original_type, 'float32'))

        # int64 → int8 / int16 / int32
        elif dtype == np.int64:
            col_min = df_opt[col].min()
            col_max = df_opt[col].max()
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df_opt[col] = df_opt[col].astype(np.int8)
                changes.append((col, original_type, 'int8'))
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df_opt[col] = df_opt[col].astype(np.int16)
                changes.append((col, original_type, 'int16'))
            else:
                df_opt[col] = df_opt[col].astype(np.int32)
                changes.append((col, original_type, 'int32'))

        # int32 → int8 / int16
        elif dtype == np.int32:
            col_min = df_opt[col].min()
            col_max = df_opt[col].max()
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df_opt[col] = df_opt[col].astype(np.int8)
                changes.append((col, original_type, 'int8'))
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df_opt[col] = df_opt[col].astype(np.int16)
                changes.append((col, original_type, 'int16'))

        # object → category
        elif dtype == object:
            if df_opt[col].nunique() / len(df_opt) < 0.5:
                df_opt[col] = df_opt[col].astype('category')
                changes.append((col, original_type, 'category'))

    mem_after = df_opt.memory_usage(deep=True).sum()
    reduction = (1 - mem_after / mem_before) * 100

    if verbose:
        print('=' * 55)
        print('         RAPPORT OPTIMISATION MÉMOIRE')
        print('=' * 55)
        print(f'  Mémoire AVANT  :  {mem_before / 1024:.2f} KB  ({mem_before:,} bytes)')
        print(f'  Mémoire APRÈS  :  {mem_after  / 1024:.2f} KB  ({mem_after:,} bytes)')
        print(f'  Réduction      :  {reduction:.1f}%')
        print('─' * 55)
        print(f'  {"Colonne":<28}  {"Avant":<10}  →  Après')
        print('─' * 55)
        for col, before, after in changes:
            print(f'  {col:<28}  {before:<10}  →  {after}')
        print('=' * 55)

    return df_opt


# ── Utilisation ──────────────────────────────────────────────────────────────
df_optimized = optimize_memory(df_processed)