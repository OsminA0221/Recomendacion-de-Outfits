"""PASO 1: limpia prendas.csv y crea data/prendas_limpio.csv

Uso (desde la raiz del proyecto):
      python -m Recomendaciones.preparar_dataset
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .colores import COLOR_ES, COLOR_HEX

DATA = Path(__file__).resolve().parents[1] / "data"   # <raiz>/data
CSV_ENTRADA = DATA / "prendas.csv"
CSV_SALIDA = DATA / "prendas_limpio.csv"
IMAGENES = DATA / "Imagenes"


def main() -> int:
    entrada = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_ENTRADA
    imagenes = Path(sys.argv[2]) if len(sys.argv) > 2 else IMAGENES

    df = pd.read_csv(entrada)
    n0 = len(df)

    # 1) texto limpio
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip()

    # 2) sin duplicados
    df = df.drop_duplicates(subset="item_id").reset_index(drop=True)

    # 3) genero al mismo vocabulario que el detector ("Hombre"/"Mujer")
    df["genero"] = df["genero"].str.lower().map({"masculino": "Hombre", "femenino": "Mujer"})
    df = df.dropna(subset=["genero"])

    # 4) corte vacio = sin corte definido
    df["corte"] = df["corte"].fillna("sin_corte").str.lower()
    df["ajuste"] = df["ajuste"].str.lower()
    df["estilo"] = df["estilo"].str.lower()

    # 5) COLOR: nombre -> HEX (esto es lo que faltaba para comparar con la piel)
    df["color_nombre"] = df["colour_group_name"]
    df["color_hex"] = df["colour_group_name"].map(COLOR_HEX)
    df["color_es"] = df["perceived_colour_master_name"].map(COLOR_ES).fillna("desconocido")
    df["patron"] = df["graphical_appearance_name"]

    # 6) ruta de imagen valida en cualquier sistema operativo
    df["imagen_path"] = df["imagen_path"].str.replace("\\", "/", regex=False)

    # 7) columnas finales (las derivadas formas/somatotipos ya no se usan)
    cols = ["item_id", "nombre", "categoria", "subcategoria", "genero", "marca",
            "estilo", "corte", "ajuste", "color_nombre", "color_hex", "color_es",
            "patron", "imagen_path"]
    df[cols].to_csv(CSV_SALIDA, index=False, encoding="utf-8")

    # ---- reporte ----
    print(f"Filas: {n0} -> {len(df)} | guardado en {CSV_SALIDA}")
    print(f"Sin HEX (Other/Unknown): {int(df['color_hex'].isna().sum())}")
    faltan = sorted(set(df['colour_group_name']) - set(COLOR_HEX))
    if faltan:
        print("AVISO: colores sin HEX en COLOR_HEX:", faltan)
    print("\nPrendas por categoria y genero:")
    print(pd.crosstab(df["categoria"], df["genero"]))
    if imagenes.exists():
        ausentes = sum(not (imagenes / Path(p).name).exists() for p in df["imagen_path"])
        print(f"\nImagenes faltantes en {imagenes}: {ausentes}")
    else:
        print(f"\n(No existe la carpeta {imagenes}; copia ahi tus imagenes.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())