"""Analizar suelos con K-Means y DBSCAN; exportar resultados reproducibles."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

VARIABLES = ['pH', 'CE_dS_m', 'Organic_Matter_pct', 'Available_P_ppm',
             'Available_K_ppm', 'Sand_pct', 'Clay_pct']


def analizar(csv, salida, k=None, eps=1.4, minimo=3):
    datos = pd.read_csv(csv)
    faltan = set(['Sample_ID'] + VARIABLES) - set(datos.columns)
    if faltan:
        raise ValueError('Faltan columnas: ' + ', '.join(sorted(faltan)))
    if datos.Sample_ID.isna().any() or datos.Sample_ID.duplicated().any():
        raise ValueError('Los IDs deben ser completos y únicos.')
    X = datos[VARIABLES].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(X.to_numpy()).all() or len(X) < 8:
        raise ValueError('Se necesitan al menos 8 filas numéricas completas.')
    if len(X.drop_duplicates()) < 6:
        raise ValueError('Se necesitan al menos 6 vectores diferentes.')
    if (X < 0).any().any() or (X.pH > 14).any():
        raise ValueError('Hay valores fuera de rango físico.')
    if (X[['Sand_pct', 'Clay_pct', 'Organic_Matter_pct']] > 100).any().any():
        raise ValueError('Los porcentajes no pueden superar 100.')
    if (X.Sand_pct + X.Clay_pct > 100.1).any():
        raise ValueError('Arena y arcilla suman más de 100%.')
    escala = StandardScaler()
    Z = escala.fit_transform(X)
    modelos, scores = {}, []
    for n in range(2, 7):
        m = KMeans(n_clusters=n, n_init=30, random_state=42).fit(Z)
        modelos[n] = m
        scores.append({'K': n, 'silueta': float(silhouette_score(Z, m.labels_)),
                       'inercia': float(m.inertia_)})
    elegido = k if k is not None else max(scores, key=lambda s: s['silueta'])['K']
    modelo = modelos[elegido]
    db = DBSCAN(eps=eps, min_samples=minimo).fit_predict(Z)
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    resultado = datos[['Sample_ID'] + VARIABLES].copy()
    resultado['Grupo_KMeans'] = modelo.labels_ + 1
    resultado['Grupo_DBSCAN'] = np.where(db < 0, 'Ruido', (db + 1).astype(str))
    resultado.to_csv(salida / 'resultados.csv', index=False)
    pd.DataFrame(scores).to_csv(salida / 'comparacion_K.csv', index=False)
    export = {'variables': VARIABLES, 'media': escala.mean_.tolist(),
              'escala': escala.scale_.tolist(), 'K': elegido,
              'centroides': modelo.cluster_centers_.tolist()}
    (salida / 'modelo.json').write_text(json.dumps(export, indent=2), encoding='utf-8')
    print(pd.DataFrame(scores).round(4).to_string(index=False))
    print('K elegido:', elegido, '| resultados en:', salida.resolve())
    print('La silueta no es precisión; los grupos no indican dosis de fertilizante.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default=Path(__file__).parent / 'dist/data/terragem_original.csv')
    parser.add_argument('--salida', default='resultados')
    parser.add_argument('--k', type=int, choices=range(2, 7), default=None)
    parser.add_argument('--eps', type=float, default=1.4)
    parser.add_argument('--minimo', type=int, default=3)
    args = parser.parse_args()
    if args.eps <= 0 or args.minimo < 2:
        parser.error('eps debe ser positivo y minimo al menos 2.')
    analizar(args.csv, args.salida, args.k, args.eps, args.minimo)
