# FlorOrda · HT-01
Clasificación supervisada de **alcornoque, adelfa, arce japonés y olivo** a partir de cuatro descriptores de hojas. Fotos reales con créditos. El programa no analiza fotografías ni reconoce plantas fuera de estas cuatro clases.

## Entregables
- `dist/FlorOrda_Analisis_Reporte_Manual.ipynb`: cuaderno ejecutado con flujo, análisis, gráficos, entrenamiento, evaluación y manual.
- `dist/Reporte_FlorOrda.html`: versión legible del reporte, con resultados y figuras.
- `data/leaf_original.csv`: fuente numérica original preservada.
- `dist/data/flororda_4_plantas.csv`: 50 registros de las cuatro clases.
- `dist/model.js`: árbol entrenado, exportado para inferencia en el navegador.

## Abrir la web
Abre `dist/index.html` o publica `dist` como sitio estático. Selecciona un ejemplo, modifica los descriptores y pulsa Clasificar mi planta. Las fotos requieren Internet.

## Reproducir
Desde la raíz: `pip install -r requirements.txt` y `python train.py`. Para ejecutar el notebook, ábrelo dentro de `dist`; sus datos están en `dist/data`.

## Evaluación
37 muestras de entrenamiento y 13 de prueba, separación estratificada, semilla 42. Árbol: profundidad 4, mínimo 2 por hoja. Exactitud de prueba: **13/13**, validación cruzada de 3 particiones solo en entrenamiento: **97.22 %**. Son resultados de una muestra pequeña, sin validación de campo. Las probabilidades del árbol no representan certeza botánica.

## Publicación
Repositorio: https://github.com/Loren07-kie/SEMINARIO-TAREAS
Web: https://flororda.onrender.com
Reporte: https://flororda.onrender.com/Reporte_FlorOrda.html

## Fuentes
Leaf, UCI: https://doi.org/10.24432/C53C78 (CC BY 4.0). Consulta `dist/CREDITOS.md` para las fotografías y sus licencias.
