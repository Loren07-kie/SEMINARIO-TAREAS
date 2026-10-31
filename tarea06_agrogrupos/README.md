# AgroGrupos · Tarea 6
**PIAD-427 / HT-02: Crea programas con algoritmos de aprendizaje no supervisado.**

Estudiante: Lorena Abigail Coca Ccorpa.
Web: https://agrogrupos.onrender.com/

## Entrega
- `dist/AgroGrupos_Tarea_6.ipynb`: cuaderno autónomo ejecutado, datos incluidos, auditoría, flujo, K-Means, DBSCAN, PCA, estabilidad, sensibilidad, manual y bibliografía.
- `dist/Reporte_AgroGrupos.html`: reporte legible sin Jupyter, con seis gráficos y código desplegable.
- `dist/index.html`, `style.css`, `app.js`, `engine.js`, `data.js`: web completa.
- `dist/data/terragem_original.csv`: fuente original preservada sin modificaciones.
- `dist/data/plantilla_suelos.csv`: formato de carga con los 29 ejemplos y las siete variables elegidas.

## Abrir localmente
Desde esta carpeta ejecutar `python -m http.server 8000 --directory dist` y abrir http://localhost:8000/.
No necesita Node, claves API ni servidor de inferencia. Funciona localmente; las fuentes tipográficas opcionales requieren Internet y tienen alternativas del sistema.

## Ejecutar el cuaderno
Python 3.11–3.13 recomendado. Instalar con `pip install -r requirements.txt`, abrir el `.ipynb` en Jupyter o VS Code y seleccionar Restart & Run All. Los datos están integrados; no requiere rutas CSV externas. Las exportaciones se habilitan con `GUARDAR=True`.
Si falta Jupyter: instalar `jupyterlab` con pip y ejecutar `jupyter lab`.

## Programa Python independiente
Ejecutar `python analizar.py` para crear resultados CSV, comparación de K y modelo JSON en `resultados/`. Opciones: `--csv archivo.csv --k 3 --eps 1.4 --minimo 3 --salida carpeta`. Si se omite K se escoge el de mayor silueta entre 2 y 6.

## Manual breve
1. Elegir algoritmo y parámetros; pulsar Analizar suelos.
2. Tocar puntos y cambiar ejes para explorar propiedades.
3. Leer perfiles, tabla y silueta; comparar varios K.
4. Descargar CSV, reporte de texto o imprimir resultados.
5. Cargar CSV propio usando las siete columnas/unidades de la plantilla; IDs únicos, valores completos, punto decimal y separador coma; 8–500 filas y al menos seis vectores diferentes.
6. Restaurar demo devuelve los datos publicados.
Los parámetros modificados se aplican al pulsar Analizar; las exportaciones siempre corresponden al último análisis ejecutado.

## Métodos y resultados
Se excluyen etiquetas y columnas no numéricas. StandardScaler sobre siete variables. K-Means scikit-learn con semilla 42 y 30 inicializaciones: K=2, silueta 0.722342, tamaños 27 y 2. La proyección PCA 1+2 conserva aproximadamente 85.57% de la variación. No son porcentajes de precisión.
DBSCAN por defecto: eps=1.4, min_samples=3; un grupo y cuatro casos de ruido. Silueta no aplicable con un solo grupo no ruidoso.
La demo exporta modelos exactos de scikit-learn. CSV propios: Lloyd en JavaScript, 20 inicios deterministas, inicialización lejana, hasta 200 iteraciones; puede converger a otra solución local. DBSCAN y silueta fueron contrastados con scikit-learn.

## Fuente y límites
Delgado Santiago, C. E. (2026). *TERRAGEM – Soil Physicochemical Analysis Dataset* [Conjunto de datos]. Zenodo. https://doi.org/10.5281/zenodo.19103772. CC BY 4.0.
29 registros anonimizados, publicados como casos funcionales. El CSV original tiene MD5 `8596e1c9cdfde356226dcad2f58049cf`, igual al publicado. La selección y el agrupamiento son adaptaciones educativas de AgroGrupos.
No representan Mazamari. Grupos y ruido no son diagnósticos agronómicos ni recomendaciones de fertilización. La muestra pequeña y los casos M-28/M-29 condicionan fuertemente la separación; el cuaderno estudia su sensibilidad.
Documentación: https://scikit-learn.org/stable/modules/clustering.html

## Privacidad
Los CSV se procesan en el navegador. No hay cuentas ni almacenamiento remoto de los datos cargados. Al recargar se restauran los datos de demostración.
