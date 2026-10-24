## 12. Nuevo módulo: identificar 10 categorías por fotografía

### Dos modelos diferentes
El análisis anterior corresponde a cuatro clases y descriptores del conjunto Leaf. El nuevo módulo usa **píxeles de imágenes** y un extractor visual CLIP preentrenado. Las diez categorías son: alcornoque, adelfa, arce japonés, olivo, rosa, tulipán, girasol, margarita, orquídea e hibisco. Algunas son categorías amplias, no especies únicas.

**Modo inicial:** compara el vector de la foto con descripciones textuales de las diez categorías y dos alternativas generales. Es clasificación zero-shot; no constituye un entrenamiento supervisado con fotos de este proyecto. No se midió una exactitud para estas diez clases.

**Mi modelo:** aprende centroides de clase a partir de fotos etiquetadas cargadas por el usuario, usando vectores CLIP de 512 dimensiones normalizados. Se requiere al menos 3 imágenes distintas por cada categoría activa y dos categorías. Para entrenar las diez se necesitan al menos 30 fotos. Se omiten duplicados idénticos por SHA-256.

### Flujo fotográfico
Foto → redimensionado y normalización de CLIP → vector visual de 512 dimensiones → comparación con descripciones (modo inicial) o centroides aprendidos (modo propio) → resultados por similitud → revisión por el usuario.

### Manual de identificación
1. Abre https://flororda.onrender.com/#fotos .
2. Activa el modelo visual; la primera descarga supera 100 MB y puede tardar varios minutos.
3. Selecciona una foto JPG, PNG o WebP (hasta 12 MB), preferiblemente de una sola planta.
4. Selecciona Modelo visual inicial y pulsa Identificar fotografía.
5. Revisa el candidato y las alternativas. Una similitud no es una probabilidad de acierto. Un resultado no concluyente requiere otra foto o una revisión humana.

### Manual de entrenamiento
1. Selecciona el nombre correcto de la categoría.
2. Añade al menos 3 fotos diferentes, con variedad de luz y perspectiva.
3. Repite para otras categorías. Usa fotos nuevas de distintas plantas, no solo recortes de una misma foto.
4. Pulsa Entrenar mi modelo. La validación por exclusión de una foto calcula cada predicción sin usar esa imagen en su centroide.
5. Tras evaluar, se ajustan los centroides finales con todas las fotos disponibles. Prueba el modelo con imágenes nuevas que no participaron en el entrenamiento.
6. Exporta la copia JSON si quieres conservarla o cambiar de equipo. Al importar se reconstruye el modelo pulsando Entrenar.

### Almacenamiento y límites
El navegador almacena etiquetas, hashes y vectores; no conserva las fotos originales. Se procesa localmente, sin enviarlas a una API de reconocimiento. Los pesos y librerías se descargan de Hugging Face y jsDelivr. El entrenamiento propio no modifica los pesos de CLIP. Su desempeño depende de los ejemplos y puede confundir clases similares. No hay prueba externa ni calibración de probabilidades. El 100 % mostrado para 13 hojas del árbol anterior **no** corresponde al reconocimiento fotográfico.

### Referencias del módulo
OpenAI. CLIP: https://github.com/openai/CLIP
Xenova. Pesos ONNX compatibles: https://huggingface.co/Xenova/clip-vit-base-patch32
Hugging Face. Transformers.js 3.8.1: https://huggingface.co/docs/transformers.js/v3.8.1/api/models
