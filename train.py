from pathlib import Path
import json,io,contextlib,base64,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
(ROOT/'dist/data').mkdir(exist_ok=True)
shutil.copy(ROOT/'data/leaf_original.csv',ROOT/'dist/data/leaf_original.csv')
cells=[];ns={}
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s):
 out=io.StringIO()
 with contextlib.redirect_stdout(out):exec(s,ns)
 outputs=[]
 if out.getvalue():outputs.append(dict(output_type='stream',name='stdout',text=out.getvalue().splitlines(True)))
 for num in plt.get_fignums():
  fig=plt.figure(num);b=io.BytesIO();fig.savefig(b,format='png',bbox_inches='tight',dpi=110)
  outputs.append(dict(output_type='display_data',metadata={},data={'image/png':base64.b64encode(b.getvalue()).decode(),'text/plain':['Gráfico generado por el código anterior']}))
 plt.close('all')
 cells.append(dict(cell_type='code',metadata={},execution_count=sum(c['cell_type']=='code' for c in cells)+1,source=s.splitlines(True),outputs=outputs))
md('''# FlorOrda · Reporte del proyecto HT-01
**Clasificación supervisada de cuatro tipos de plantas mediante características de sus hojas.**

## Objetivo y alcance
Entrenar un árbol de decisión para distinguir alcornoque (*Quercus suber*), adelfa (*Nerium oleander*), arce japonés (*Acer palmatum*) y olivo (*Olea europaea*). Las fotografías de la web son referencias visuales; el programa no procesa imágenes ni clasifica flores por color.

## Cómo ejecutar este manual
Descarga el repositorio completo, abre este cuaderno desde su carpeta `dist` y ejecuta las celdas en orden. Instala antes `pip install -r requirements.txt` desde la raíz. El CSV se distribuye con el proyecto: no necesitas descargar datos adicionales.

## Fuente de los datos
Leaf, UCI Machine Learning Repository, Pedro Silva y André Marçal. DOI: https://doi.org/10.24432/C53C78 . Licencia CC BY 4.0. Archivo numérico obtenido del espejo https://github.com/mljs/ml/blob/main/examples/leafDataset/leaf.csv . Se conserva el original y se filtran las clases 1, 8, 11 y 35. La selección de estas clases y de cuatro características se fija antes de evaluar.

## 1. Flujo del proyecto
El diagrama muestra la separación entre desarrollo y prueba; el conjunto de prueba no interviene en la validación cruzada.
''')
code('''import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(9, 7))
ax.set(xlim=(0,10), ylim=(0,10)); ax.axis('off')
steps=[('Datos Leaf / fuente y licencia',5,9),('Filtrar 4 clases y revisar calidad',5,7.5),('Separación estratificada 75 % / 25 %',5,6),('Entrenar + validar solo en entrenamiento',2.6,4.3),('Reservar prueba sin entrenar',7.4,4.3),('Evaluación final: exactitud, F1 y matriz',5,2.6),('Exportar reglas → web y manual de uso',5,1)]
for text,x,y in steps:
    ax.add_patch(FancyBboxPatch((x-2.2,y-.45),4.4,.9,boxstyle='round,pad=.1',facecolor='#e6eedf',edgecolor='#486749'))
    ax.text(x,y,text,ha='center',va='center',fontsize=9)
for a,b in [(0,1),(1,2),(2,3),(2,4),(3,5),(4,5),(5,6)]:
    _,x,y=steps[a]; _,xx,yy=steps[b]
    ax.annotate('',xy=(xx,yy+.55),xytext=(x,y-.55),arrowprops={'arrowstyle':'->','color':'#486749'})
plt.title('FlorOrda · Flujo reproducible',fontsize=16)
''')
md('## 2. Cargar y reconocer la estructura\nEl CSV tiene 16 columnas: clase, identificador del ejemplar y 14 descriptores. El identificador se excluye del modelo para evitar que aprenda el orden de los registros.')
code('''import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
path = Path('data/leaf_original.csv')
if not path.exists(): path = Path('dist/data/leaf_original.csv')
columns=['clase','ejemplar','excentricidad','relacion_aspecto','elongacion','solidez','convexidad','factor_isoperimetrico','indentacion','lobulos','intensidad','contraste','suavidad','tercer_momento','uniformidad','entropia']
raw=pd.read_csv(path,header=None,names=columns)
names={1:'Alcornoque',8:'Adelfa',11:'Arce japonés',35:'Olivo'}
df=raw[raw.clase.isin(names)].copy()
df['planta']=df.clase.map(names)
print('Original:',raw.shape,'Seleccionado:',df.shape)
print(df.head().to_string(index=False))
''')
md('## 3. Revisar calidad y distribución\nSe cuentan valores faltantes y duplicados completos. No se inventan muestras ni se cambian etiquetas. Los extremos observados se conservan: con pocos datos, eliminar muestras sin justificación distorsionaría el análisis.')
code('''print('Valores faltantes:',df.isna().sum().sum())
print('Duplicados completos:',df.duplicated().sum())
print('Muestras por planta:')
print(df.planta.value_counts().to_string())
assert df.isna().sum().sum()==0
fig,ax=plt.subplots(figsize=(7,3))
df.planta.value_counts().plot.bar(ax=ax,color=['#547f66','#9375ae','#cc9760','#7891a0'],rot=0)
ax.set(ylabel='Muestras',title='Distribución de las cuatro clases')
''')
md('''## 4. Variables utilizadas
| Variable | Interpretación | Unidad |
|---|---|---|
| Excentricidad | Describe qué tan alargada es la elipse asociada a la hoja | Adimensional |
| Relación de aspecto | Proporción geométrica de sus dimensiones | Adimensional |
| Solidez | Relación entre área de la hoja y área de su envolvente convexa | Adimensional |
| Factor isoperimétrico | Descriptor de compacidad del contorno | Adimensional |

Son descriptores calculados a partir de imágenes en el conjunto original. No son centímetros y no se obtienen mirando el color de una fotografía. La web permite explorar ejemplos y editar los valores numéricos, pero no extrae estos descriptores automáticamente.
''')
code('''features=['excentricidad','relacion_aspecto','solidez','factor_isoperimetrico']
print(df[features].describe().round(4).to_string())
print('Correlación:')
print(df[features].corr().round(3).to_string())
fig,ax=plt.subplots(figsize=(7,4))
for k,name in names.items():
    rows=df[df.clase==k]
    ax.scatter(rows.excentricidad,rows.relacion_aspecto,label=name,alpha=.8)
ax.set(xlabel='Excentricidad',ylabel='Relación de aspecto',title='Exploración de medidas de hojas'); ax.legend()
''')
md('## 5. Separar los datos\nEl 75 % se usa para entrenar y el 25 % para prueba final, con estratificación y semilla 42. Los gráficos anteriores exploran los datos; los parámetros del modelo se fijan aquí y no se ajustan mirando los aciertos de prueba.')
code('''X=df[features].to_numpy(); y=df.clase.map({1:0,8:1,11:2,35:3}).to_numpy()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
print('Entrenamiento:',len(y_train),'Prueba:',len(y_test))
print('Clases en entrenamiento:',np.bincount(y_train))
print('Clases en prueba:',np.bincount(y_test))
''')
md('## 6. Entrenamiento y validación cruzada\nUn árbol de decisión genera reglas interpretables. Se limita a profundidad 4 y mínimo 2 muestras por hoja. No requiere estandarizar para sus cortes. La validación de 3 particiones se realiza solo con entrenamiento; el pequeño tamaño de la muestra limita la estabilidad de las métricas.')
code('''model=DecisionTreeClassifier(max_depth=4,min_samples_leaf=2,random_state=42)
cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
scores=cross_val_score(model,X_train,y_train,cv=cv,scoring='accuracy')
model.fit(X_train,y_train)
print('Exactitud en cada partición:',scores)
print(f'Media CV: {scores.mean():.2%} | desviación: {scores.std():.2%}')
fig,ax=plt.subplots(figsize=(12,6))
plot_tree(model,feature_names=features,class_names=list(names.values()),filled=True,rounded=True,ax=ax,fontsize=8)
''')
md('## 7. Evaluación independiente\nExactitud: proporción total de aciertos. Precisión: aciertos entre las predicciones de una clase. Recall: fracción de ejemplares reales recuperados. F1 combina precisión y recall. La matriz muestra clase real en filas y predicha en columnas.')
code('''pred=model.predict(X_test)
accuracy=accuracy_score(y_test,pred)
cm=confusion_matrix(y_test,pred,labels=[0,1,2,3])
print(f'Exactitud de prueba: {accuracy:.2%} ({sum(pred==y_test)}/{len(y_test)})')
print(classification_report(y_test,pred,target_names=list(names.values()),zero_division=0))
fig,ax=plt.subplots(figsize=(7,5))
ConfusionMatrixDisplay(cm,display_labels=list(names.values())).plot(ax=ax,cmap='Greens',colorbar=False)
plt.xticks(rotation=15)
''')
md('## 8. Predicción y exportación\nSe exporta exactamente el árbol evaluado; no se reentrena con las muestras de prueba. El navegador recorre los mismos cortes. Las probabilidades son proporciones de clase en cada hoja del árbol, no una certeza botánica.')
code('''sample=X_train[0:1]
print('Valores de entrada:',dict(zip(features,sample[0])))
print('Predicción:',list(names.values())[model.predict(sample)[0]])
print('Probabilidades:',model.predict_proba(sample)[0])
''')
md('''## 9. Manual de uso de la web
1. Abre el enlace publicado indicado en el README del repositorio.
2. Revisa las cuatro fichas de plantas y sus fotografías reales.
3. Pulsa **Ejemplo** para cargar una muestra real de una clase.
4. Examina o modifica sus cuatro descriptores, dentro de los rangos del conjunto.
5. Pulsa **Clasificar mi planta**. Lee la clase predicha y sus probabilidades.
6. Despliega **¿Cómo llegó a este resultado?** para ver las reglas recorridas.
7. Consulta la matriz y los gráficos para interpretar el desempeño.
8. Descarga este cuaderno y el CSV desde la sección del proyecto.

## 10. Conclusiones y límites
Este programa cumple HT-01: aprende usando datos etiquetados, evalúa muestras separadas y predice una clase. La muestra de cuatro clases es pequeña; los porcentajes no garantizan rendimiento en campo. Una planta ajena a estas cuatro clases también recibirá una de ellas porque no se entrenó una clase “desconocida”. El programa no reconoce fotografías, enfermedades ni calidad de cultivos. Para una ampliación se requieren más hojas independientes, una evaluación externa y un extractor de descriptores consistente.

## 11. Estructura de archivos
- `train.py`: reproduce el análisis, entrenamiento y exportación.
- `data/leaf_original.csv`: original numérico conservado.
- `dist/data/flororda_4_plantas.csv`: subconjunto utilizado.
- `dist/model.js`: reglas del modelo y resultados reales.
- `dist/index.html`, `style.css`, `app.js`: interfaz web.
- `dist/FlorOrda_Analisis_Reporte_Manual.ipynb`: este reporte ejecutado.
- `dist/CREDITOS.md`: procedencia y licencias de las fotos.
- `requirements.txt`: dependencias para reproducir.

## Referencias
Silva, P., y Marçal, A. (2014). *Leaf* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C53C78

Scikit-learn. *DecisionTreeClassifier*. https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
''')
if (ROOT/'MANUAL_FOTOS.md').exists(): md((ROOT/'MANUAL_FOTOS.md').read_text())
old=Path.cwd()
# Cells run from ROOT; dataset path resolution supports both root and dist.
nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3')),nbformat=4,nbformat_minor=4)
(ROOT/'dist/FlorOrda_Analisis_Reporte_Manual.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
ns['df'].to_csv(ROOT/'dist/data/flororda_4_plantas.csv',index=False)
t=ns['model'].tree_
def tree(i):
 if t.children_left[i]==-1:
  v=t.value[i][0];return {'p':(v/v.sum()).tolist()}
 return {'f':int(t.feature[i]),'t':float(t.threshold[i]),'l':tree(t.children_left[i]),'r':tree(t.children_right[i])}
X=ns['X']; y=ns['y']; df=ns['df']
d=dict(tree=tree(0),accuracy=float(ns['accuracy']),cv=float(ns['scores'].mean()),matrix=ns['cm'].tolist(),total=len(y),train=len(ns['y_train']),test=len(ns['y_test']),correct=int(sum(ns['pred']==ns['y_test'])),counts=[int(sum(y==i)) for i in range(4)],ranges=[[float(X[:,i].min()),float(X[:,i].max())] for i in range(4)],examples=[X[y==i][0].tolist() for i in range(4)],points=[dict(x=float(x[0]),y=float(x[1]),c=int(k)) for x,k in zip(X,y)])
(ROOT/'dist/model.js').write_text('const DATA='+json.dumps(d)+';')
# Standalone HTML report, with the same real code outputs and embedded plots.
import html,re
def render_md(text):
    def inline(t):
        t=html.escape(t)
        t=re.sub(r'`([^`]+)`',r'<code>\1</code>',t)
        t=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',t)
        t=re.sub(r'\*([^*]+)\*',r'<em>\1</em>',t)
        return t
    lines=text.splitlines(); out=[]; in_table=False
    for line in lines:
        if line.startswith('|'):
            if not in_table: out.append('<table>'); in_table=True
            if re.match(r'^\|[- |]+\|$',line):continue
            out.append('<tr>'+''.join('<td>'+inline(x.strip())+'</td>' for x in line.strip('|').split('|'))+'</tr>');continue
        if in_table:out.append('</table>');in_table=False
        m=re.match(r'^(#{1,3}) (.*)',line)
        if m: out.append(f'<h{len(m[1])}>'+inline(m[2])+f'</h{len(m[1])}>')
        elif line:out.append('<p>'+inline(line)+'</p>')
    if in_table:out.append('</table>')
    return ''.join(out)
parts=['<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>FlorOrda · Reporte ejecutado</title><style>body{max-width:980px;margin:40px auto;padding:20px;font:15px/1.65 system-ui;color:#183f32;background:#f6f5ef}pre{white-space:pre-wrap;background:white;padding:18px;border:1px solid #ddd;border-radius:8px}img{max-width:100%}section{margin:25px 0}pre,code{overflow-wrap:anywhere}h1{font:42px Georgia}h2{margin-top:35px}table{border-collapse:collapse;width:100%}td{border:1px solid #ccd6c4;padding:9px}p{margin:10px 0}a{color:#735ba2}</style><a href="index.html">← Volver a FlorOrda</a>']
for c in cells:
 s=''.join(c['source'])
 if c['cell_type']=='markdown':parts.append('<section>'+render_md(s)+'</section>')
 else:
  parts.append('<pre>'+html.escape(s)+'</pre>')
  for o in c['outputs']:
   if o['output_type']=='stream':parts.append('<pre>'+html.escape(''.join(o['text']))+'</pre>')
   else:parts.append('<img alt="Gráfico del análisis" src="data:image/png;base64,'+o['data']['image/png']+'">')
(ROOT/'dist/Reporte_FlorOrda.html').write_text('\n'.join(parts))
(ROOT/'dist/HT01_Flores.ipynb').unlink(missing_ok=True)
print(json.dumps({k:d[k] for k in ['accuracy','cv','total','train','test','correct','counts','ranges']}))
