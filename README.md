# Reconocimiento de símbolos matemáticos para LaTeX

Proyecto académico de la **PC2 de Computación Gráfica (UNI)**.

La idea es aprender comandos de **LaTeX de forma interactiva**: el usuario dibuja una letra griega o un símbolo matemático, y una CNN propone tres coincidencias cuando existe un modelo entrenado. El usuario compara las alternativas y copia el comando correcto.

## Guías para integrantes

- [Instalación y colaboración: Ubuntu 24.04 y Windows 11](docs/INSTALACION.md).
- [Cómo completar la contribución de Alan](docs/CONTRIBUCION_ALAN.md).
- [Reparto de exposición: dos diapositivas y 25 % del tiempo por integrante](docs/EXPOSICION.md).
- [Recolección y evaluación de dibujos desconocidos](docs/RECHAZO_DESCONOCIDOS.md).
- [Plan de entrega y cobertura de la rúbrica](docs/PLAN_EQUIPO_PC2.md).

La versión funcional está en `feat/cnn-latex`. Usen esa rama hasta integrar los PR del proyecto en `main`. Para reproducir el entorno se toma Python 3.12 como referencia.

## Estado actual

**Recolección, entrenamiento en CPU y consulta de la CNN implementados.** El rechazo de dibujos fuera del catálogo es preliminar hasta calibrarlo y evaluarlo con ejemplos desconocidos separados.

- Lienzo para mouse, lápiz y pantalla táctil, sin recursos externos de internet.
- Seis categorías definidas en un catálogo común.
- Código de participante y método de entrada registrados con cada dibujo.
- PNG y metadatos guardados en esta computadora.
- Contadores por categoría y propuesta de una categoría con menos muestras.
- Rechazo de lienzos vacíos, categorías inválidas y PNG exactamente repetidos.
- Exportación de imágenes centradas de 32 × 32, etiquetas y participantes.

El modo **Recolectar ejemplos** solicita el símbolo y guarda la categoría indicada por la persona. El modo **Consultar LaTeX** clasifica sin pedir la categoría, muestra las puntuaciones de las tres clases principales cuando acepta la entrada y permite copiar los comandos. Softmax no es una probabilidad calibrada de acierto; un rechazo no muestra alternativas como reconocimientos.

Los dibujos etiquetados como externos al catálogo se pueden recopilar aparte en `data/unknown/`. El procedimiento para compararlos con el rechazo Softmax y calibrar energía exclusivamente con validación está en [RECHAZO_DESCONOCIDOS.md](docs/RECHAZO_DESCONOCIDOS.md). Sin conjuntos desconocidos separados para validación y prueba, no se deben presentar tasas de rechazo ni afirmar que la estrategia mejoró.

El repositorio incluye un **notebook de referencia ejecutado con HASYv2** y el código para reproducirlo. Los pesos se generan localmente y se excluyen de Git. Las métricas de ese experimento evalúan un subconjunto público de seis clases, no los dibujos hechos con mouse por el equipo. La consulta permanece deshabilitada hasta que exista un modelo compatible; recolectar funciona sin instalar PyTorch.

En esta primera versión se reconocen **símbolos aislados**, no expresiones completas.

## Catálogo y etiquetas

El orden de `config/symbols.json` se comparte entre recolección, entrenamiento e interfaz. **No cambies los índices después de recolectar o entrenar sin preparar una migración.**

| Índice | Identificador | Símbolo | Nombre | LaTeX | ID HASY |
|---:|---|:---:|---|---|---:|
| 0 | alpha | α | Alfa minúscula | `\alpha` | 82 |
| 1 | xi | ξ | Xi minúscula | `\xi` | 166 |
| 2 | infty | ∞ | Infinito | `\infty` | 944 |
| 3 | equiv | ≡ | Equivalencia | `\equiv` | 603 |
| 4 | partial | ∂ | Símbolo de derivada parcial | `\partial` | 182 |
| 5 | in | ∈ | Pertenencia | `\in` | 888 |

Los comandos se usan en modo matemático, por ejemplo `\(\alpha\)` o `$x \in A$`.

## Inicio rápido en Ubuntu

La guía del equipo usa **Python 3.12** como referencia; consulta [INSTALACION.md](docs/INSTALACION.md) para requisitos previos, Windows, CPU y notebook. Desde la carpeta del repositorio:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Abrir **http://127.0.0.1:5000**. Detener el servidor con `Ctrl+C`.

La instalación inicial requiere descargar dependencias; después la recolección funciona sin internet. La aplicación se escucha únicamente en esta computadora. Las rutas de datos se calculan desde el repositorio, independientemente de la carpeta desde la que arranques Python.

## Primera prueba: seis dibujos

1. Usa el código que te corresponda: **P01**, **P02**, **P03** o **P04**.
2. Dibuja el símbolo indicado y pulsa **Guardar dibujo**.
3. Repite hasta guardar una muestra de cada categoría. El lienzo se borra y propone la siguiente categoría después de guardar.
4. Pulsa **Preparar datos**, después **Descargar colección preparada**.
5. Comprueba que el ZIP incluya `X.npy`, `y.npy`, `participants.npy` y los metadatos.

Haz esta prueba antes de recolectar cientos de muestras. Los dibujos permanecen en tus archivos originales y en la exportación. Las colecciones antiguas que contengan códigos `DEMO` siguen siendo legibles y el entrenamiento los excluye por defecto; la interfaz nueva solo admite los cuatro códigos asignados. No elimina archivos.

## Recolección

La meta inicial del catálogo es **500 dibujos por categoría**, 3000 en total. La consigna indica entre 500 y 1000 por categoría. Se puede ajustar la meta antes de iniciar el trabajo.

- Los cuatro integrantes usan respectivamente `P01`, `P02`, `P03` y `P04`.
- La interfaz normaliza `p1`, `P1` y `P01` al valor canónico `P01` (y lo mismo para los otros tres códigos); rechaza cualquier código fuera de ese conjunto.
- El mismo participante debe conservar su código en todos sus dibujos. No introduzcas nombres completos.
- Guarda dibujos nuevos en lugar de enviar repetidamente el mismo PNG.
- El dibujo guardado debe corresponder al símbolo solicitado. La revisión humana de las etiquetas sigue siendo necesaria.
- Recolecta estilos de varias personas para poder evaluar dibujantes nuevos.

Para un **piloto del flujo**, pide a los cuatro integrantes que dibujen cada categoría diez veces: 4 × 6 × 10 = 240 dibujos. Es un ensayo pequeño, no una muestra suficiente para demostrar calidad. Cada persona debe usar siempre su único código estable; cambiar de código no crea un dibujante independiente.

### Nombres de los PNG y compatibilidad

Cada captura nueva se guarda como `data/raw/<categoria>/<participante>_<categoria>_<correlativo:04d>_<uuid8>.png`; por ejemplo, `data/raw/alpha/P01_alpha_0001_a8f32b91.png`. El correlativo es independiente para cada combinación de participante y categoría y se reconstruye al iniciar desde el manifiesto y los archivos existentes. El UUID completo, la ruta y el SHA-256 se conservan en `samples.csv` con las mismas columnas de siempre.

Los PNG antiguos llamados `<uuid>.png` no se renombran, eliminan ni modifican. La exportación abre la ruta registrada en `samples.csv`, por lo que admite conjuntamente nombres antiguos y nuevos. Tanto la exportación como el entrenamiento usan el manifiesto, las etiquetas y los arrays como fuente de verdad; no deducen participantes ni clases desde el nombre del archivo.

**HASYv2** puede complementar los experimentos; confirmen con el profesor si permite sustituir la recolección propia por un conjunto público. Esta exportación utiliza únicamente los dibujos recolectados en la aplicación. No mezcla imágenes de HASY ni las carpetas originales `I/`, `N/` y `U/`.

## Notebook y experimento con HASYv2

La consigna solicita **un notebook con entrenamiento y resultados**, además del código modificado. Se incluye `notebooks/01_entrenamiento_simbolos.ipynb`, que utiliza las mismas funciones y CNN de la aplicación. El experimento público es una referencia inicial; la consigna no aclara expresamente si acepta un dataset público para cubrir la recolección. Mantén documentados los datos que realmente recolectó el equipo.

Ya instaladas las dependencias base, CPU y Matplotlib:

```bash
python -m pip install -r requirements-notebook.txt
python -m scripts.run_notebook
LATEX_MODEL_DIR=models/hasy-notebook python main.py
```

El notebook descarga el archivo oficial de 34.6 MB y comprueba su checksum publicado. La importación selecciona **800 muestras por clase**, 4800 en total, entre ejemplos válidos y únicos. Valida los IDs HASY contra los comandos del catálogo, conserva los IDs de origen mediante `HASY_<user_id>` y aplica el procesamiento compartido.

Los conteos originales verificados son:

| Clase | Muestras originales HASYv2 |
|---|---:|
| alpha | 2601 |
| xi | 2567 |
| infty | 2914 |
| equiv | 2518 |
| partial | 2413 |
| in | 2296 |

El archivo descargado queda en `data/external/`; el subconjunto, PNG originales, arrays y procedencia quedan en `data/hasy/`. Se descartan dibujos vacíos, IDs de origen inválidos, duplicados de PNG con fondo normalizado y dibujos idénticos que tienen etiquetas distintas. Tus capturas permanecen en `data/raw/` y no se incorporan automáticamente a este experimento.

También puedes preparar e inspeccionar el subconjunto desde la terminal:

```bash
python -m scripts.prepare_hasy --per-class 800
python -m scripts.inspect_dataset --data data/hasy/processed
```

Si el subconjunto ya existe, no vuelvas a importarlo en el mismo destino; la herramienta lo conserva. El notebook reutiliza un subconjunto compatible y resultados del mismo experimento. Para cambiar los parámetros conserva las versiones anteriores y usa otros directorios de datos/modelos.

En VS Code abre el notebook, selecciona el kernel de `.venv` y ejecuta las celdas en orden. Para entrenar con capturas propias, cambia `SOURCE` a `"own"`; tras completar la meta cambia `PILOT_OWN` a `False` y asigna un nombre de experimento propio. Guarda el notebook con las salidas reales antes de entregar.

**HASY se separa por imágenes estratificadas**, 70/15/15 por clase. El [artículo, sección III](https://arxiv.org/abs/1701.08380) explica que `user_id` no identifica personas fiables: `16925` agrupa contribuciones de Detexify. Se conserva este campo como procedencia y no se presenta como número de autores. Las capturas propias sí se separan por participante con códigos estables. Este protocolo no usa los diez folds oficiales, por lo que sus cifras no se comparan directamente con el benchmark del artículo. El modelo queda marcado como referencia externa, aunque tenga 800 imágenes por clase. Prueba además dibujos nuevos hechos con mouse; una buena cifra en HASY no garantiza ese desempeño.

Resultados del experimento incluido (semilla 42, CPU):

| Medida | Resultado |
|---|---:|
| Entrenamiento / validación / prueba | 3360 / 720 / 720 imágenes |
| Accuracy top-1 | 98.61% (710/720) |
| Accuracy top-3 | 99.86% (719/720) |
| F1 macro | 0.9861 |
| Mejor época / épocas ejecutadas | 7 / 12 |

La confusión más frecuente fue `xi` → `in` (4 imágenes). Top-3 recuperó 9 de los 10 errores de la primera sugerencia. Son resultados de imágenes reservadas de este subconjunto, no una evaluación de dibujantes nuevos ni de las capturas propias.

## Dónde quedan los datos

| Ruta | Contenido |
|---|---|
| `data/raw/<categoria>/<participante>_<categoria>_<correlativo>_<uuid8>.png` | Dibujo original nuevo con fondo blanco; también se admiten los nombres UUID anteriores. |
| `data/raw/samples.csv` | Categoría, índice, participante, entrada, fecha UTC y hash del PNG. |
| `data/processed/X.npy` | Imágenes `float32` con forma `(N, 32, 32, 1)`. |
| `data/processed/y.npy` | Etiquetas enteras del 0 al 5, alineadas con las imágenes. |
| `data/processed/participants.npy` | Códigos de participantes en el mismo orden. |
| `data/processed/labels.json` | Copia del catálogo usado para preparar los datos. |
| `data/processed/manifest.csv` | Copia de los registros incluidos en la exportación. |
| `data/processed/metadata.json` | Conteos, dimensiones, procesamiento y estado de la meta. |
| `data/processed/dataset.zip` | Los archivos preparados para el entrenamiento. |

El ZIP contiene los arrays y metadatos; **los PNG originales permanecen en `data/raw/`**. Respalda también esa carpeta para conservar la colección original.

`data/raw/`, `data/processed/`, `data/external/`, `data/hasy/`, `models/`, `reports/` y `.venv/` están excluidos de Git. Los compañeros deben compartir los datos y modelos por un medio acordado, manteniendo el catálogo y los metadatos. Un push de código no respalda esas carpetas.

## Inspeccionar y entrenar

Pulsa **Preparar datos** después de añadir muestras. Los scripts leen la exportación local en `data/processed/`; no hace falta descargar y descomprimir el ZIP en la misma computadora.

```bash
python -m scripts.inspect_dataset
```

La inspección muestra los conteos por clase y persona, excluye pruebas `DEMO` y comprueba si se pueden formar tres particiones con todas las clases. Si solo hay una persona, **no existe una evaluación de usuarios nuevos**. Reunir más dibujos de esa misma persona no resuelve ese problema.

Instala las dependencias de entrenamiento una vez:

```bash
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements-ml.txt
```

Se fija PyTorch 2.7.1 para CPU; no se requiere CUDA, GPU ni un servicio de pago. `requirements-ml.txt` añade Matplotlib para generar las figuras.

Con el piloto de tres personas y todas las clases presentes:

```bash
python -m scripts.train --pilot --epochs 25
python main.py
```

El modelo queda en `models/symbols/`. Reinicia el servidor si estaba ejecutándose y elige **Consultar LaTeX**. Cambiar de modo limpia el lienzo para evitar guardar una consulta con una etiqueta incorrecta. Las consultas no se añaden automáticamente al dataset.

Para el entrenamiento con la meta completa, omite `--pilot`:

```bash
python -m scripts.train --output models/final \
  --validation-participants P03 --test-participants P04
LATEX_MODEL_DIR=models/final python main.py
```

Los códigos deben existir en tu colección y cada partición debe incluir las seis clases. En el ejemplo, `P01` y `P02` se usan para entrenamiento. Si omites los códigos de validación y prueba, una asignación aleatoria con semilla 42 separa personas completas; con solo cuatro personas, las proporciones necesariamente son aproximadas. Conserva los grupos de prueba al comparar experimentos: usa los códigos guardados en `split.json`.

El entrenamiento rechaza un directorio de salida que ya exista. Para comparar otro experimento utiliza otro `--output`, conservando el modelo y los resultados anteriores.

## CNN y evaluación

La entrada exportada `(N, 32, 32, 1)` se transforma al orden de canales de PyTorch `(N, 1, 32, 32)`:

| Capa | Salida por dibujo |
|---|---|
| Convolución 3 × 3, 16 filtros + ReLU | 16 × 32 × 32 |
| Max pooling 2 × 2 | 16 × 16 × 16 |
| Convolución 3 × 3, 32 filtros + ReLU | 32 × 16 × 16 |
| Max pooling 2 × 2 | 32 × 8 × 8 |
| Flatten + capa de 64 + ReLU + dropout 0.25 | 64 |
| Capa de salida | 6 logits |

Adam minimiza la entropía cruzada. La salida del modelo durante entrenamiento son logits; softmax se aplica al consultar y evaluar. Esta primera CNN no aplica aumentos de datos. Cualquier aumento posterior debe limitarse a entrenamiento y conservar la identidad del símbolo.

Se conserva la época con menor pérdida de **validación** y se detiene tras cinco épocas sin mejora. El conjunto de **prueba** se evalúa una vez después de seleccionar la época, sin usarse para ajustar el modelo.

Cada experimento guarda:

| Archivo en el directorio del modelo | Contenido |
|---|---|
| `model.pt` | Pesos y contrato de arquitectura, catálogo y procesamiento. |
| `model_info.json` | Semilla, versiones, muestras, origen, protocolo de partición y huella del dataset. |
| `split.json` | Índices, IDs de muestras y códigos de origen/participante de cada partición. |
| `history.json` / `learning_curves.png` | Pérdida y accuracy por época. |
| `metrics.json` | Accuracy top-1/top-3, F1 macro y precisión/recall/F1 por clase. |
| `confusion_matrix.png` | Filas = clase real; columnas = predicción. |
| `test_predictions.csv` | Etiqueta real y tres candidatos para cada muestra de prueba. |

**Top-3 merece una interpretación crítica:** con seis clases, tres sugerencias distintas elegidas al azar aciertan el 50 %; top-1 tiene una referencia aleatoria del 16.7 %. Reporta ambas métricas junto con F1 macro, cantidad de participantes y ejemplos de errores. Un ensayo piloto o una buena cifra con una sola persona no demuestra generalización.

Las sugerencias se limitan a estas seis categorías. El modelo no fue entrenado para detectar símbolos desconocidos; puede dar candidatos convincentes para otro dibujo. La interfaz no convierte las puntuaciones softmax en porcentajes de certeza. Los comandos salen del catálogo fijo, no de texto generado por la CNN.

## Procesamiento compartido

`src/preprocessing.py`:

1. Compone los PNG transparentes sobre fondo blanco y convierte a escala de grises.
2. Representa la tinta con valores altos y el fondo con cero.
3. Rechaza lienzos sin trazos suficientes.
4. Recorta la caja que contiene los trazos.
5. Redimensiona conservando las proporciones y centra con un margen de cuatro píxeles.
6. Devuelve una imagen de 32 × 32 con un canal y valores entre 0 y 1.

La exportación y la predicción llaman a **esta misma función**. El entrenamiento carga los arrays ya procesados y valida su convención. Para preparar imágenes de HASY, también deben aplicar la misma convención de fondo y tinta.

La exportación admite una colección pequeña para probar el flujo. `complete_collection` solo es verdadero cuando todas las categorías alcanzan la meta; exportar no significa que el conjunto esté listo para entrenar.

## Organización del código

| Archivo | Responsabilidad |
|---|---|
| `main.py` | Rutas Flask y validación de las capturas. |
| `config/symbols.json` | Categorías, nombres, índices y comandos. |
| `src/catalog.py` | Carga y comprobación del catálogo. |
| `src/preprocessing.py` | Procesamiento común de las imágenes. |
| `src/dataset.py` | PNG, manifiesto CSV y exportación. |
| `src/training_data.py` | Validación de arrays/manifiesto y separación por participante. |
| `src/model.py` / `src/training.py` | CNN y entrenamiento. |
| `src/evaluation.py` | Métricas con orden fijo de clases. |
| `src/prediction.py` | Carga del modelo y tres candidatos. |
| `src/hasy_import.py` / `scripts/prepare_hasy.py` | Subconjunto público con IDs de origen y procedencia. |
| `scripts/inspect_dataset.py` / `scripts/train.py` | Herramientas de consola. |
| `notebooks/01_entrenamiento_simbolos.ipynb` | Entrenamiento, figuras, métricas e integración. |
| `scripts/run_notebook.py` | Ejecuta el notebook y guarda sus salidas. |
| `templates/index.html` | Interfaz de recolección y consulta. |
| `static/app.js` | Lienzo y envío de dibujos. |
| `tests/test_collection.py` | Pruebas de captura y preparación. |
| `tests/test_training.py` | Particiones, métricas, modelos y consulta. |
| `tests/test_hasy.py` | Integridad, selección e independencia del dataset público. |

Las rutas principales son `GET /` para la interfaz, `POST /upload` para guardar, `POST /prepare` para preparar, `GET /dataset.zip` para descargar, `GET /api/model` para consultar disponibilidad y `POST /predict` para obtener candidatos.

## Verificación

Con el entorno activado:

```bash
python -m unittest discover -s tests -v
```

Las pruebas usan directorios temporales: no agregan dibujos a la colección real. Comprueban nombres, normalización de participantes, correlativos por persona y categoría, reinicios, colisiones, nombres UUID heredados, exportación, entrenamiento, preprocesamiento, alineación, métricas, comandos y procedencia de HASY. Las pruebas unitarias de entrenamiento utilizan dibujos sintéticos para comprobar el flujo, sin pretender medir reconocimiento real. Las pruebas de CNN se omiten si PyTorch no está instalado. El notebook de referencia se ejecuta adicionalmente con las imágenes reales de HASYv2.

Después de una captura manual puedes revisar, sin modificar datos:

```bash
find data/raw -type f -name '*.png' | sort | head
head -n 3 data/raw/samples.csv
```

El nombre debe ser legible, mientras `sample_id` en el CSV conserva los 32 caracteres hexadecimales y `image_sha256` conserva el hash completo.

## Trabajo en equipo

Usamos `main` como versión integrada y ramas por tarea. El recolector está pensado para ejecución local en un solo proceso; el manifiesto CSV usa un bloqueo entre los hilos de ese proceso.

Antes de entrenar:

- Revisar ejemplos de las seis categorías y corregir etiquetas incorrectas.
- Confirmar los conteos requeridos y conservar copias de los datos.
- Reservar participantes propios para validación y prueba.
- Guardar los índices de las particiones; aplicar aumentos de datos solo al entrenamiento.
- Evaluar accuracy top-1, accuracy top-3, F1 macro y matriz de confusión.

La rama `feat/cnn-latex` parte de `feat/captura-simbolos`; su Pull Request usa el recolector como base para revisar solo esta etapa. Integra primero el recolector en `main` y luego cambia la base del PR de la CNN a `main` antes de integrarlo.

## Créditos

Desarrollado como actividad académica del curso **Computación Gráfica — UNI**.

Basado en el repositorio del profesor: [PeterTXS09/letras_uni](https://github.com/PeterTXS09/letras_uni).

Referencia de datos: [MartinThoma/HASY](https://github.com/MartinThoma/HASY). Los IDs del catálogo corresponden a su `symbols.csv`.

Instalación de CPU y guardado de pesos: [versiones de PyTorch](https://pytorch.org/get-started/previous-versions/) y [guardar/cargar modelos](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).

## Piloto con mouse y plan del equipo

El piloto manual reportado por Alan incluye 30 consultas: top-1 19/30 (63,33 %) y top-3 29/30 (96,67 %). Es una evaluación pequeña de una persona, basada en conteos agregados; no permite calcular F1 ni matriz de confusión. Se añade al notebook como registro manual separado de las métricas HASY.

Consulta [el plan de entrega y responsabilidades para cuatro integrantes](docs/PLAN_EQUIPO_PC2.md) y [los conteos del piloto](docs/mouse_piloto.json). Antes de entregar, confirmar la aceptación del origen de datos y comprobar que la rama enlazada contiene el código adaptado y el notebook. Actualmente main conserva el código base y las PR de integración siguen abiertas.
