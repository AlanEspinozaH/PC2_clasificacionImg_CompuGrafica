# Reconocimiento de símbolos matemáticos para LaTeX

Proyecto académico de la **PC2 de Computación Gráfica (UNI)**.

La idea es aprender comandos de **LaTeX de forma interactiva**: el usuario dibuja una letra griega o un símbolo matemático, y una CNN propone tres coincidencias cuando existe un modelo entrenado. El usuario compara las alternativas y copia el comando correcto.

## Estado actual

**Recolección, entrenamiento en CPU y consulta de tres candidatos implementados.**

- Lienzo para mouse, lápiz y pantalla táctil, sin recursos externos de internet.
- Seis categorías definidas en un catálogo común.
- Código de participante y método de entrada registrados con cada dibujo.
- PNG y metadatos guardados en esta computadora.
- Contadores por categoría y propuesta de una categoría con menos muestras.
- Rechazo de lienzos vacíos, categorías inválidas y PNG exactamente repetidos.
- Exportación de imágenes centradas de 32 × 32, etiquetas y participantes.

El modo **Recolectar ejemplos** solicita el símbolo y guarda la categoría indicada por la persona. El modo **Consultar LaTeX** clasifica sin pedir la categoría y muestra símbolo, nombre, comando, ejemplo de sintaxis y botón para copiar.

**El repositorio no incluye pesos entrenados ni una accuracy demostrada.** Primero hay que reunir y revisar los datos. La consulta permanece deshabilitada hasta que exista un modelo compatible; recolectar funciona sin instalar PyTorch.

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

## Ejecutar en Ubuntu

La configuración fijada de dependencias requiere **Python 3.10 a 3.13**. Desde la carpeta del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Abrir **http://127.0.0.1:5000**. Detener el servidor con `Ctrl+C`.

La instalación inicial requiere descargar dependencias; después la recolección funciona sin internet. La aplicación se escucha únicamente en esta computadora. Las rutas de datos se calculan desde el repositorio, independientemente de la carpeta desde la que arranques Python.

## Primera prueba: seis dibujos

1. Usa un código de prueba como **DEMO01**.
2. Dibuja el símbolo indicado y pulsa **Guardar dibujo**.
3. Repite hasta guardar una muestra de cada categoría. El lienzo se borra y propone la siguiente categoría después de guardar.
4. Pulsa **Preparar datos**, después **Descargar colección preparada**.
5. Comprueba que el ZIP incluya `X.npy`, `y.npy`, `participants.npy` y los metadatos.

Haz esta prueba antes de recolectar cientos de muestras. Los dibujos de prueba permanecen en tus archivos originales y en la exportación; el entrenamiento excluye por defecto los participantes cuyo código comienza por `DEMO`. No elimina archivos.

## Recolección

La meta inicial del catálogo es **500 dibujos por categoría**, 3000 en total. La consigna indica entre 500 y 1000 por categoría. Se puede ajustar la meta antes de iniciar el trabajo.

- Asigna un código distinto a cada persona: `P01`, `P02`, etc.
- El mismo participante debe conservar su código en todos sus dibujos. No introduzcas nombres completos.
- Guarda dibujos nuevos en lugar de enviar repetidamente el mismo PNG.
- El dibujo guardado debe corresponder al símbolo solicitado. La revisión humana de las etiquetas sigue siendo necesaria.
- Recolecta estilos de varias personas para poder evaluar dibujantes nuevos.

Para un **piloto del flujo**, pide a tres personas distintas que dibujen cada categoría diez veces: 3 × 6 × 10 = 180 dibujos. Es un ensayo pequeño, no una muestra suficiente para demostrar calidad. Para la entrega, completa la meta y busca más variedad de participantes, idealmente diez o más. Cada persona debe usar un único código estable; cambiar de código no crea un dibujante independiente.

**HASYv2** puede complementar los experimentos; confirmen con el profesor si permite sustituir la recolección propia por un conjunto público. Esta exportación utiliza únicamente los dibujos recolectados en la aplicación. No mezcla imágenes de HASY ni las carpetas originales `I/`, `N/` y `U/`.

## Dónde quedan los datos

| Ruta | Contenido |
|---|---|
| `data/raw/<categoria>/<sample_id>.png` | Dibujo original con fondo blanco. |
| `data/raw/samples.csv` | Categoría, índice, participante, entrada, fecha UTC y hash del PNG. |
| `data/processed/X.npy` | Imágenes `float32` con forma `(N, 32, 32, 1)`. |
| `data/processed/y.npy` | Etiquetas enteras del 0 al 5, alineadas con las imágenes. |
| `data/processed/participants.npy` | Códigos de participantes en el mismo orden. |
| `data/processed/labels.json` | Copia del catálogo usado para preparar los datos. |
| `data/processed/manifest.csv` | Copia de los registros incluidos en la exportación. |
| `data/processed/metadata.json` | Conteos, dimensiones, procesamiento y estado de la meta. |
| `data/processed/dataset.zip` | Los archivos preparados para el entrenamiento. |

El ZIP contiene los arrays y metadatos; **los PNG originales permanecen en `data/raw/`**. Respalda también esa carpeta para conservar la colección original.

`data/raw/`, `data/processed/`, `models/`, `reports/` y `.venv/` están excluidos de Git. Los compañeros deben compartir los datos y modelos por un medio acordado, manteniendo el catálogo y los metadatos. Un push de código no respalda esas carpetas.

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
  --validation-participants P09 --test-participants P10
LATEX_MODEL_DIR=models/final python main.py
```

Los códigos son un ejemplo: deben existir en tu colección y cada partición debe incluir las seis clases. El resto de personas se usa para entrenamiento. Si omites los códigos de validación y prueba, una asignación aleatoria con semilla 42 separa personas completas, aproximadamente 70/15/15; con pocas personas, las proporciones varían. Conserva los grupos de prueba al comparar experimentos: usa los códigos guardados en `split.json`, aunque añadas participantes.

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
| `model_info.json` | Semilla, versiones, número de muestras, propósito piloto/final y huella del dataset. |
| `split.json` | Personas, índices y IDs de muestras de entrenamiento, validación y prueba. |
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
| `scripts/inspect_dataset.py` / `scripts/train.py` | Herramientas de consola. |
| `templates/index.html` | Interfaz de recolección y consulta. |
| `static/app.js` | Lienzo y envío de dibujos. |
| `tests/test_collection.py` | Pruebas de captura y preparación. |
| `tests/test_training.py` | Particiones, métricas, modelos y consulta. |

Las rutas principales son `GET /` para la interfaz, `POST /upload` para guardar, `POST /prepare` para preparar, `GET /dataset.zip` para descargar, `GET /api/model` para consultar disponibilidad y `POST /predict` para obtener candidatos.

## Verificación

Con el entorno activado:

```bash
python -m unittest discover -s tests -v
```

Las pruebas usan directorios temporales: no agregan dibujos a la colección real. Comprueban captura, preprocesamiento, alineación, separación por persona, métricas y correspondencia de comandos. La prueba de entrenamiento utiliza dibujos sintéticos para comprobar el flujo, sin pretender medir reconocimiento real. Las cinco pruebas de CNN se omiten si PyTorch no está instalado.

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
