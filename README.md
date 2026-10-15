# Reconocimiento de símbolos matemáticos para LaTeX

Proyecto académico de la **PC2 de Computación Gráfica (UNI)**.

La idea es aprender comandos de **LaTeX de forma interactiva**: el usuario dibuja una letra griega o un símbolo matemático, y una CNN propondrá tres coincidencias. El usuario podrá elegir la correcta, consultar su representación y copiar el comando.

## Estado actual

**Etapa 1: recolección y preparación de datos implementadas.**

- Lienzo para mouse, lápiz y pantalla táctil, sin recursos externos de internet.
- Seis categorías definidas en un catálogo común.
- Código de participante y método de entrada registrados con cada dibujo.
- PNG y metadatos guardados en esta computadora.
- Contadores por categoría y propuesta de una categoría con menos muestras.
- Rechazo de lienzos vacíos, categorías inválidas y PNG exactamente repetidos.
- Exportación de imágenes centradas de 32 × 32, etiquetas y participantes.

**Pendiente:** notebook de entrenamiento, CNN, métricas de clasificación y modo de consulta con las tres sugerencias. La pantalla actual solicita qué dibujar; todavía no predice su categoría.

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

Requiere **Python 3.10 o superior**. Desde la carpeta del repositorio:

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

Haz esta prueba antes de recolectar cientos de muestras. Los dibujos de prueba quedan en tus datos: para una colección definitiva puedes usar otro directorio de trabajo o revisar manualmente esas muestras.

## Recolección

La meta inicial del catálogo es **500 dibujos por categoría**, 3000 en total. La consigna indica entre 500 y 1000 por categoría. Se puede ajustar la meta antes de iniciar el trabajo.

- Asigna un código distinto a cada persona: `P01`, `P02`, etc.
- El mismo participante debe conservar su código en todos sus dibujos. No introduzcas nombres completos.
- Guarda dibujos nuevos en lugar de enviar repetidamente el mismo PNG.
- El dibujo guardado debe corresponder al símbolo solicitado. La revisión humana de las etiquetas sigue siendo necesaria.
- Recolecta estilos de varias personas para poder evaluar dibujantes nuevos.

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

`data/raw/`, `data/processed/` y `.venv/` están excluidos de Git. Los compañeros deben compartir los datos por un medio acordado, manteniendo el catálogo y los metadatos. Un push de código no respalda esas carpetas.

## Procesamiento compartido

`src/preprocessing.py`:

1. Compone los PNG transparentes sobre fondo blanco y convierte a escala de grises.
2. Representa la tinta con valores altos y el fondo con cero.
3. Rechaza lienzos sin trazos suficientes.
4. Recorta la caja que contiene los trazos.
5. Redimensiona conservando las proporciones y centra con un margen de cuatro píxeles.
6. Devuelve una imagen de 32 × 32 con un canal y valores entre 0 y 1.

El futuro notebook y la predicción deberán llamar a **esta misma función**. Para preparar imágenes de HASY, también deben aplicar la misma convención de fondo y tinta.

La exportación admite una colección pequeña para probar el flujo. `complete_collection` solo es verdadero cuando todas las categorías alcanzan la meta; exportar no significa que el conjunto esté listo para entrenar.

## Organización del código

| Archivo | Responsabilidad |
|---|---|
| `main.py` | Rutas Flask y validación de las capturas. |
| `config/symbols.json` | Categorías, nombres, índices y comandos. |
| `src/catalog.py` | Carga y comprobación del catálogo. |
| `src/preprocessing.py` | Procesamiento común de las imágenes. |
| `src/dataset.py` | PNG, manifiesto CSV y exportación. |
| `templates/index.html` | Interfaz de recolección. |
| `static/app.js` | Lienzo y envío de dibujos. |
| `tests/test_collection.py` | Pruebas de captura y preparación. |

Las rutas principales son `GET /` para la interfaz, `POST /upload` para guardar, `POST /prepare` para preparar y `GET /dataset.zip` para descargar.

## Verificación

Con el entorno activado:

```bash
python -m unittest discover -s tests -v
```

Las pruebas usan directorios temporales: no agregan dibujos a la colección real. Comprueban fondo transparente y blanco, centrado, proporciones, validación, duplicados y alineación entre imágenes, etiquetas y participantes.

## Trabajo en equipo y siguiente etapa

Usamos `main` como versión integrada y ramas por tarea. El recolector está pensado para ejecución local en un solo proceso; el manifiesto CSV usa un bloqueo entre los hilos de ese proceso.

Antes de entrenar:

- Revisar ejemplos de las seis categorías y corregir etiquetas incorrectas.
- Confirmar los conteos requeridos y conservar copias de los datos.
- Reservar participantes propios para validación y prueba.
- Guardar los índices de las particiones; aplicar aumentos de datos solo al entrenamiento.
- Evaluar accuracy top-1, accuracy top-3, F1 macro y matriz de confusión.

## Créditos

Desarrollado como actividad académica del curso **Computación Gráfica — UNI**.

Basado en el repositorio del profesor: [PeterTXS09/letras_uni](https://github.com/PeterTXS09/letras_uni).

Referencia de datos: [MartinThoma/HASY](https://github.com/MartinThoma/HASY). Los IDs del catálogo corresponden a su `symbols.csv`.
