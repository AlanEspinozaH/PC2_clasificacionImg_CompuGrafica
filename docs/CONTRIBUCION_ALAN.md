# Cómo cerrar la contribución de Alan

Tu responsabilidad principal es **integración, preprocesamiento y demo**. Ya ejecutaste y subiste el entrenamiento en `500872b`; no necesitas volver a entrenar para aportar otra evidencia. La exposición de las capas corresponde al estudiante 3 y la interpretación final de métricas al estudiante 4, aunque todos deben entender el proyecto para responder preguntas.

Esta guía propone trabajo pendiente. Una casilla se marca cuando tienes la evidencia; no declara realizadas capturas o pruebas nuevas.

## 1. Preparar una rama y confirmar el estado

En tu carpeta actual `letras_uni`, con cambios anteriores guardados y árbol limpio:

```bash
git status
git fetch origin
git switch feat/cnn-latex
git pull --ff-only
git switch -c docs/flujo-demo
```

Si esa rama ya existe, usa `git switch docs/flujo-demo`. Conserva tu entorno que ya funciona; la [guía de instalación](INSTALACION.md) está pensada también para compañeros que comienzan desde cero.

## 2. Explicar el flujo usando el código real

Revisa estos archivos y escribe `docs/FLUJO_Y_DEMO.md` con una explicación breve en tus palabras:

| Archivo | Qué debes poder señalar |
|---|---|
| `static/app.js` | Cómo se dibuja, se genera el PNG y se envía a guardar o consultar. |
| `main.py` | Diferencia entre `/upload` y `/predict`, rechazo de entradas y carga del modelo. |
| `src/preprocessing.py` | Transparencia sobre blanco, escala de gris, recorte, proporción, centrado y valores. |
| `src/image_input.py` y `src/dataset.py` | Cómo consulta y exportación aplican el mismo procesamiento. |
| `config/symbols.json` | Cómo la clase predicha se convierte en símbolo y comando LaTeX. |

Incluye cuatro apartados: **cambio frente a I/N/U**, **flujo de una consulta**, **procesamiento con evidencia** y **prueba de la demo**. Usa un diagrama breve o una captura anotada. Explica la CNN como un componente con entrada y salida; deja sus capas y optimización al estudiante 3.

La imagen preparada es `float32`, de `32 × 32 × 1`, fondo 0 y tinta cercana a 1. PyTorch organiza la entrada por lote como `N × 1 × 32 × 32`. Se ajusta el recorte a un área de hasta `24 × 24`, se conserva su proporción y se centra con margen.

## 3. Obtener una evidencia visual propia

Crea `notebooks/02_preprocesamiento_demo.ipynb` y selecciona tu entorno Python. Haz primero un dibujo de prueba en **Recolectar ejemplos**, con `DEMO01`. Ese PNG estará en `data/raw/` y no contará como dato válido para entrenar.

Usa estas celdas en el notebook nuevo; no edites el de entrenamiento:

```python
from pathlib import Path
import sys

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "config/symbols.json").is_file()), None)
if ROOT is None:
    raise RuntimeError("Abre el notebook dentro del repositorio.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

```python
from PIL import Image
import matplotlib.pyplot as plt
from src.preprocessing import flatten_on_white, preprocess_image

files = sorted((ROOT / "data" / "raw").rglob("*.png"))
if not files:
    raise RuntimeError("Guarda antes un dibujo DEMO01 en la aplicación.")
path = files[0]  # Elige otro PNG de esta lista si quieres mostrar otro símbolo.
with Image.open(path) as image:
    original = flatten_on_white(image)
    prepared = preprocess_image(image)

fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("PNG original")
axes[1].imshow(1 - prepared[..., 0], cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Procesado: 32 x 32")
for ax in axes:
    ax.axis("off")
(ROOT / "reports").mkdir(exist_ok=True)
fig.savefig(ROOT / "reports" / "preprocesamiento.png", dpi=160)
print(path.relative_to(ROOT), prepared.shape, prepared.dtype)
plt.show()
```

La inversión `1 - prepared` es solo para mostrar tinta negra sobre fondo blanco en la figura; no cambia la entrada del modelo. Ejecuta y guarda el notebook con su figura.

Repite con un símbolo alargado y otro dibujado cerca de un borde. Describe qué cambia y qué se conserva. Esa comparación permite explicar el procesamiento; no mide accuracy. Puedes usar la figura de `reports/preprocesamiento.png` en tus diapositivas. No inventes imágenes o resultados para completar la evidencia.

## 4. Verificar y ensayar la demo

En el entorno que ya tienes activo:

```bash
python -m unittest discover -s tests -q
LATEX_MODEL_DIR=models/hasy-notebook python main.py
```

Si tu entorno no está activo o usas Windows, utiliza los comandos equivalentes de [INSTALACION.md](INSTALACION.md). Comprueba en tu computadora y registra el resultado real:

- [ ] El modelo disponible corresponde a tu experimento; conservaste su hash y commit.
- [ ] Dibujas un símbolo y recibes tres propuestas con símbolo, nombre y comando.
- [ ] Copias el comando y muestras un uso en modo matemático.
- [ ] Una consulta no aumenta los contadores de recolección.
- [ ] El lienzo vacío produce un mensaje útil.
- [ ] Puedes repetir el arranque con los comandos documentados.

Para exponer, ensaya **dibujar → consultar → comparar → copiar** en 30–45 segundos, dentro de tu cuarto del tiempo total. No depende de que el primer resultado acierte: explica cómo el usuario elige entre propuestas. Lleva capturas del recorrido real como respaldo si falla la demo. El estudiante 4 explicará cuánto ayudan las tres alternativas según la evaluación.

Registra en `docs/FLUJO_Y_DEMO.md` tu sistema, versiones principales, comando de arranque, resultado de la suite y fecha de la prueba. Si aún no verificaste Windows, escríbelo; un compañero completa esa comprobación. Evita atribuirte autoría exclusiva de todo el código o convertir una prueba de software en evidencia de calidad del clasificador.

## 5. Subir la evidencia y obtener revisión

Revisa que el notebook nuevo conserve la figura y sus salidas:

```bash
git add docs/FLUJO_Y_DEMO.md notebooks/02_preprocesamiento_demo.ipynb
git diff --cached --stat
git diff --cached -- docs/FLUJO_Y_DEMO.md
git commit -m "docs: evidencia del procesamiento y demo local"
git push -u origin docs/flujo-demo
```

Abre PR hacia `feat/cnn-latex` y pide al estudiante 2 que revise el procesamiento y los comandos de arranque. Incluye qué comprobaste y cualquier limitación encontrada. No subas `data/` ni `models/`.

Tu contribución queda lista cuando entregas el documento, notebook visual, verificación real, PR revisado y **tus dos diapositivas ensayadas**. La integración de los PR del proyecto se coordina después de la revisión; no reemplaza las contribuciones de los compañeros.

## Orden recomendado para tu próxima sesión

1. Actualiza el repositorio y crea la rama de tu evidencia.
2. Guarda dibujos DEMO y ejecuta el notebook pequeño de preprocesamiento.
3. Escribe el flujo y registra la verificación.
4. Sube el PR y prepara las diapositivas 3–4 del [reparto de exposición](EXPOSICION.md).

No vuelvas a ejecutar el entrenamiento compartido para esta tarea. Si al revisar encuentras un fallo de código, documéntalo y corrígelo en una rama de alcance concreto antes de cerrar la demo.
