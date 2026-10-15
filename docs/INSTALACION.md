# Instalación y colaboración reproducible

Esta guía permite levantar el proyecto localmente en **Ubuntu 24.04 LTS** y **Windows 11 x64**, con **Python 3.12** y entrenamiento en CPU. No requiere GPU, servicios de pago ni despliegue.

La base de trabajo actual es `feat/cnn-latex`. `main` todavía conserva la aplicación del profesor. Cuando el equipo integre los PR, usará `main` como base y actualizará esta indicación. Si tu instalación ya funciona, conserva tu entorno y comienza en [Colaboración](#7-colaboración-en-ramas).

## 1. Qué instalar y qué se reproduce

| Componente | Versión del proyecto | Uso |
|---|---|---|
| Python | 3.12, de 64 bits | Entorno común entre integrantes. |
| Flask / NumPy / Pillow | 3.1.3 / 2.2.6 / 12.0.0 | Aplicación y procesamiento. |
| PyTorch | 2.7.1+cpu | CNN y consulta del modelo. |
| Matplotlib | 3.10.8 | Figuras del entrenamiento. |
| ipykernel / nbformat | 6.30.1 / 5.10.4 | Notebook y ejecución con salidas. |
| Git | Instalación vigente | Ramas, commits y PR. |

Las dependencias directas están fijadas en `requirements*.txt`. Los pasos siguientes reproducen el procedimiento; pueden existir diferencias numéricas entre sistemas y CPU. Para identificar una ejecución conserva el commit, las versiones instaladas, la huella del dataset, la semilla y los archivos del experimento. Los paquetes transitivos no constituyen un lock completo del entorno.

Trabaja en una carpeta local, con espacio para dependencias, dataset y modelos. Ejecuta cada comando desde la raíz del repositorio. Si una instalación falla, resuelve el error antes del siguiente paso.

## 2. Ubuntu 24.04: terminal Bash

Instala herramientas del sistema una vez:

```bash
sudo apt update
sudo apt install git python3.12 python3.12-venv
git --version
python3.12 --version
```

Clona el repositorio del equipo. El nombre local puede ser `letras_uni` u otro; no cambia los comandos internos.

```bash
mkdir -p ~/proyectos
cd ~/proyectos
git clone --branch feat/cnn-latex https://github.com/AlanEspinozaH/PC2_clasificacionImg_CompuGrafica.git PC2Simbolos
cd PC2Simbolos
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python -m pip install -r requirements-ml.txt -r requirements-notebook.txt
.venv/bin/python -m pip check
```

Se usa el Python del entorno explícitamente; activarlo es opcional. **Instala PyTorch en un comando separado**: su archivo configura el índice CPU de PyTorch. Combinarlo con los demás archivos en una sola instalación puede impedir encontrar paquetes de PyPI.

Para solo recolectar, basta instalar `requirements.txt` y ejecutar `.venv/bin/python main.py`. Instala el resto cuando necesites entrenar, consultar o ejecutar todas las pruebas.

## 3. Windows 11 x64: PowerShell

Instala [Git para Windows](https://git-scm.com/install/windows) y **Python 3.12 x64** desde [Python.org](https://www.python.org/downloads/windows/). Si usas el instalador tradicional, la [versión 3.12.10](https://www.python.org/downloads/release/python-31210/) ofrece “Windows installer (64-bit)”; incluye el lanzador `py`. Si usas Python Install Manager, instala la serie 3.12 con `py install 3.12`. Reabre PowerShell después de instalar.

Verifica que el lanzador encuentre la versión correcta:

```powershell
git --version
py -3.12 --version
py -3.12 -c "import struct; print(struct.calcsize('P') * 8)"
```

Debe indicar Python 3.12.x y `64`. Después:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\proyectos" | Out-Null
Set-Location "$env:USERPROFILE\proyectos"
git clone --branch feat/cnn-latex https://github.com/AlanEspinozaH/PC2_clasificacionImg_CompuGrafica.git PC2Simbolos
Set-Location PC2Simbolos
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m pip check
$env:PYTHONUTF8 = "1"
```

Estos comandos no requieren activar `Activate.ps1` ni modificar la política de ejecución. No copies una `.venv` creada en Linux a Windows: cada computadora crea la suya.

Para solo recolectar, basta `requirements.txt` y `.\.venv\Scripts\python.exe main.py`. Gunicorn está condicionado a sistemas distintos de Windows en ese archivo; para esta demo local se ejecuta Flask con `main.py`.

## 4. Comprobar la instalación

Con todas las dependencias instaladas:

**Ubuntu**

```bash
.venv/bin/python -c "import sys, torch, numpy; print(sys.executable); print(sys.version); print(torch.__version__, numpy.__version__); print('CUDA:', torch.cuda.is_available())"
.venv/bin/python -m unittest discover -s tests -q
```

**Windows / PowerShell**

```powershell
.\.venv\Scripts\python.exe -c "import sys, torch, numpy; print(sys.executable); print(sys.version); print(torch.__version__, numpy.__version__); print('CUDA:', torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Espera `2.7.1+cpu`, `2.2.6` y `CUDA: False`. La suite existente tiene 38 pruebas; al final debe mostrar `OK`. Sus entrenamientos sintéticos breves comprueban el software: no entrenan el modelo de entrega.

Para una instalación de recolección sin PyTorch, usa en cambio `-m unittest discover -s tests -p test_collection.py -q` con el Python de tu entorno.

**Alcance de la verificación:** el proyecto y su suite se han ejecutado en Linux con las dependencias fijadas. Los comandos de PowerShell y las rutas de Windows están documentados, pero esta guía no registra una ejecución real en Windows 11. Un integrante debe realizarla y conservar su resultado antes de usar ese equipo en la exposición.

## 5. Dataset, notebook y modelo

Git no incluye `data/raw/`, `data/hasy/` ni `models/`. Clonar descarga el código y el notebook ejecutado; **no descarga los pesos**. El notebook versionado permite leer los resultados de Alan sin ejecutarlo nuevamente.

Elige el recorrido que necesitas:

### A. Reproducir el entrenamiento en tu equipo

Conserva la configuración de HASY del notebook: `SOURCE="hasy"`, 800 imágenes por clase y semilla 42. La primera ejecución descarga el archivo original, prepara 4800 imágenes y entrena en CPU.

**Ubuntu**

```bash
.venv/bin/python -m scripts.run_notebook
```

**Windows / PowerShell**

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m scripts.run_notebook
```

Este comando guarda las salidas en `notebooks/01_entrenamiento_simbolos.ipynb` y el experimento en `models/hasy-notebook/`. Si hay una configuración o huella distinta en ese directorio, el notebook se detiene para conservarla: cambia `EXPERIMENT` por un nombre nuevo, por ejemplo `hasy-mi-equipo`, y usa ese mismo directorio al arrancar la aplicación. No borres el experimento de otro integrante para resolver el mensaje.

Si ya existen datos y un experimento compatibles, se leen los resultados guardados. Ese recorrido no cuenta como un entrenamiento nuevo.

También puedes abrir el notebook en VS Code con las extensiones Python y Jupyter, seleccionar el kernel `.venv` y ejecutar todas las celdas. Coordina con el responsable de CNN antes de subir cambios al notebook compartido; el entrenamiento existente de Alan debe preservarse.

### B. Probar la aplicación con el modelo de Alan

Solicita al equipo una copia del **directorio completo** de su experimento, con `model.pt`, `model_info.json`, métricas, historial y figuras. Colócalo como `models/hasy-notebook/`. Conserva el origen y la versión de esos pesos.

Este recorrido permite consultar sin entrenar. No mezcles sus resultados con un dataset preparado por separado ni los presentes como entrenamiento propio. Para reproducir después, usa un `EXPERIMENT` nuevo.

### Arrancar la consulta

**Ubuntu**

```bash
LATEX_MODEL_DIR=models/hasy-notebook .venv/bin/python main.py
```

**Windows / PowerShell**

```powershell
$env:LATEX_MODEL_DIR = "models/hasy-notebook"
.\.venv\Scripts\python.exe main.py
```

Abre [http://127.0.0.1:5000](http://127.0.0.1:5000). `LATEX_MODEL_DIR` debe apuntar al directorio que contiene `model.pt` y `model_info.json`, no al archivo de pesos. Detén el servidor con `Ctrl+C`. Después de instalar y guardar el modelo, esta consulta funciona localmente sin internet.

Comprueba: modelo disponible, tres candidatos, comandos copiables, lienzo vacío rechazado y consultas que no aumentan los contadores de recolección. Si guardas ejemplos de prueba, usa un código `DEMO01`; el entrenamiento excluye los códigos que empiezan con `DEMO`.

Para trabajar con capturas propias, el responsable de datos debe preparar y revisar su colección antes de cambiar `SOURCE` a `"own"`. HASY y las capturas del equipo tienen procedencias y protocolos distintos. El equipo aún debe confirmar si el dataset público satisface el requisito de recopilar 500–1000 imágenes por categoría.

## 6. Registro mínimo de reproducibilidad

Registra `git rev-parse HEAD`, sistema operativo y versión de Python junto con el resultado de las pruebas. El notebook ya registra versiones, particiones y la huella del dataset.

Puedes conservar un listado del entorno y el hash de los pesos:

**Ubuntu**

```bash
mkdir -p reports
.venv/bin/python -m pip freeze > reports/entorno-local.txt
sha256sum models/hasy-notebook/model.pt
```

**Windows / PowerShell**

```powershell
New-Item -ItemType Directory -Force reports | Out-Null
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 reports/entorno-local.txt
Get-FileHash models/hasy-notebook/model.pt -Algorithm SHA256
```

`reports/` está excluido de Git. Comparte el registro por el medio acordado o resume los datos pertinentes en la descripción del PR. El listado instalado es evidencia de ese equipo; no reemplaza los archivos de dependencias del proyecto.

## 7. Colaboración en ramas

Cada integrante usa su cuenta de GitHub y su autor de Git. Si aún no lo configuró, sustituye estos ejemplos por sus propios datos:

```bash
git config user.name "Tu nombre"
git config user.email "Tu correo de GitHub o correo noreply"
git remote -v
git status
```

Estos comandos suponen acceso de colaborador y `origin` apuntando al repositorio del equipo. El propietario concede ese acceso. Con un fork, `origin` apunta a tu fork y debes añadir el repositorio del equipo como `upstream`, actualizar desde él y dirigir el PR al repositorio del equipo; adapta el flujo antes de usar los comandos siguientes.

Con el árbol limpio, actualiza la rama común antes de empezar:

```bash
git fetch origin
git switch feat/cnn-latex
git pull --ff-only
```

Crea **una** rama según tu responsabilidad:

| Integrante | Rama sugerida | Archivo principal de evidencia |
|---|---|---|
| Alan | `docs/flujo-demo` | `docs/FLUJO_Y_DEMO.md` y notebook separado de preprocesamiento. |
| Estudiante 2 | `docs/datos-procedencia` | `docs/DATOS.md`. |
| Estudiante 3 | `docs/cnn-entrenamiento` | Notebook de entrenamiento y `docs/CNN.md`. |
| Estudiante 4 | `docs/evaluacion-conclusiones` | `docs/EVALUACION.md` y resultados auditables. |

Ejemplo para Alan; el resto sustituye el nombre de su rama:

```bash
git switch -c docs/flujo-demo
```

Al finalizar, selecciona los archivos de tu cambio, revisa y sube:

```bash
git add docs/FLUJO_Y_DEMO.md
git diff --cached --stat
git diff --cached
git commit -m "docs: flujo y verificación de la demo"
git push -u origin docs/flujo-demo
```

Abre un PR con **base `feat/cnn-latex`** y tu rama como comparación. Incluye qué cambiaste, evidencia y validación. Otro integrante lo revisa. Si tienes cambios locales pendientes, guárdalos en tu rama antes de cambiar de rama; si `--ff-only` falla, revisa la divergencia en lugar de usar `reset --hard` o un push forzado.

El notebook compartido tiene un responsable de edición: estudiante 3. Los demás envían observaciones o usan notebooks separados. Nadie sube entornos virtuales, datasets, pesos ni datos personales con `git add .`.

Cuando el equipo integre la versión funcional en `main`, actualice esta guía y cree las nuevas ramas desde `origin/main`. El [plan del equipo](PLAN_EQUIPO_PC2.md) describe las bases de los PR #1 y #2; no basta integrar PR #2 dejando su base en la rama de captura.

## 8. Problemas frecuentes

| Mensaje o situación | Qué revisar |
|---|---|
| `py` no existe o no encuentra 3.12 | Reabrir PowerShell, comprobar el lanzador/Install Manager y que Python 3.12 esté instalado. |
| `externally-managed-environment` en Ubuntu | Usar `.venv/bin/python -m pip`, sin instalar paquetes en el Python del sistema. |
| Falta `venv` / `ensurepip` | Instalar `python3.12-venv` y crear el entorno con Python 3.12. |
| No encuentra una distribución de `torch` | Verificar Python 3.12 x64 y ejecutar `requirements-cpu.txt` por separado con acceso al índice CPU. |
| Modelo no disponible / respuesta 503 | Verificar PyTorch instalado, ruta de `LATEX_MODEL_DIR` y directorio completo del modelo. Recolectar sigue disponible. |
| Puerto 5000 ocupado | Detener la instancia anterior en su terminal con `Ctrl+C` antes de arrancar otra. |
| No llegan cambios del equipo | Comprobar `git status`, rama, `origin` y actualización con `git fetch origin`. |
| El notebook cambia al reproducirlo | Es normal que cambien salidas; acordar qué ejecución conservar antes de proponer el PR. |

## Referencias oficiales

- [Entorno Python en Ubuntu](https://ubuntu.com/developers/docs/howto/python-setup/).
- [Entornos virtuales y uso sin activación](https://docs.python.org/3.12/library/venv.html).
- [Lanzador Python en Windows](https://docs.python.org/3.12/using/windows.html).
- [Python Install Manager](https://docs.python.org/3/using/windows.html).
- [PyTorch 2.7.1: instalación CPU en Linux y Windows](https://pytorch.org/get-started/previous-versions/#v2-7-1).
