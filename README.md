# Reconocimiento de símbolos matemáticos para LaTeX

Proyecto académico de la **PC2 de Computación Gráfica (UNI)**.

La idea es aprender comandos de **LaTeX de forma interactiva**: el usuario dibuja a mano una letra griega o un símbolo matemático, y la aplicación intenta reconocerlo mediante una red neuronal convolucional (CNN).

## ¿Cómo funcionará?

1. El usuario dibuja **un símbolo a la vez** en un lienzo.
2. El modelo propone las **tres coincidencias más probables**.
3. El usuario elige la opción correcta y consulta el nombre, el símbolo y su comando LaTeX.
4. Puede copiar el comando para utilizarlo en sus apuntes o documentos.

Por ejemplo, si dibuja **α**, la aplicación podría sugerir «Alfa» y mostrar `\alpha`.

## Símbolos iniciales

| Símbolo | Nombre | LaTeX |
|:---:|---|---|
| α | Alfa | `\alpha` |
| ξ | Xi | `\xi` |
| ∞ | Infinito | `\infty` |
| ≡ | Equivalencia | `\equiv` |
| ∂ | Derivada parcial | `\partial` |
| ∈ | Pertenencia | `\in` |

En esta primera versión se busca reconocer **símbolos aislados**, no expresiones matemáticas completas.

## Tecnologías previstas

- **Python y Flask:** aplicación web local.
- **HTML y JavaScript:** lienzo de dibujo.
- **CNN:** clasificación de imágenes dibujadas a mano.
- **NumPy y herramientas de procesamiento de imágenes:** preparación de datos.

## Estado del proyecto

**En desarrollo.** Se parte del repositorio del profesor, que permite dibujar y recolectar imágenes de las letras **U, N e I**. La adaptación a los seis símbolos, el entrenamiento de la CNN y las recomendaciones de LaTeX están pendientes.

El trabajo contempla dos modos: **recolección de dibujos etiquetados** y **consulta de símbolos**. Se evaluará el clasificador con métricas como accuracy top-1, accuracy top-3 y matriz de confusión.

## Ejecutar la aplicación base en Ubuntu

Requiere Python 3 y `pip`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Abrir **http://127.0.0.1:5000** en el navegador.

> Por ahora, estos comandos ejecutan el **recolector original de U, N e I**, no el clasificador final.

## Créditos

Desarrollado como actividad académica del curso **Computación Gráfica — UNI**.

Basado en el proyecto de referencia del profesor: [PeterTXS09/letras_uni](https://github.com/PeterTXS09/letras_uni).

