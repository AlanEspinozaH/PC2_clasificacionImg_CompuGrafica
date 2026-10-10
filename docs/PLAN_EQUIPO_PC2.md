# Plan de entrega y trabajo del equipo — PC2

El proyecto clasifica seis símbolos aislados dibujados y propone tres comandos LaTeX. La versión funcional está en `feat/cnn-latex`; la rama `main` aún contiene el código base del profesor. Este documento registra el estado verificado el 10 de octubre de 2026 y una distribución propuesta para cuatro integrantes.

## 1. Qué ya está hecho

- Problema definido: aprender comandos LaTeX a partir de dibujos de α, ξ, ∞, ≡, ∂ y ∈.
- Código adaptado: interfaz Flask, dibujo, recolección con etiquetas, procesamiento común, CNN, tres alternativas y copia del comando.
- HASYv2 preparado: 800 imágenes por clase, 4800 en total; imágenes de 32 × 32 y procedencia documentada.
- Notebook `notebooks/01_entrenamiento_simbolos.ipynb`: nueve celdas de código ejecutadas sin errores, incluida la llamada a `train_model` y sus salidas. Importar la función desde `src/training.py` mantiene el entrenamiento reproducible y compartido con la aplicación; no reemplaza la ejecución del notebook.
- Entrenamiento local reportado y versionado por Alan: 3360 imágenes de entrenamiento, 720 de validación y 720 de prueba; 12 épocas ejecutadas, pesos de la época 7 seleccionados por menor pérdida de validación.
- Métricas HASY: top-1 98,61 %, top-3 99,86 %, F1 macro 0,9861; curvas y matriz de confusión incluidas.
- Piloto manual con mouse: 30 consultas, descrito en la sección 3. La ruta Flask también pasó una comprobación de integración con un PNG de prueba.

El notebook contiene el entrenamiento y los resultados solicitados. Los pesos y datasets permanecen en carpetas excluidas de Git. Cada integrante debe poder ejecutar el proyecto con el modelo correcto; respalden y compartan el directorio completo del experimento por un medio acordado.

## 2. Cobertura de la rúbrica y pendientes

| Criterio | Peso | Evidencia actual | Acción para cerrar |
|---|---:|---|---|
| Código desarrollado | 3 | Repositorio adaptado y aplicación funcional. | Revisión compartida, prueba del entorno de la demo y versión de entrega accesible. |
| Tema escogido | 2 | Clasificador de seis símbolos para aprendizaje de LaTeX. | Explicar usuario, problema, utilidad y alcance de símbolos aislados. |
| Red neuronal | 5 | CNN y notebook ejecutado con pérdida, validación y pesos seleccionados. | Cada responsable debe explicar capas, entrada/salida, aprendizaje y selección de época. Resolver primero si el origen de los datos satisface la consigna. |
| Métricas | 2 | Top-1, top-3, F1 macro y confusión de HASY; piloto manual agregado. | Interpretar errores y distinguir evaluación HASY de uso con mouse. |
| Exposición y conclusiones | 8 | Análisis inicial y propuesta de guion. | Adaptar la plantilla Canva, ensayar con todos y sostener conclusiones con evidencias. |

Exposición y conclusiones representan el **40 % de los 20 puntos**. Todos deben exponer; según la consigna, quien no expone recibe 0 individualmente.

**Pendiente principal: procedencia de la recolección.** La cantidad de 800 ejemplos por clase se alcanza con HASY, pero no se debe presentar como datos dibujados por el equipo. La consigna pide recopilar 500–1000 por categoría y no aclara si acepta un dataset público. Confirmar con el profesor si permite HASY para cubrir esa parte. Si exige capturas propias, falta esa recolección y entrenar/evaluar con ella; 30 consultas de este piloto no la sustituyen. Con seis clases, el requisito equivale a 3000–6000 dibujos etiquetados.

**Pendiente de entrega en GitHub.** PR #1 apunta de `feat/captura-simbolos` a `main`; PR #2 apunta de `feat/cnn-latex` a `feat/captura-simbolos`. Ambas están abiertas. Tras la revisión del grupo, integrar primero PR #1; después cambiar la base de PR #2 a `main`, revisar e integrar. Comprobar que el link final abre el código adaptado y el notebook. Mientras tanto, usar el enlace explícito a `feat/cnn-latex`; un enlace genérico al repositorio abre la versión base actual.

El despliegue es opcional. Una demostración local reproducible permite dedicar el esfuerzo al entrenamiento, las métricas y la exposición.

## 3. Piloto completo de dibujos con mouse

Origen: conteos agregados comunicados por Alan. Se reportan cinco consultas de cada símbolo, realizadas por una persona con mouse. No se aportaron PNG ni predicciones por consulta; estos resultados no provienen de una nueva ejecución automatizada.

| Clase | Símbolo | Consultas | Top-1 | Top-3 |
|---|:---:|---:|---:|---:|
| alpha | α | 5 | 3/5 | 5/5 |
| xi | ξ | 5 | 3/5 | 4/5 |
| infty | ∞ | 5 | 3/5 | 5/5 |
| equiv | ≡ | 5 | 3/5 | 5/5 |
| partial | ∂ | 5 | 3/5 | 5/5 |
| in | ∈ | 5 | 4/5 | 5/5 |
| **Total** | | **30** | **19/30 — 63,33 %** | **29/30 — 96,67 %** |

Las tres propuestas recuperan **10 de los 11 errores** de la primera sugerencia. La diferencia entre HASY y el piloto justifica mantener tres alternativas y evaluar dibujos nuevos, pero no demuestra por sí sola cuál es la causa de los errores. El rendimiento observado en HASY no garantiza el mismo rendimiento para esta persona o para otros dibujantes.

El piloto es pequeño. No permite calcular F1, precisión por clase ni una matriz de confusión porque faltan las etiquetas predichas de cada consulta. El único fallo top-3 reportado corresponde a xi. Con cinco ejemplos por clase, un caso cambia la tasa de una clase en 20 puntos porcentuales.

Datos estructurados: [mouse_piloto.json](mouse_piloto.json).

Para ampliar la evaluación, registrar antes de cada consulta el símbolo esperado y después las tres clases propuestas. Incluir participante con código estable, ID del dibujo, método de entrada, PNG y versión/hash del modelo. Mantener esa colección separada de entrenamiento mientras se evalúa. Si se utilizan sus errores para ajustar el modelo, la siguiente evaluación debe usar dibujos nuevos reservados. No presentar un aumento de accuracy sobre ejemplos ya usados para ajustar como prueba independiente.

## 4. Distribución propuesta entre cuatro estudiantes

Todos revisan la rúbrica, comprenden el flujo completo y preparan sus propias diapositivas. La tabla asigna una responsabilidad principal y evidencias verificables, no autoría exclusiva de todo el código existente.

| Integrante | Responsabilidad principal | Trabajo concreto | Evidencia y exposición |
|---|---|---|---|
| **Alan** | Integración, preprocesamiento y demo. | Revisar Flask/lienzo, explicar el procesamiento compartido, reproducir carga del modelo, ordenar la versión de entrega y coordinar revisión de PR. | Flujo del dibujo al LaTeX, ejemplos antes/después del procesamiento, comandos de ejecución, prueba y demo. |
| **Estudiante 2** | Problema, dataset y calidad de datos. | Resolver con el profesor la aceptación de HASY; revisar conteos, correspondencia de etiquetas, procedencia y duplicados. Coordinar recolección propia si corresponde. | Justificación del tema, catálogo, muestras, cantidades y documento del origen real. |
| **Estudiante 3** | CNN y notebook. | Comprender las capas, pérdida, Adam, dropout y criterio de selección; revisar el notebook ejecutado y su reproducibilidad. Hacer otro experimento solo si hay una hipótesis concreta. | Arquitectura, celda de entrenamiento, curvas, explicación de época 7 y diferencia entre validación y prueba. |
| **Estudiante 4** | Evaluación, errores y conclusiones. | Interpretar top-1/top-3/F1/confusión; organizar consultas nuevas de varios integrantes y conservar errores; comparar HASY con el piloto y redactar conclusiones. Coordinar formato Canva con los aportes de todos. | Tabla de métricas, casos de error, limitaciones y conclusiones sustentadas. |

Para la revisión cruzada: Alan revisa el flujo de datos con estudiante 2; estudiantes 3 y 4 revisan juntos las particiones, métricas y conclusiones. Cada integrante debe conocer qué datos entrenaron el modelo y qué ejemplos se reservaron.

Una ampliación ligera es que los otros tres integrantes hagan también cinco consultas por clase. Junto al piloto de Alan serían 120 consultas de cuatro personas. Es una evaluación más variada, todavía pequeña, y requiere conservar los resultados por consulta para producir métricas auditables. No se inventan esas consultas ni se cuentan como realizadas antes de recibirlas.

## 5. Qué debe dominar y entregar Alan

1. **Cambio respecto al código base:** de I/N/U a seis símbolos, catálogo único, recolección con participantes, entrenamiento reproducible y tres comandos LaTeX.
2. **Flujo de una consulta:** lienzo → PNG → procesamiento → entrada de un canal 32 × 32 → CNN con seis logits → ranking → catálogo y comandos.
3. **Preprocesamiento:** transparencia sobre blanco, conversión a gris/tinta, rechazo de vacío, recorte, conservación de proporciones y centrado. Explicar por qué exportación y predicción deben llamar a la misma función.
4. **Entrenamiento que ejecutó:** 800 ejemplos por clase, semilla 42, partición HASY por imágenes, selección con validación y evaluación de prueba posterior. Los user_id HASY no son personas fiables; para capturas propias se usan códigos reales y separación por participante.
5. **Resultado crítico:** HASY top-1 98,61 % frente a piloto mouse 63,33 %; top-3 ayuda, pero no es certeza calibrada. Los comandos salen del catálogo. Un símbolo fuera de las seis clases puede producir sugerencias incorrectas.
6. **Evidencias:** commit local del notebook `500872b`, reporte del piloto, demo y una contribución documentada que pueda explicar. Los commits identifican versiones; el número de commits no mide calidad ni comprensión.
7. **Reproducibilidad:** modelo local correcto, dependencias instaladas, instrucciones claras y ensayo previo. La instalación/descarga inicial necesita internet; la aplicación y la consulta con el modelo guardado pueden ejecutarse localmente.

Verificación ligera en el entorno del equipo:

```bash
python -m unittest discover -s tests -q
LATEX_MODEL_DIR=models/hasy-notebook python main.py
```

La suite existente tuvo 38 pruebas satisfactorias durante el desarrollo. Para la sustentación importa también poder explicar qué verifican y qué no: las pruebas del software no sustituyen la evaluación con dibujos reales.

## 6. Guion inicial para adaptar a Canva

Sin duración de exposición indicada, repartir tiempos semejantes y reservar espacio para la demo. Ajustar el número de diapositivas cuando llegue la plantilla.

| Diapositiva | Contenido | Expositor |
|---:|---|---|
| 1 | Problema, usuario y alcance: dibujar para aprender LaTeX. | Estudiante 2 |
| 2 | Catálogo, HASY/origen autorizado, cantidades y ejemplos. | Estudiante 2 |
| 3 | Aplicación y flujo completo de una consulta. | Alan |
| 4 | Preprocesamiento común y demo breve. | Alan |
| 5 | CNN: capas, dimensiones, logits y aprendizaje. | Estudiante 3 |
| 6 | Notebook, particiones, curvas y selección de época. | Estudiante 3 |
| 7 | Métricas HASY, matriz y errores observados. | Estudiante 4 |
| 8 | Piloto con mouse, limitaciones y conclusiones. | Estudiante 4 |

Mostrar fragmentos concretos del código y salidas legibles; evitar recitar archivos enteros. Cada integrante prepara una explicación y una conclusión de su parte. Preguntas de ensayo: ¿por qué tres alternativas?, ¿por qué separar validación y prueba?, ¿por qué no tratar los IDs HASY como personas?, ¿qué se puede concluir de 30 consultas?, ¿qué cambia entre HASY y dibujos propios?, ¿qué ocurre con un símbolo desconocido?

## 7. Colaboración y cierre

Clonar el repositorio del equipo, no crear cuatro proyectos inconexos. Hasta integrar la versión funcional en main, crear ramas cortas desde `origin/feat/cnn-latex` y proponer PR hacia `feat/cnn-latex`. Ejemplos: `docs/flujo-demo`, `docs/datos-procedencia`, `docs/cnn-entrenamiento`, `docs/evaluacion-conclusiones`. Después de la integración, usar `main` como base común. Cada PR debe explicar la evidencia o el comportamiento que aporta; coordinar las ediciones del notebook para evitar sobrescribir salidas de otros.

Lista de cierre:

- Confirmar el origen de datos permitido y documentar lo que realmente recopiló el equipo.
- Revisar el notebook con entrenamiento, métricas, figuras y conclusiones coherentes.
- Respaldar modelo y datos; comprobar la carga del modelo en la computadora de la demo.
- Dejar la versión adaptada accesible desde el link de entrega.
- Incorporar la plantilla Canva con aportes y exposición de los cuatro.
- Ensayar las explicaciones y la demo; todos deben exponer.

Base docente: https://github.com/PeterTXS09/letras_uni/tree/main

Repositorio del equipo: https://github.com/AlanEspinozaH/PC2_clasificacionImg_CompuGrafica/tree/feat/cnn-latex
