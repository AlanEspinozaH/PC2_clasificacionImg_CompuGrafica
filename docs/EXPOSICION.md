# Exposición de cuatro integrantes: responsabilidades y límites

Propuesta para adaptar a la plantilla Canva cuando esté disponible. Usa **ocho diapositivas, dos por persona, y 25 % del tiempo para cada integrante**. Reemplaza “estudiante 2–4” por los nombres acordados.

Si la exposición dura `T` minutos, cada bloque ocupa `T/4`. Por ejemplo, para 12 minutos son 3 por persona. La demo está incluida en el tiempo de Alan; las transiciones están incluidas en cada bloque. Ajusten la densidad si el profesor fija otra duración.

## 1. Reparto del trabajo y del discurso

| Orden / responsable | Diapositivas y pregunta central | Entregable a preparar | Límite para evitar solapamiento |
|---|---|---|---|
| **1. Estudiante 2 — problema y datos** | **1:** usuario, problema y seis símbolos. **2:** de dónde vienen los datos y cómo se revisaron. | `docs/DATOS.md`, catálogo, cantidades, procedencia y muestras. | Presenta datos y calidad de etiquetas. Deja el procesamiento de píxeles a Alan, las particiones al estudiante 3 y los resultados al estudiante 4. |
| **2. Alan — aplicación y procesamiento** | **3:** cómo pasa un dibujo a una consulta LaTeX. **4:** original frente a procesado y demo breve. | `docs/FLUJO_Y_DEMO.md`, notebook visual, instrucciones y demo comprobada. | Explica interfaz, rutas, entrada y catálogo. Menciona la CNN como componente; deja capas y entrenamiento al estudiante 3. Deja porcentajes y conclusiones de rendimiento al estudiante 4. |
| **3. Estudiante 3 — red y entrenamiento** | **5:** arquitectura de la CNN. **6:** particiones, aprendizaje, curvas y selección del modelo. | `docs/CNN.md` y revisión del notebook ejecutado, preservando sus salidas. | Explica tamaños, pérdida, optimizador, regularización y validación. No vuelve a justificar el dataset ni recita la demo. El estudiante 4 interpreta el conjunto de prueba y sus errores. |
| **4. Estudiante 4 — evaluación y conclusiones** | **7:** top-1, top-3, F1 y confusión de HASY. **8:** piloto con mouse, limitaciones y conclusiones. | `docs/EVALUACION.md`, resultados trazables y casos de error. | Define e interpreta métricas y evidencia. No repite capas, código de Flask ni historia del dataset. Cierra con lo que los resultados permiten afirmar y la mejora siguiente. |

**Equidad de trabajo:** cada persona escribe sus dos diapositivas, entrega su evidencia y revisa el trabajo de otra. No se asigna toda la presentación a una persona. El formato Canva se aplica con una plantilla común; el estudiante 4 coordina la coherencia de las conclusiones, no redacta las ocho diapositivas.

Si se requiere recolección propia, todos aportan una cuota semejante en las seis categorías; estudiante 2 coordina etiquetas y conteos. Con cuatro dibujantes y meta de 500 por clase, una división inicial es 125 por clase/persona, 750 por integrante. Buscar más dibujantes mejora la variedad; esa división de trabajo no determina por sí sola la partición de entrenamiento.

### Archivos de referencia para cambios necesarios

El código principal ya existe. Cada responsable revisa y explica su parte; si encuentra una mejora, limita su PR a ese alcance y coordina cualquier archivo compartido.

| Responsable | Área de código |
|---|---|
| Alan | `main.py`, `static/`, `templates/`, `src/image_input.py`, `src/preprocessing.py` y carga/consulta en `src/prediction.py`. |
| Estudiante 2 | Procedencia y colección en `src/dataset.py`, `src/hasy_import.py` y `scripts/prepare_hasy.py`. Revisa el catálogo sin cambiar sus índices ya usados. |
| Estudiante 3 | `src/model.py`, `src/training.py`, `src/training_data.py`, `scripts/run_notebook.py` y notebook de entrenamiento. |
| Estudiante 4 | Métricas en `src/evaluation.py`, registros del piloto y análisis de errores. |

## 2. Handoffs: una frase y continúa el siguiente

| Transición | Frase sugerida |
|---|---|
| Estudiante 2 → Alan | “Con estos símbolos y datos definidos, Alan mostrará cómo procesamos el dibujo del usuario.” |
| Alan → estudiante 3 | “Esta imagen preparada es la entrada de la CNN; ahora veremos cómo aprende a distinguir las seis clases.” |
| Estudiante 3 → estudiante 4 | “Elegimos el modelo con validación; ahora evaluaremos su resultado en las imágenes de prueba y en dibujos nuevos.” |

Estos enlaces no requieren otra diapositiva ni un resumen del bloque anterior. Algún dato básico puede reaparecer como referencia; la explicación completa tiene un solo responsable.

## 3. Evidencia que debe aparecer

- **Estudiante 2:** 800 imágenes por clase del subconjunto HASY, 4800 en total. No son 4800 dibujos recopilados por el equipo. Documenta la respuesta del profesor sobre el requisito de recolección.
- **Alan:** comparación visual real, entrada `32 × 32 × 1`, procesamiento compartido y consulta que devuelve tres comandos sin guardar un dato de entrenamiento.
- **Estudiante 3:** partición HASY de 3360/720/720, semilla 42, 12 épocas ejecutadas y selección de época 7 por menor pérdida de validación. Los IDs de HASY no garantizan identidad de persona; si cambia el origen, documenta el protocolo correspondiente.
- **Estudiante 4:** HASY top-1 98,61 %, top-3 99,86 % y F1 macro 0,9861. Piloto de Alan: 19/30 top-1 y 29/30 top-3. Aclara que son conjuntos y procedimientos distintos; 30 consultas agregadas no permiten una nueva matriz de confusión ni F1 del mouse.
- **Rechazo OOD, si ya se ejecutó:** usa `data/rejection-evaluation-*/metrics.json` y sus gráficos. Expón cantidades, participantes, método y separación validación/prueba; no ajustes con la prueba ni presentes una métrica hasta que exista el artefacto real. Consulta [el protocolo de rechazo](RECHAZO_DESCONOCIDOS.md).

Esos son los resultados ya disponibles, no promesas de resultados futuros. Cualquier experimento nuevo debe identificar su propia versión. La presentación no debe dar a entender que la herramienta reconoce símbolos fuera del catálogo o que las puntuaciones son certeza calibrada.

## 4. Revisión y ensayo

Revisión en cadena: estudiante 2 revisa Alan; Alan revisa estudiante 3; estudiante 3 revisa estudiante 4; estudiante 4 revisa estudiante 2. Se revisan claridad, correspondencia con evidencia y ausencia de repeticiones. Las comprobaciones técnicas pueden hacerse además entre responsables relacionados.

1. Cada persona explica su bloque con cronómetro y lo termina dentro de `T/4`.
2. Hacen un ensayo completo con las transiciones y la demo.
3. Quitan explicaciones repetidas y verifican que cifras y versiones coincidan.
4. Todos practican preguntas de cualquier bloque; saber responder fuera de tu tema no implica repetirlo en el discurso.
5. Comprueban que los cuatro intervienen: la consigna asigna 0 individual a quien no expone.

El propietario de cada bloque responde primero las preguntas específicas de su tema; otros complementan brevemente si hace falta.

## 5. Cierre de las diapositivas

Cada bloque debe dejar una idea:

- Estudiante 2: “El problema es encontrar el comando de un símbolo aislado y conocido por el catálogo.”
- Alan: “Un procesamiento común hace compatible la imagen recolectada con la imagen consultada.”
- Estudiante 3: “El modelo se selecciona usando validación, antes de interpretar el resultado final de prueba.”
- Estudiante 4: “Tres sugerencias ayudaron en este piloto; necesitamos evaluar más dibujantes para sostener conclusiones más amplias.”

Solo el estudiante 4 desarrolla las conclusiones globales. El equipo completo las revisa antes del ensayo.

Documentos de apoyo: [instalación](INSTALACION.md), [contribución de Alan](CONTRIBUCION_ALAN.md) y [plan de entrega con rúbrica](PLAN_EQUIPO_PC2.md).
