# Rechazo de dibujos fuera del catálogo

La CNN clasifica seis clases cerradas. Softmax siempre distribuye una puntuación
entre ellas; ni un porcentaje alto ni un margen grande prueban que el dibujo sea
una de esas seis clases. El modo de consulta presenta las puntuaciones como
salidas del modelo, no como certeza, y oculta alternativas cuando el rechazo se
activa.

## Recolección separada

En modo **Consultar LaTeX**, dibuja un símbolo fuera del catálogo y abre
**Registrar un dibujo desconocido para evaluación**. Etiqueta la referencia
como integral, raíz cuadrada, más, pi, garabato aleatorio u otro; usa el código
estable de participante `P01`–`P04`. La muestra se escribe en `data/unknown/`,
con PNG y `samples.csv`, sin añadirla a `data/raw/`, `data/processed/` ni a las
seis clases conocidas. Esa carpeta está excluida de Git.

Las etiquetas son una referencia humana: registra únicamente un dibujo cuya
identidad conozcas. Un garabato aleatorio no representa por sí solo todos los
dibujos desconocidos. No cambies el código de una persona para repartir muestras
entre particiones. Los directorios históricos `I/`, `N/` y `U/` no se incorporan
automáticamente; primero verifica sus etiquetas y procedencia.

## Comparación reproducible

El script requiere el mismo dataset exacto con el que se entrenó el modelo,
incluido en `model_info.json`, y ejemplos de integral, raíz, más, pi y garabato
aleatorio para participantes de validación **y** de prueba reservados en ese
modelo. No usa el conjunto de prueba conocido ni el desconocido de prueba para
calibrar el umbral.

Desde la raíz del repositorio:

```powershell
python -m scripts.evaluate_rejection `
  --data data/processed `
  --unknown-data data/unknown `
  --model models/symbols `
  --output data/rejection-evaluation-run-01 `
  --write-policy
```

En Bash, reemplaza los acentos graves de continuación por `\`. Escoge un nombre
de salida nuevo: el script se niega a sobrescribir resultados. `--write-policy`
es opcional; si se especifica, crea `rejection_calibration.json` al lado del
modelo con apertura exclusiva **solo si** la estrategia de energía supera al
Softmax actual en accuracy balanceada macro de validación. No reemplaza
`model.pt`, y se niega a reemplazar una política anterior. La política está
vinculada al SHA-256 exacto del checkpoint; si el modelo cambia, la aplicación
no la usa.

La estrategia de referencia es el rechazo preliminar existente: Softmax top-1
≥ 0.80 y margen top-1/top-2 ≥ 0.20. La alternativa calcula energía
`-logsumexp(logits)` de los logits de la CNN y selecciona un umbral maximizando
accuracy balanceada en **validación** conocida y desconocida. El informe
compara ambas en validación y en la prueba intacta; informa accuracy/F1 de
clasificación conocida, accuracy entre conocidos aceptados, tasa de rechazo
incorrecto de conocidos, aceptación incorrecta y rechazo correcto de
desconocidos, resultados por clase/tipo, y matrices de confusión.

El umbral de energía maximiza accuracy balanceada macro-promediada entre las
clases conocidas y los tipos desconocidos presentes en validación, para que una
categoría numerosa no domine por sí sola la calibración.

El directorio de salida contiene `metrics.json`, `predictions.csv`,
`rejection_rates.png`, `energy_confusion_matrix.png` y
`unknown_confusion_matrix.png`. Incluye esos artefactos
reales en el notebook de exposición junto con los hashes, participantes,
cantidades por tipo y configuración de la ejecución. Si no hay muestras
desconocidas de validación y prueba, si las particiones no están representadas,
si el modelo no existe o si su dataset no coincide, la evaluación falla con un
mensaje explícito; no se emiten métricas ficticias ni una política calibrada.

El método de energía es una heurística de rechazo, no una garantía de detección
perfecta. Su umbral solo refleja las muestras reunidas y debe revisarse cuando
cambie el origen, el estilo de dibujo o el modelo. La imagen procesada se puede
mostrar opcionalmente en la interfaz; se conserva el procesamiento de 32 × 32
del entrenamiento y no se requiere reentrenar para activar la vista previa.
