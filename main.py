"""Recolector local de símbolos científicos. La CNN se añadirá en la siguiente etapa."""

import base64
import binascii
import io
import json
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, UnidentifiedImageError

from src.catalog import PROJECT_ROOT, load_catalog
from src.dataset import DatasetStore, DuplicateDrawingError
from src.preprocessing import EmptyDrawingError, preprocess_image


def create_app(data_root=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    catalog = load_catalog()
    store = DatasetStore(
        Path(data_root) if data_root is not None else PROJECT_ROOT / "data", catalog
    )
    app.extensions["dataset_store"] = store

    @app.get("/")
    def index():
        current_counts = store.counts()
        try:
            metadata = json.loads(
                (store.processed_root / "metadata.json").read_text(encoding="utf-8")
            )
            export_ready = (
                (store.processed_root / "dataset.zip").exists()
                and metadata["class_counts"] == current_counts
            )
        except (OSError, ValueError, KeyError):
            export_ready = False
        return render_template(
            "index.html", catalog=catalog, counts=current_counts, export_ready=export_ready,
        )

    @app.get("/api/counts")
    def counts():
        return jsonify(counts=store.counts())

    @app.post("/upload")
    def upload():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Envía los datos del dibujo en formato JSON."), 400
        class_id = payload.get("class_id")
        if not isinstance(class_id, str) or class_id not in store.symbols:
            return jsonify(error="Selecciona una categoría del catálogo."), 400
        participant_id = payload.get("participant_id")
        if not isinstance(participant_id, str):
            return jsonify(error="Ingresa un código de participante, por ejemplo P01."), 400
        participant_id = participant_id.strip().upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{1,31}", participant_id):
            return jsonify(error="Usa un código de 2 a 32 letras, números o guiones."), 400
        input_method = payload.get("input_method", "unknown")
        if input_method not in ("mouse", "touch", "pen", "unknown"):
            return jsonify(error="El método de entrada no es válido."), 400
        data_url = payload.get("image")
        prefix = "data:image/png;base64,"
        if not isinstance(data_url, str) or not data_url.startswith(prefix):
            return jsonify(error="El dibujo debe enviarse como una imagen PNG."), 400
        try:
            decoded = base64.b64decode(data_url[len(prefix):], validate=True)
            if len(decoded) > 1024 * 1024:
                return jsonify(error="La imagen supera el tamaño permitido."), 400
            with Image.open(io.BytesIO(decoded)) as image:
                if image.format != "PNG" or max(image.size) > 1024:
                    return jsonify(error="Usa un PNG de hasta 1024 píxeles por lado."), 400
                image.load()
                preprocess_image(image, size=catalog["image_size"])
                sample_image = image.copy()
        except (binascii.Error, UnidentifiedImageError, Image.DecompressionBombError, OSError):
            return jsonify(error="No se pudo leer el PNG. Vuelve a dibujar el símbolo."), 400
        except EmptyDrawingError as error:
            return jsonify(error=str(error)), 400
        try:
            record = store.save(sample_image, class_id, participant_id, input_method)
        except DuplicateDrawingError as error:
            return jsonify(error=str(error)), 409
        except OSError:
            app.logger.exception("No se pudo guardar el dibujo")
            return jsonify(error="No se pudo guardar el dibujo. Revisa el espacio en disco."), 500
        return jsonify(
            sample_id=record["sample_id"], participant_id=participant_id, counts=store.counts()
        ), 201

    @app.post("/prepare")
    def prepare():
        try:
            metadata = store.export()
        except (ValueError, OSError, KeyError) as error:
            return jsonify(error=f"No se pudieron preparar los datos: {error}"), 400
        return jsonify(metadata=metadata, download_url="/dataset.zip")

    @app.get("/dataset.zip")
    def download_dataset():
        archive = store.processed_root / "dataset.zip"
        if not archive.exists():
            return jsonify(error="Primero prepara los datos desde la aplicación."), 404
        return send_file(archive, as_attachment=True, download_name="dataset_simbolos.zip")

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error="La imagen supera el tamaño permitido."), 413

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
