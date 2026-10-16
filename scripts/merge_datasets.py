
import csv
import hashlib
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEST = DATA / "raw"

SOURCES = {
    "P01": DATA / "importaciones/P01_dataset_762/raw",
    "P03": DATA / "importaciones/P03_dataset/raw",
    "P04": DATA / "importaciones/raw_maurizio_p04/raw",
}

CLASSES = {"alpha", "equiv", "in", "infty", "partial", "xi"}

FIELDS = [
    "sample_id", "path", "class_id", "class_index",
    "participant_id", "input_method", "created_at", "image_sha256"
]


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    csv_dest = DEST / "samples.csv"
    existing = read_csv(csv_dest)

    hashes = set()
    ids = set()
    counts = Counter()
    pending = []
    skipped = Counter()

    for row in existing:
        file = DEST / row["path"]

        if not file.is_file():
            raise RuntimeError(f"Archivo P02 faltante: {file}")

        hashes.add(checksum(file))
        ids.add(row["sample_id"])
        counts[row["class_id"]] += 1

    for participant, source in SOURCES.items():
        rows = read_csv(source / "samples.csv")

        for row in rows:
            cls = row["class_id"]

            if cls not in CLASSES:
                skipped["clase_invalida"] += 1
                continue

            if row["participant_id"].upper() != participant:
                skipped["participante_incorrecto"] += 1
                continue

            source_file = source / row["path"]

            if not source_file.is_file():
                skipped["archivo_faltante"] += 1
                continue

            digest = checksum(source_file)

            if digest in hashes:
                skipped["duplicado"] += 1
                continue

            new_id = f"{participant}_{row['sample_id']}"

            if new_id in ids:
                raise RuntimeError(f"ID repetido: {new_id}")

            relative = f"{cls}/{new_id}_{digest[:12]}.png"
            destination = DEST / relative

            if destination.exists():
                raise RuntimeError(f"Archivo existente: {destination}")

            new_row = {
                **row,
                "sample_id": new_id,
                "path": relative,
                "participant_id": participant,
                "image_sha256": digest,
            }

            pending.append((source_file, destination, new_row))
            hashes.add(digest)
            ids.add(new_id)
            counts[cls] += 1

    print("\n=== RESUMEN DE FUSION ===")
    print("Registros P02 existentes:", len(existing))
    print("Imagenes nuevas:", len(pending))
    print("Omitidas:", dict(skipped))

    print("\nTotales previstos por simbolo:")
    for cls in sorted(CLASSES):
        print(f"{cls}: {counts[cls]}")

    print("\nTotal previsto:", len(existing) + len(pending))

    confirm = input("\nEscribe FUSIONAR para continuar: ")

    if confirm != "FUSIONAR":
        print("Operacion cancelada.")
        return

    backup = DEST / "samples_premerge_backup.csv"

    if backup.exists():
        raise RuntimeError(
            "Ya existe samples_premerge_backup.csv. "
            "Revisa el respaldo antes de repetir."
        )

    shutil.copy2(csv_dest, backup)

    for source_file, destination, row in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)

    all_rows = existing + [row for _, _, row in pending]

    temp_csv = DEST / "samples_merged.tmp"

    with temp_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    temp_csv.replace(csv_dest)

    print("\nFUSION COMPLETADA")
    print("Total de registros:", len(all_rows))


if __name__ == "__main__":
    main()
