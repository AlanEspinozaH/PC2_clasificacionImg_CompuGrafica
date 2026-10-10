"""Entrena una CNN con participantes separados y guarda el experimento completo."""

import argparse

from src.catalog import PROJECT_ROOT
from src.training_data import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--output", default=PROJECT_ROOT / "models" / "symbols")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pilot", action="store_true", help="Permite una colección menor a la meta; evaluación preliminar.")
    parser.add_argument("--validation-participants", nargs="+")
    parser.add_argument("--test-participants", nargs="+")
    args = parser.parse_args()
    try:
        dataset = load_dataset(args.data)
        from src.training import train_model
        info, metrics = train_model(
            dataset, args.output, epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience, seed=args.seed, pilot=args.pilot, threads=args.threads,
            validation_participants=args.validation_participants, test_participants=args.test_participants,
        )
        print(f"\nModelo guardado en {args.output}; mejor época: {info['best_epoch']}")
        print(f"Prueba: top-1={metrics['top_1_accuracy']:.3f}, top-3={metrics['top_3_accuracy']:.3f}, "
              f"F1 macro={metrics['macro_f1']:.3f}")
        if args.pilot:
            print("Ensayo preliminar: estas métricas no acreditan la calidad del modelo final.")
        return 0
    except ImportError:
        print("Instala requirements-cpu.txt y requirements-ml.txt en el entorno virtual.")
        return 1
    except (ValueError, OSError, KeyError) as error:
        print(f"No se pudo entrenar: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
