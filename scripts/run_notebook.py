"""Ejecuta las celdas con IPython y conserva texto, tablas y figuras en el notebook."""

import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.catalog import PROJECT_ROOT


def execute_notebook(notebook, directory):
    import nbformat
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    from IPython.utils.capture import capture_output

    shell = TerminalInteractiveShell.instance(simple_prompt=True)
    shell.display_formatter.active_types = [
        "text/plain", "text/html", "text/markdown", "text/latex", "image/png", "application/json"
    ]
    previous_directory = Path.cwd()
    try:
        os.chdir(directory)
        with capture_output():
            shell.run_line_magic("matplotlib", "inline")
        execution_count = 0
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            execution_count += 1
            print(f"Ejecutando celda {index + 1}/{len(notebook.cells)}...", flush=True)
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
            if result.error_before_exec or result.error_in_exec:
                print(captured.stdout, end="")
                print(captured.stderr, end="")
                raise RuntimeError(f"Falló la celda {index + 1}; el notebook anterior se conserva.") from (
                    result.error_before_exec or result.error_in_exec
                )
            cell.execution_count = execution_count
            cell.outputs = []
            for name, value in (("stdout", captured.stdout), ("stderr", captured.stderr)):
                if value:
                    cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=value))
            for output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output(
                    "display_data", data=output.data, metadata=output.metadata
                ))
            print(captured.stdout, end="", flush=True)
        notebook.metadata["execution"] = {
            "runner": "IPython in the current Python process",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        nbformat.validate(notebook)
    finally:
        os.chdir(previous_directory)
        TerminalInteractiveShell.clear_instance()
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=PROJECT_ROOT / "notebooks" / "01_entrenamiento_simbolos.ipynb")
    args = parser.parse_args()
    try:
        import nbformat
        notebook = nbformat.read(args.notebook, as_version=4)
        print("Ejecutando notebook. La primera ejecución descarga datos y entrena en CPU...", flush=True)
        execute_notebook(notebook, PROJECT_ROOT)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".ipynb", dir=args.notebook.parent, delete=False) as output:
            temporary = Path(output.name)
            nbformat.write(notebook, output)
        os.replace(temporary, args.notebook)
        print(f"Notebook ejecutado y guardado: {args.notebook}")
        return 0
    except ImportError:
        print("Instala requirements-notebook.txt en el entorno virtual.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
