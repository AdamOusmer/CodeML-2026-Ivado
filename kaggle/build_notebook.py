import argparse
from pathlib import Path

import nbformat

PACKAGE = Path(__file__).resolve().parent
OUT_DIR = "/kaggle/working/tabm_residual"

INTRO = """# TabM residual on the historical committee labels

Trains the bounded TabM residual on the historical data only and exports `residuals.csv` and `manifest.json`.
Inputs: the two supplied CSVs (`donnees_demandes.csv`, `candidats_evaluation.csv`) attached as a Kaggle dataset.
Enable a free GPU accelerator before running."""

INSTALL = "!pip install -q tabm"

RUN = f"""import subprocess, sys
command = [sys.executable, '-u', 'train.py', '--out-dir', '{OUT_DIR}']
with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
    for line in process.stdout:
        print(line, end='')
if process.returncode:
    raise RuntimeError(f'Training exited with code {{process.returncode}}')"""

COLLECT = f"""import shutil
print(shutil.make_archive('/kaggle/working/tabm_residual', 'zip', '{OUT_DIR}'))"""


def build(target: Path) -> None:
    train_source = (PACKAGE / "train.py").read_text()
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(INTRO),
        nbformat.v4.new_code_cell(INSTALL),
        nbformat.v4.new_code_cell(f"from pathlib import Path\nPath('train.py').write_text({train_source!r})"),
        nbformat.v4.new_code_cell(RUN),
        nbformat.v4.new_code_cell(COLLECT),
    ]
    nbformat.validate(notebook)
    nbformat.write(notebook, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Kaggle notebook that runs train.py.")
    parser.add_argument("--out", type=Path, default=PACKAGE / "train_tabm_residual.ipynb")
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
