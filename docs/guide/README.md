# Setup Guide

## Python Environment

The notebooks in [`notebooks/`](../../notebooks/) use a local virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Raw data is expected under `data/raw/` (already present in this repo checkout, and gitignored so it isn't re-committed).

## LaTeX Report

The report source is [`latex/main.tex`](../../latex/main.tex), split into per-section files under `latex/content/`. It must be compiled with **XeLaTeX**, not plain `pdflatex` — the cover page includes Vietnamese names with diacritics that only render correctly through XeLaTeX's `fontspec`-based font handling.

### 1. Install a TeX distribution

On macOS:

```bash
brew install --cask basictex
```

BasicTeX is a small (~100 MB) TeX Live subset and is enough to build this report. If you'd rather not deal with occasional missing packages, install the full [MacTeX](https://www.tug.org/mactex/) instead — it includes nearly everything, at the cost of a much larger download.

After installing, open a **new** terminal (or restart VS Code) so the TeX binaries are picked up on your `PATH`. If they still aren't found, add them manually:

```bash
export PATH="/Library/TeX/texbin:$PATH"
```

Then install the packages this report depends on, listed in [`latex/requirements.txt`](../../latex/requirements.txt):

```bash
sudo tlmgr install $(cat latex/requirements.txt)
```

### 2. Compile from the command line

```bash
cd latex
xelatex main.tex
xelatex main.tex   # run a second time so the table of contents is correct
```

This produces `latex/main.pdf`. If XeLaTeX reports a missing `.sty` file (BasicTeX ships a minimal package set), install it with:

```bash
sudo tlmgr install <package-name>
```

### 3. Compiling from VS Code (optional)

If you use the **LaTeX Workshop** extension, it defaults to building with `latexmk`, which BasicTeX doesn't include. Create `.vscode/settings.json` (gitignored, since the binary path below is machine-specific) with:

```json
{
    "latex-workshop.latex.tools": [
        {
            "name": "xelatex",
            "command": "/Library/TeX/texbin/xelatex",
            "args": [
                "-interaction=nonstopmode",
                "-synctex=1",
                "-file-line-error",
                "%DOC%"
            ]
        }
    ],
    "latex-workshop.latex.recipes": [
        {
            "name": "xelatex x2",
            "tools": ["xelatex", "xelatex"]
        }
    ],
    "latex-workshop.latex.recipe.default": "xelatex x2",
    "latex-workshop.latex.outDir": "%DIR%"
}
```

The `command` path above is where `brew install --cask basictex` puts binaries on macOS. On Linux/Windows, replace it with wherever your TeX Live install put `xelatex` (or just `"xelatex"` if it's already on your `PATH`).

To build: open `latex/main.tex`, click into that editor tab so it's the active document, then run the build command (▶ button in the top-right, or `Cmd+Option+B`).

### Troubleshooting

- **"Cannot find LaTeX root file"** — the active editor tab wasn't `main.tex` when you triggered the build. Click into `main.tex` first.
- **"spawn latexmk ENOENT"** — `latexmk` isn't installed and isn't needed; create the `.vscode/settings.json` from step 3 above so LaTeX Workshop builds with `xelatex` instead. After creating/editing it, reload the VS Code window (`Cmd+Shift+P` → "Developer: Reload Window") so the setting takes effect. To install `latexmk` anyway: `sudo tlmgr install latexmk`.
- Build artifacts (`*.aux`, `*.toc`, `*.pdf`, etc.) are gitignored, so there's no need to clean them up before committing.
