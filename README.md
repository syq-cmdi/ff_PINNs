# ff_PINNs code

This public repository contains source code for scale-consistent
Kuramoto–Sivashinsky (KS) solver and physics-informed neural-network
experiments.

It is intentionally **code-only**. It does not distribute any unpublished
manuscript, Supporting Information, submission PDF/DOCX, bibliography,
submission metadata, research data, computed result, figure, checkpoint, or
trained model weight.

## Installation

Create an isolated Python environment and install the recorded scientific
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Verification

Check the code-only allow-list and compile every released Python module:

```bash
PYTHON=python3 ./reproduce.sh verify
```

Run the deterministic reference-solver workflow in a new local output
directory:

```bash
PYTHON=python3 ./reproduce.sh reference
```

Run the complete computational workflow, including PINN training:

```bash
PYTHON=python3 PINN_DEVICE=mps ./reproduce.sh full
```

Generated inputs and outputs are written beneath the ignored `reproduced/`
directory and are never added to the repository by the workflow.

## Repository map

- `code/`: numerical solvers, PINN training, inverse identification, data
  retrievers, audits, and code-only manifest checks.
- `reproduce.sh`: deterministic reference and full-workflow entry points.
- `requirements.txt`: recorded Python dependencies.
- `RELEASE_MANIFEST.json`: hashes for the released code and support files.
- `THIRD_PARTY_NOTICES.md`: upstream source and license information.

## External inputs

The scripts retrieve external inputs from their authoritative sources and
verify pinned identifiers or checksums. No source dataset is committed here.

- KS archive: `maziarraissi/PINNs`, commit
  `13ce051750dbc2c9afa3b105b09f373c524ef272`.
- FETRIG Version 1.0: <https://doi.org/10.18419/DARUS-4998>, CC BY 4.0.

## License

Project source code is released under the MIT License. External inputs retain
their original licenses; see `THIRD_PARTY_NOTICES.md`.
