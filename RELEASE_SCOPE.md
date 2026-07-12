# Public repository scope

This repository uses a strict code-only allow-list.

## Included

- Scientific Python source code.
- The shell entry point used to run verification and calculations.
- Dependency metadata, license, software citation metadata, third-party
  notices, and a SHA-256 manifest of the released files.

## Excluded

- Manuscripts, Supporting Information, bibliographies, submission metadata,
  cover letters, and all journal-submission files.
- PDF and DOCX documents of any kind.
- Raw or processed research data.
- Computed results, tables, plots, and figures.
- Checkpoints, predictions, trained model weights, and other binary artifacts.
- Publisher content, books, literature PDFs, local QA files, caches, and
  exploratory runs.

Generated files belong in the ignored `reproduced/` directory. The manifest
builder refuses any public-tree path or extension that violates this policy.
