# Third-party notices and data attribution

## Pinned Kuramoto–Sivashinsky archive

The file used by the recalculation originates from the `maziarraissi/PINNs` repository:

- Source: `main/Data/KS.mat`
- Commit: `13ce051750dbc2c9afa3b105b09f373c524ef272`
- Repository: <https://github.com/maziarraissi/PINNs>
- License: MIT
- Expected SHA-256: `a47286bb83b79d1ac0d46fe818693a627df411556166afe454386d88ad5d1ac7`

The public package provides a downloader and checksum verification. The upstream MIT notice must be retained if the data file is redistributed.

## FETRIG experimental data

The experimental source is:

> M. Wirth, J. Hagedorn, B. Weigand, and S. Kabel, “Replication Data for FETRIG initial measurement campaign of the falling film evaporation test rig (FETRIG) at ITLR,” DaRUS, Version 1.0 (2025), <https://doi.org/10.18419/DARUS-4998>.

- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- License URL: <https://creativecommons.org/licenses/by/4.0/>
- Modification/selection: the workflow selects 18 zero-gas-flow Keyence records at three archive-labelled stations and six liquid-Reynolds-number labels; alarm values are excluded from dimensional statistics, and short alarm gaps are interpolated only for spectra.

Raw FETRIG traces are not committed in this filtered release. `code/fetch_fetrig_subset.py` retrieves the exact subset and writes a provenance/checksum manifest.

## Excluded material

No upstream dataset, publisher PDF, book, manuscript, submission document,
computed result, neural checkpoint, prediction, table, or figure is committed
to this code-only repository. Users retrieve external inputs directly from the
authoritative sources above and retain their original licenses and attribution
requirements.
