# SUBMISSION CHECKLIST — IEEE TIM (target; **user-selected 2026-10-11: 保底档 TIM**)

Status: draft compiled (`paper/main.pdf`, 9 pages, 0 undefined refs) · citations verified 2026-10-11
Date: 2026-10-11

## Citation verification & downloads (2026-10-11)
- [x] 13/13 arXiv entries verified via arXiv API against title metadata
- [x] **One critical error caught and fixed**: `lei2018` had wrong arXiv ID (1806.05585 → correct **1604.04173**, Lei/G'Sell/Rinaldo/Tibshirani/Wasserman); bib, wiki page, literature list, and paper plan updated; PDF re-downloaded
- [x] 2/2 DOI entries verified via CSL-JSON (XJTU-SY TR 2020; EVT/Siffer KDD 2017)
- [x] 13 PDFs downloaded to `literature/pdfs/`; paywalled items (2) listed in `paper/DOWNLOAD_TODO.md` for subscription download
- [x] Dataset page snapshots: `literature/pages/` (PHM16, Paderborn; PVDAQ via S3 listing URL)

## Author information
- [ ] Fill `paper/AUTHOR_INFO_TEMPLATE.md` (names, affiliations, ORCID, corresponding author, funding, data-availability, conflict statement)

## Format
- [x] IEEEtran journal class (`\documentclass[journal]{IEEEtran}`), double column
- [x] Abstract (<= 250 words) + IEEEkeywords present
- [x] Page count: 9 (regular-paper range; confirm current TIM overlength policy on the submission portal)
- [x] Figures: vector PDF, embedded; referenced and discussed (Figs. 1-4)
- [x] Tables: booktabs, referenced (Tables I-VIII)
- [x] Bibliography: IEEEtran.bst; 18 entries; all cited keys resolved
- [x] Author block + affiliations — signed version (single-blind per IEEE TIM); ORCID optional, pending user
- [ ] Funding / acknowledgments / data-availability statement — placeholders to fill
- [ ] Venue template overlay: confirm TIM-specific style file requirements on the portal (IEEEtran journal is standard)

## Scientific integrity
- [x] Below-chance folds explicitly excluded from detection claims (stated in abstract, Table II footnote, Sec. V-E)
- [x] Synthetic-only fold flagged in every table/figure caption (F3)
- [x] Negative results reported (divergence/Mondrian/invariance/TTA/gating; D-bar-inflation null)
- [x] Protocols for pseudo-labels documented (PHM16 q10; PV injection; run-to-failure last-20%)
- [x] Provenance-checked datasets (PHM16 cross-mirror verification; CARE byte-verified; mirrors documented)
- [x] 2 dataset citations carry explicit camera-ready verification TODOs (PHM16 URL resolved; Paderborn original metadata)

## Reproducibility package (to prepare at submission)
- [ ] Code release: `experiments/code/ssi-ad-ablation-spectral-relation-band/` (runners, configs, figure scripts)
- [ ] Result JSONs: `results/scc-main/`, `results/scc-main-traces/`, `results/scc-alt-scores/`
- [ ] Dataset provenance READMEs (PHM16/CARE/PVDAQ/CWRU/XJTU-SY/Paderborn/MFPT)
- [ ] Choose repository (GitHub) + license; de-identify before public release

## Cover letter
- [x] Template ready: `paper/COVER_LETTER_TEMPLATE.md` (fill author info)

## Remaining user actions
1. Fill author names/affiliations/ORCID; decide anonymity mode per portal.
2. Confirm funding/acknowledgment text; data-availability wording.
3. Confirm suggested-reviewer list (optional).
4. Verify Paderborn dataset citation with KAt before camera-ready.
