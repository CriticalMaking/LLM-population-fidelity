# Paper report set

Section: `all`  
Questions: d_happy, d_polpos, d_religiousp, d_trust  
Series: NTP-GPT-4T, NTP-Llama-3-70B, NTP-Mixtral-8x7B, FA-GPT-3, FA-Llama-3-70B, FA-Mixtral-8x7B

## Tables

- `outputs/reports/table-2-average-nemd.csv` / `outputs/reports/tex/table-2-average-nemd.tex`
- `outputs/reports/table-3-quality.csv` / `outputs/reports/tex/table-3-quality.tex`
- `outputs/reports/table-4-pairwise.csv` / `outputs/reports/tex/table-4-pairwise.tex`
- `outputs/reports/table-5-regression-fit.csv` / `outputs/reports/tex/table-5-regression-fit.tex`
- `outputs/reports/table-6-f-tests.csv` / `outputs/reports/tex/table-6-f-tests.tex`
- `outputs/reports/response-distributions.csv` / `outputs/reports/tex/response-distributions.tex`
- `outputs/reports/subpopulation-distances.csv` / `outputs/reports/tex/subpopulation-distances.tex`
- `outputs/reports/regression-coefficients.csv` / `outputs/reports/tex/regression-coefficients.tex`
- `outputs/reports/table-S1-sample-size.csv` / `outputs/reports/tex/table-S1-sample-size.tex`
- `outputs/reports/table-S2-predictors.csv` / `outputs/reports/tex/table-S2-predictors.tex`
- `outputs/reports/table-S4-ntp-compliance.csv` / `outputs/reports/tex/table-S4-ntp-compliance.tex`
- `outputs/reports/table-S6-prompting-strategies.csv` / `outputs/reports/tex/table-S6-prompting-strategies.tex`
- `outputs/reports/table-S7-quantization.csv` / `outputs/reports/tex/table-S7-quantization.tex`
- `outputs/reports/table-S8-discriminator.csv` / `outputs/reports/tex/table-S8-discriminator.tex`
- `outputs/reports/table-S9-backtranslation.csv` / `outputs/reports/tex/table-S9-backtranslation.tex`
- `outputs/reports/temperature-distances.csv` / `outputs/reports/tex/temperature-distances.tex`
- `outputs/reports/table-S16-coefficient-comparison.csv` / `outputs/reports/tex/table-S16-coefficient-comparison.tex`

## Figures

- `figures/reports/Figure-2-nEMD-density.pdf`
- `figures/reports/Figure-3-social-FA-Llama-3-70B.pdf`
- `figures/reports/Figure-3-social-NTP-GPT-4T.pdf`
- `figures/reports/Figure-3-social-NTP-Llama-3-70B.pdf`
- `figures/reports/Figure-3-social-NTP-Mixtral-8x7B.pdf`
- `figures/reports/Figure-4-MDS-NTP.pdf`
- `figures/reports/Figure-6-FA-Llama-3-70B.pdf`
- `figures/reports/Figure-6-NTP-GPT-4T.pdf`
- `figures/reports/Figure-6-NTP-Llama-3-70B.pdf`
- `figures/reports/Figure-6-NTP-Mixtral-8x7B.pdf`
- `figures/reports/Figure-S1-outcome-distributions.pdf`
- `figures/reports/Figure-S10-MDS-compare-NTP-FA.pdf`
- `figures/reports/Figure-S12-prompting-strategy-MDS.pdf`
- `figures/reports/Figure-S13-temperature-nEMD-density.pdf`
- `figures/reports/Figure-S14-temperature-MDS.pdf`
- `figures/reports/Figure-S15-correlations.pdf`
- `figures/reports/Figure-S16-cofdif-boot-FA.pdf`
- `figures/reports/Figure-S16-cofdif-boot-NTP.pdf`
- `figures/reports/Figure-S17-backnn-country-FA-GPT-3.pdf`
- `figures/reports/Figure-S17-backnn-country-FA-Llama-3-70B.pdf`
- `figures/reports/Figure-S17-backnn-country-FA-Mixtral-8x7B.pdf`
- `figures/reports/Figure-S17-backnn-country-NTP-GPT-4T.pdf`
- `figures/reports/Figure-S17-backnn-country-NTP-Llama-3-70B.pdf`
- `figures/reports/Figure-S17-backnn-country-NTP-Mixtral-8x7B.pdf`
- `figures/reports/Figure-S18-backnn-decade-FA-GPT-3.pdf`
- `figures/reports/Figure-S18-backnn-decade-FA-Llama-3-70B.pdf`
- `figures/reports/Figure-S18-backnn-decade-FA-Mixtral-8x7B.pdf`
- `figures/reports/Figure-S18-backnn-decade-NTP-GPT-4T.pdf`
- `figures/reports/Figure-S18-backnn-decade-NTP-Llama-3-70B.pdf`
- `figures/reports/Figure-S18-backnn-decade-NTP-Mixtral-8x7B.pdf`
- `figures/reports/Figure-S2-MDS-NTP.pdf`
- `figures/reports/Figure-S4-MDS-FA.pdf`
- `figures/reports/Figure-S9-R2-soc-all.pdf`

## Not reproduced

- **Figure 1** — Schematic in the paper; no generator in the replication package.
- **Figure 5** — Schematic in the paper; no generator in the replication package.
- **Table 1** — Descriptive listing in the paper; not produced by 6-results.R.
- **Figure S3** — No generator in 6-results.R; the MDS set jumps S2 to S4.
- **Figure S11** — No generator in 6-results.R.
- **Table S3** — No generator in 6-results.R; the table set jumps S2 to S4.

## Notes

- **Figure S1** — 6-results.R:1238 prints this to a device and never saves it; written here so the appendix set is complete on disk.
- **Figure S12** — 6-results.R:1438 writes this plate as Figure-S10-prompting-strategy-MDS.png into the working directory. Emitted here under its correct S12 name.
- **Table S5** — Identical in construction to Table 3; both are emitted from one computation.
- **Figures S5-S8** — 6-results.R:1287 redraws Figure 3 for the remaining series rather than generating separate plates. Emitted here under the Figure-3-social-<series> names.
- **Table S8** — sklearn RandomForestClassifier(oob_score=True) stands in for R ranger.
- **Figure S16** — statsmodels MNLogit stands in for R nnet::multinom.
