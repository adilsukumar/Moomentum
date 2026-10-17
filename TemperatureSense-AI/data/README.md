# Canine temperature research data

## Included dataset

- Original workbook: download `S1 Dataset.xlsx` from the [Mendeley source](https://doi.org/10.17632/ykfvgt49gg.1) into `raw/neander_labrador_temperature/` before rebuilding
- Convenient, reproducible export: [`derived/labrador_temperature_phases.csv`](derived/labrador_temperature_phases.csv)
- Rebuild export: `python scripts/export_labrador_reference.py` after installing
  the optional `data` dependencies (`pip install -e ".[data]"`).

**Attribution:** Neander C, Baker J, Kelsey K, Feugang J, Perry E (2021),
*Comparison of black vs yellow coat color on rectal and gastrointestinal
temperature in Labrador Retrievers—dataset*, Mendeley Data, version 1.
DOI: [10.17632/ykfvgt49gg.1](https://doi.org/10.17632/ykfvgt49gg.1).
The original dataset is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The upstream workbook is not redistributed here. Its verified source SHA-256 is
`89fcf869aeb6d597a6885af6c152557069cf6b483e4cd1fd6447cca35884ad17`.

The workbook has `Dog data` and `Enviromental` (source spelling) sheets. It
records rectal, gastrointestinal, infrared eye and abdomen temperatures across
three phases: indoor resting baseline, walking in sunlight, and indoor cooling.
It also records respiratory rate and environmental conditions. The published
description says 16 Labradors; one workbook row (`Yellow 8`) explicitly says
`REMOVED PRIOR TO STUDY`, so **15 dogs have usable measurements**.

The derived CSV is an **adaptation**: it converts the wide `Dog data` sheet to
one row per dog and phase (45 rows), trims dog IDs, normalizes column names,
and omits the excluded dog and fields not needed for temperature comparisons.
Blank source cells remain blank (e.g. one missing baseline eye measurement).
`baseline` means **resting, not confirmed sleeping**. Temperature columns are
degrees Celsius; `respiration_per_min` follows the source's respiratory-rate
measurements. The source's environmental time series remains in the original
workbook and is **not aligned per dog or phase** in the derived CSV.

## Appropriate use

This is a small, **single-breed, non-clinical** research example—not training
data for a fever/rabies classifier and not a table of safe breed thresholds.
The eye and abdomen temperatures were taken by infrared imaging, **not** by a
neck-contact NTC. A dog's `rectal_temp_c` and `gi_temp_c` must never be treated
as interchangeable with the collar's surface reading. The dataset has no
illness-severity labels, no sleep-state labels, and no SHT40 measurements.

Use it to explore how measurement site and environmental phase can change the
reading, and to test data-import code. Setting or validating collar alerts
requires paired readings from the actual NTC placement, SHT40, and a veterinary
reference thermometer across dogs, coats, environments, and health states.
