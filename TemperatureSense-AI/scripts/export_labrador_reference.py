"""Rebuild the research-reference CSV from the unmodified source workbook."""

import csv
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "neander_labrador_temperature" / "S1 Dataset.xlsx"
DESTINATION = ROOT / "data" / "derived" / "labrador_temperature_phases.csv"

FIELDS = (
    "dog_id",
    "coat_color",
    "phase",
    "body_condition_score",
    "weight_kg",
    "age_years",
    "sex",
    "rectal_temp_c",
    "gi_temp_c",
    "eye_surface_temp_c",
    "abdomen_surface_temp_c",
    "respiration_per_min",
)

PHASE_COLUMNS = (
    ("baseline", 5, 8, 11, 14, 17),
    ("sunlight", 6, 9, 12, 15, 18),
    ("cooling", 7, 10, 13, 16, 19),
)


def main() -> None:
    workbook = load_workbook(SOURCE, read_only=True, data_only=True)
    try:
        sheet = workbook["Dog data"]
        rows = []
        excluded = []
        for values in list(sheet.values)[1:]:
            dog_id = str(values[0]).strip()
            if not isinstance(values[5], (int, float)):
                excluded.append(dog_id)
                continue
            coat_color = dog_id.split()[0].lower()
            for phase, rectal, gi, eye, abdomen, respiration in PHASE_COLUMNS:
                rows.append(
                    (
                        dog_id,
                        coat_color,
                        phase,
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                        values[rectal],
                        values[gi],
                        values[eye],
                        values[abdomen],
                        values[respiration],
                    )
                )
        if len(rows) != 45 or excluded != ["Yellow 8"]:
            raise ValueError(f"Unexpected source records: {len(rows)} rows, {excluded} excluded")
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        with DESTINATION.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            writer.writerows(rows)
    finally:
        workbook.close()


if __name__ == "__main__":
    main()
