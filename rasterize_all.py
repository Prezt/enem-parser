"""Re-rasterize all year/day PDFs with the new page-d{day}-{year}-NN.png naming."""
from __future__ import annotations

from pathlib import Path

from enem_extractor.pdf import rasterize_pages

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
DAYS = [1, 2]


def main() -> None:
    for year in YEARS:
        pages_dir = OUTPUT_DIR / str(year) / "pages"
        if not pages_dir.exists():
            print(f"[skip] {pages_dir} does not exist")
            continue

        for day in DAYS:
            pdfs = sorted(DATA_DIR.glob(f"{year}_PV_impresso_D{day}_*.pdf"))
            if not pdfs:
                print(f"[skip] no PDF found for {year} D{day}")
                continue

            pdf = pdfs[0]
            prefix = f"page-d{day}-{year}"

            # Skip if already rasterized with this prefix
            existing = sorted(pages_dir.glob(f"{prefix}-*.png"))
            if existing:
                print(f"[skip] {year} D{day}: {len(existing)} pages already exist ({existing[0].name} ...)")
                continue

            print(f"[rasterize] {pdf.name} -> {pages_dir}/{prefix}-NN.png")
            pages = rasterize_pages(pdf, pages_dir, dpi=200, prefix=prefix)
            print(f"  {len(pages)} pages written")

        # Remove old page-NN.png files
        old = sorted(pages_dir.glob("page-[0-9]*.png"))
        if old:
            for f in old:
                f.unlink()
            print(f"[clean] {year}: removed {len(old)} old page-NN.png files")


if __name__ == "__main__":
    main()
