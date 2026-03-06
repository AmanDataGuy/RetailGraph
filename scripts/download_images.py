import pandas as pd
import requests
import os
import csv
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV = "data/raw/train.csv"
IMAGE_DIR = "data/images/train"
REPORT_CSV = "data/raw/download_report.csv"
TIMEOUT = 5
MAX_WORKERS = 10

Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)


def download_image(row):
    sample_id = row["sample_id"]
    url = row["image_link"]
    save_path = os.path.join(IMAGE_DIR, f"{sample_id}.jpg")

    # skip if already downloaded
    if os.path.exists(save_path):
        return {"sample_id": sample_id, "url": url, "status": "already_exists"}

    try:
        response = requests.get(url, timeout=TIMEOUT)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return {"sample_id": sample_id, "url": url, "status": "success"}
        elif response.status_code == 404:
            return {"sample_id": sample_id, "url": url, "status": "404"}
        else:
            return {"sample_id": sample_id, "url": url, "status": f"error_{response.status_code}"}
    except requests.exceptions.Timeout:
        return {"sample_id": sample_id, "url": url, "status": "timeout"}
    except Exception as e:
        return {"sample_id": sample_id, "url": url, "status": f"failed_{str(e)[:50]}"}


def main():
    df = pd.read_csv(INPUT_CSV)
    rows = df[["sample_id", "image_link"]].to_dict("records")

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_image, row): row for row in rows}
        for future in tqdm(as_completed(futures), total=len(rows), desc="Downloading images"):
            results.append(future.result())

    # ── Save report ───────────────────────────────────────────────────────────
    with open(REPORT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "url", "status"])
        writer.writeheader()
        writer.writerows(results)

    # ── Summary ───────────────────────────────────────────────────────────────
    report_df = pd.DataFrame(results)
    print("\n── Download Report ──────────────────────────────")
    print(report_df["status"].value_counts().to_string())
    print(f"\nTotal: {len(report_df)}")
    print(f"Usable (success + already_exists): {len(report_df[report_df['status'].isin(['success', 'already_exists'])])}")
    print(f"Report saved to: {REPORT_CSV}")


if __name__ == "__main__":
    main()