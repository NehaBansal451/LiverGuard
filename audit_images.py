import pandas as pd
from PIL import Image
from pathlib import Path
from collections import Counter

manifest = Path(
    "demo_workspace/dataset/"
    "patient_level_manifest_seed42_f9452f1f5b23.csv"
)

df = pd.read_csv(manifest)

formats = Counter()
dimensions = Counter()
small_images = []
large_images = []
failed = []

print(f"Inspecting {len(df)} images...")

for _, row in df.iterrows():
    path = Path(row["path"])

    try:
        with Image.open(path) as img:
            formats[img.format] += 1
            dimensions[img.size] += 1

            width, height = img.size

            if width < 128 or height < 128:
                small_images.append((str(path), img.size))

            if width > 4096 or height > 4096:
                large_images.append((str(path), img.size))

    except Exception as e:
        failed.append((str(path), str(e)))

print("\nIMAGE FORMATS:")
for fmt, count in formats.items():
    print(fmt, ":", count)

print("\nMOST COMMON DIMENSIONS:")
for size, count in dimensions.most_common(15):
    print(size, ":", count)

print("\nUNUSUALLY SMALL IMAGES:", len(small_images))
print("UNUSUALLY LARGE IMAGES:", len(large_images))
print("FAILED IMAGES:", len(failed))

if small_images:
    print("\nFIRST 10 SMALL IMAGES:")
    for item in small_images[:10]:
        print(item)

if large_images:
    print("\nFIRST 10 LARGE IMAGES:")
    for item in large_images[:10]:
        print(item)