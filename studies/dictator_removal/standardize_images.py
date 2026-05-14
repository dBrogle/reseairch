"""Standardize images: center-crop to square, resize to common size."""

from pathlib import Path
from PIL import Image

IMAGE_DIR = Path(__file__).parent / "data" / "images"
TARGET_SIZE = 512


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def main():
    for path in sorted(IMAGE_DIR.iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        img = Image.open(path)
        original_size = img.size
        img = center_crop_square(img)
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        img.save(path)
        print(f"{path.name}: {original_size} -> {img.size}")


if __name__ == "__main__":
    main()
