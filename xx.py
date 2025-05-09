from PIL import Image
import os

# Ayarlar
tif_path = "gorsel.tif"
output_dir = "parcalar"
tile_size = 840

# Klasörü oluştur
os.makedirs(output_dir, exist_ok=True)

# Görseli aç
with Image.open(tif_path) as img:
    width, height = img.size
    count = 0

    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            box = (left, top, left + tile_size, top + tile_size)
            tile = img.crop(box).convert("RGB")  # JPEG için RGB'ye çevir

            # Görsel sınırlarını aşmasın
            tile = tile.crop((0, 0, min(tile_size, tile.width), min(tile_size, tile.height)))

            tile.save(f"{output_dir}/tile_{count:04d}.jpg", "JPEG", quality=90)
            count += 1

print(f"{count} adet 840x840 parça kaydedildi.")
