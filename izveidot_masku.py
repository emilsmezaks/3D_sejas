
from rembg import remove
from PIL import Image
import io
import os

ievades_attels = 'demo/6.png'
izvades_fails = 'demo/6_mask.png'

if not os.path.exists(ievades_attels):
    print(f"Kļūda: {ievades_attels} nav atrasts!")
    exit()

print(f"1. Apstrādā attēlu: {ievades_attels}")

with open(ievades_attels, 'rb') as f:
    img = f.read()

print("2. Ģenerē masku")
mask = remove(img, only_mask=True)

mask_img = Image.open(io.BytesIO(mask))
mask_img.save(izvades_fails)

print(f"Maska saglabāta: {izvades_fails}")