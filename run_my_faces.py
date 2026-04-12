import subprocess
import os

# Jūsu seju saraksts
faces = ['Face_1', 'Face_2', 'Face_3', 'Face_4', 'Face_5', 'Face_6',
         'Face_7', 'Face_8', 'Face_9', 'Face_10', 'Face_11', 'Face_12']

print("="*50)
print("SeIF - 12 SEJU REKONSTRUKCIJA")
print("="*50)

for i, face in enumerate(faces, 1):
    print(f"\n[{i}/12] Rekonstruēju: {face}")
    
    # Pārbauda vai visi faili eksistē
    if not os.path.exists(f'demo/{face}.png'):
        print(f"  ✗ Trūkst: demo/{face}.png")
        continue
    if not os.path.exists(f'demo/{face}_mask.png'):
        print(f"  ✗ Trūkst: demo/{face}_mask.png")
        continue
    if not os.path.exists(f'demo/{face}_normal.png'):
        print(f"  ✗ Trūkst: demo/{face}_normal.png")
        continue
    
    # Izveido main_test.py ar šo seju - izmanto UTF-8 kodējumu
    with open('main_test.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace("'demo/1.png'", f"'demo/{face}.png'")
    content = content.replace("'demo/1_mask.png'", f"'demo/{face}_mask.png'")
    content = content.replace("'demo/1_normal.png'", f"'demo/{face}_normal.png'")
    content = content.replace("'demo/1.obj'", f"'demo/{face}.obj'")
    
    with open('main_test_temp.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Palaid rekonstrukciju
    result = subprocess.run(["python", "main_test_temp.py"])
    
    if result.returncode == 0:
        print(f"  ✓ {face}.obj izveidots!")
    else:
        print(f"  ✗ Kļūda ar {face}")
    
    # Izdzēš pagaidu failu
    if os.path.exists('main_test_temp.py'):
        os.remove('main_test_temp.py')

print("\n" + "="*50)
print("REKONSTRUKCIJA PABEIGTA!")
print("Rezultāti atrodas: demo/*.obj")
print("="*50)