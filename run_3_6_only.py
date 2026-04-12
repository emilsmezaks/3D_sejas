import subprocess
import os

project_dir = r"D:\Download\Desktop\3D_Sejas"
os.chdir(project_dir)

# Apstrādā tikai 3 un 6
for num in [3, 6]:
    print(f"\n--- Processing {num}.png ---")
    
    # Pārbauda vai faili eksistē
    if not os.path.exists(f'demo/{num}.png'):
        print(f"  ERROR: demo/{num}.png not found!")
        continue
    if not os.path.exists(f'demo/{num}_mask.png'):
        print(f"  ERROR: demo/{num}_mask.png not found!")
        continue
    if not os.path.exists(f'demo/{num}_normal.png'):
        print(f"  ERROR: demo/{num}_normal.png not found!")
        continue
    
    # Izveido pagaidu main_test.py ar šo numuru
    with open('main_test.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace("'demo/1.png'", f"'demo/{num}.png'")
    content = content.replace("'demo/1_mask.png'", f"'demo/{num}_mask.png'")
    content = content.replace("'demo/1_normal.png'", f"'demo/{num}_normal.png'")
    content = content.replace("'demo/1.obj'", f"'demo/{num}.obj'")
    
    with open('main_test_temp.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Palaid
    result = subprocess.run(["python", "main_test_temp.py"])
    
    if result.returncode == 0:
        print(f"  OK - {num}.obj created!")
    else:
        print(f"  FAILED - {num}.obj")
    
    # Iztīra
    if os.path.exists('main_test_temp.py'):
        os.remove('main_test_temp.py')

print("\nDone!")