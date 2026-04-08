import shutil
import os

src = r"C:\Users\USER\.gemini\antigravity\brain\0f1be3c7-678b-42ee-ac7e-64833766186c\media__1775641494024.png"
dst = r"C:\Users\USER\OneDrive\Documents\script_voice_over\static\logo_rkb.png"

shutil.copy2(src, dst)
print(f"✅ Logo berhasil disalin ke: {dst}")
print(f"   Ukuran file: {os.path.getsize(dst):,} bytes")
