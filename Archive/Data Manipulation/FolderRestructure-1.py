import os
import shutil

base_dir = r"D:\WSU\Animal data may 2025 visit\copy of the 13C data\2025-05-07-animal injection coil #9\animal05"
filenames = [
    "acqu.par", "data.1d", "data.csv", "data.png", "data.pt1",
    "spectrum.1d", "spectrum.csv", "spectrum.png", "spectrum.pt1"
]

for n in range(1, 45):
    subfolder = os.path.join(base_dir, str(n), "1")
    target_folder = os.path.join(base_dir, str(n))

    for fname in filenames:
        old_path = os.path.join(subfolder, fname)
        new_path = os.path.join(target_folder, fname)

        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
            print(f"Moved: {old_path} -> {new_path}")
        else:
            print(f"Missing: {old_path}")

    # Attempt to remove the now-empty "1" folder
    try:
        os.rmdir(subfolder)
        print(f"Removed folder: {subfolder}")
    except OSError:
        print(f"Could not remove folder (not empty?): {subfolder}")