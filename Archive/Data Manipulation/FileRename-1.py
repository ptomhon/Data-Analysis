import os

base_dir = r"D:\WSU\Animal data may 2025 visit\copy of the 13C data\2025-05-07-animal injection coil #9\animal05"

for n in range(1, 45):
    folder = os.path.join(base_dir, str(n))
    old_file = os.path.join(folder, "data.csv")
    new_file = os.path.join(folder, "fid.csv")

    if os.path.exists(old_file):
        os.rename(old_file, new_file)
        print(f"Renamed: {old_file} -> {new_file}")
    else:
        print(f"File not found: {old_file}")
