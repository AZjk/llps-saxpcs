from pathlib import Path
import shutil
import h5py
import tqdm
import os
import traceback
import concurrent.futures

META_TEMPLATE = Path("/home/beams10/8IDIUSER/Documents/llps-saxpcs/reanalysis_2025_0428/sample_metadata.hdf")
MAX_DEPTH = 5

def process_folder(source_folder, dest_folder, max_workers=None):
    source_folder = Path(source_folder)
    dest_folder = Path(dest_folder)

    dest_folder.mkdir(parents=True, exist_ok=True)

    all_subfolders = list(walk_subfolders(source_folder, max_depth=MAX_DEPTH))

    tasks = [(subfolder, source_folder, dest_folder) for subfolder in all_subfolders]

    if max_workers == 1:
        for task in tqdm.tqdm(tasks, desc="Processing subfolders"):
            worker_process_subfolder(task)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm.tqdm(executor.map(worker_process_subfolder, tasks), total=len(tasks), desc="Processing subfolders"))

def walk_subfolders(base_path, max_depth, current_depth=0):
    if current_depth > max_depth:
        return

    for entry in base_path.iterdir():
        if entry.is_dir():
            yield entry
            yield from walk_subfolders(entry, max_depth, current_depth + 1)

def worker_process_subfolder(args):
    subfolder_path, source_folder, dest_folder = args
    process_subfolder(Path(subfolder_path), Path(source_folder), Path(dest_folder))

def process_subfolder(subfolder_path, source_folder, dest_folder):
    try:
        bin_files = list(subfolder_path.glob("*.bin"))
        hdf_files = list(subfolder_path.glob("*.hdf"))

        if len(bin_files) != 1:
            print(f"Skipping {subfolder_path}: found {len(bin_files)} .bin files.")
            return
        if len(hdf_files) != 1:
            print(f"Skipping {subfolder_path}: found {len(hdf_files)} .hdf files.")
            return

        relative_path = subfolder_path.relative_to(source_folder)
        save_folder = dest_folder / relative_path
        save_folder.mkdir(parents=True, exist_ok=True)

        bin_file = bin_files[0]
        rawmeta_file = hdf_files[0]

        linkname = save_folder / bin_file.name
        metaname = linkname.with_name(linkname.stem + "_metadata.hdf")

        # Create symlink safely
        if linkname.exists():
            if linkname.is_symlink() and linkname.resolve() == bin_file.resolve():
                pass  # Symlink already correct
            else:
                linkname.unlink()
                os.symlink(bin_file, linkname)
        else:
            os.symlink(bin_file, linkname)

        # Copy the template metadata file
        shutil.copy2(META_TEMPLATE, metaname)

        # Read exposure time from original metadata
        with h5py.File(rawmeta_file, "r") as f:
            exptime = f["/measurement/instrument/detector/exposure_period"][()]

        # Modify the copied metadata
        with h5py.File(metaname, "r+") as f:
            frame_time_path = "/entry/instrument/detector_1/frame_time"
            if frame_time_path in f:
                del f[frame_time_path]
            f.create_dataset(frame_time_path, data=exptime)
        
        temp_all = {} 
        with h5py.File(rawmeta_file, "r") as f:
            for zone_idx in [1, 2, 3]:
                temp_data = f[f"/measurement/sample/QNW_Zone{zone_idx}_Temperature"][()]
                temp_all[zone_idx] = temp_data

        with h5py.File(metaname, "r+") as f:
            for zone_idx in [1, 2, 3]:
                key = f"/entry/sample/qnw{zone_idx}_temperature"
                del f[key]
                f[key] = temp_all[zone_idx]

    except Exception as e:
        print(f"Error processing {subfolder_path}: {e}")
        traceback.print_exc()
        relative_path = subfolder_path.relative_to(source_folder)
        save_folder = dest_folder / relative_path
        if save_folder.exists():
            shutil.rmtree(save_folder)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process SAXPCS data folders.")
    parser.add_argument("source_folder", type=str, help="Source folder containing data.")
    parser.add_argument("dest_folder", type=str, help="Destination folder for processed data.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes. Set to 1 for sequential processing.")

    args = parser.parse_args()

    process_folder(args.source_folder, args.dest_folder, max_workers=args.workers)
