from netCDF4 import Dataset
import math
import os
import gc
import multiprocessing
import functools
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed

def print_netcdf_info(file_path):
    """
    Print information about a NetCDF file, including dimensions, variables, and attributes.
    """

    # Open the NetCDF file in read mode ('r')
    try:
        nc_file = Dataset(file_path, 'r')

        # Print global attributes
        print("Global Attributes:")
        for attr_name in nc_file.ncattrs():
            print(f"{attr_name}: {nc_file.getncattr(attr_name)}")

        # Print dimensions
        print("\nDimensions:")
        for dim_name, dim in nc_file.dimensions.items():
            print(f"{dim_name}: size = {len(dim)}")

        # Print variables
        print("\nVariables:")
        for var_name, var in nc_file.variables.items():
            print(f"{var_name}: shape = {var.shape}, dtype = {var.dtype}")
            # To access variable data:
            # data = nc_file.variables[var_name][:]
            # print(data)

        # Close the NetCDF file
        nc_file.close()

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")




def open_netcdf(file_path):
    """
    Open a NetCDF file and return the dataset object.
    """

    # Open the NetCDF file in read mode ('r')
    try:
        nc_file = Dataset(file_path, 'r')
        return nc_file
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def close_netcdf(nc_file):
    """
    Close a NetCDF file.
    """

    try:
        nc_file.close()
    except Exception as e:
        print(f"An error occurred: {e}")

def plot_2d_variables(nc_file):
    """
    Plots all 2D variables (shape (2240, 2856) after squeezing) from a netCDF file.
    
    Parameters:
        nc_file: A netCDF file object (e.g., returned by netCDF4.Dataset or xarray.open_dataset)
    """
    variables_to_plot = []
    
    # Loop over all variables in the file
    for var_name, var in nc_file.variables.items():
        # Try to load the data and squeeze it to remove any singleton dimensions
        try:
            data = var[:].squeeze()
        except Exception as e:
            print(f"Skipping variable {var_name} due to error: {e}")
            continue
        
        # Check if the squeezed data is 2D and matches the expected dimensions.
        if data.ndim == 2 and data.shape == (2240, 2856):
            variables_to_plot.append((var_name, data))
    
    n_plots = len(variables_to_plot)
    if n_plots == 0:
        print("No 2D variables with shape (2240, 2856) found.")
        return
    
    # Determine grid layout (e.g., roughly square)
    ncols = math.ceil(math.sqrt(n_plots))
    nrows = math.ceil(n_plots / ncols)
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    
    # Flatten the axs array for easy iteration; handle case of a single subplot.
    if n_plots == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    
    # Loop through each variable and plot its data
    for ax, (var_name, data) in zip(axs, variables_to_plot):
        # Use imshow for 2D array plotting
        cax = ax.imshow(data, origin='lower', aspect='auto')
        ax.set_title(var_name)
        fig.colorbar(cax, ax=ax, orientation='vertical')
    
    # Hide any extra axes if the grid is larger than the number of plots
    for ax in axs[n_plots:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()

def random_crop_and_partition(image, crop_size=2048, patch_size=256):
    """
    Takes a 2D image (e.g., shape (2240, 2856)), performs a random crop of size crop_size x crop_size,
    and partitions that crop into non-overlapping patches of size patch_size x patch_size.

    Parameters:
        image (np.ndarray): Input 2D array with shape (2240, 2856).
        crop_size (int): The size of the square crop. Default is 2048.
        patch_size (int): The size of each square patch. Default is 256.
    
    Returns:
        crop (np.ndarray): The cropped 2048x2048 image.
        patches (np.ndarray): An array of patches with shape (num_rows, num_cols, patch_size, patch_size),
                              where num_rows = num_cols = crop_size / patch_size (i.e., 8).
    """
    H, W = image.shape
    # Ensure the image is large enough for the crop
    if H < crop_size or W < crop_size:
        raise ValueError("Image dimensions are smaller than the desired crop size.")

    # Determine maximum starting indices so the crop fits in the image
    max_y = H - crop_size
    max_x = W - crop_size

    # Choose a random starting position for the crop
    start_y = np.random.randint(0, max_y + 1)
    start_x = np.random.randint(0, max_x + 1)
    
    # Extract the crop
    crop = image[start_y:start_y + crop_size, start_x:start_x + crop_size]

    # Number of patches along each dimension (should be 8 for a 2048x2048 crop and 256x256 patches)
    num_patches = crop_size // patch_size

    # Partition the crop into patches
    patches = np.empty((num_patches, num_patches, patch_size, patch_size), dtype=crop.dtype)
    for i in range(num_patches):
        for j in range(num_patches):
            y0 = i * patch_size
            x0 = j * patch_size
            patches[i, j] = crop[y0:y0 + patch_size, x0:x0 + patch_size]

    return crop, patches

def process_nc_file(nc_file, crop_size=2048, patch_size=256):
    """
    Processes each 2D variable in the netCDF file with shape (2240, 2856). For each such variable, a random 
    2048 x 2048 crop is taken and partitioned into non-overlapping 256 x 256 patches.
    
    Parameters:
        nc_file: The netCDF file object (e.g., from netCDF4.Dataset or similar).
        crop_size (int): The size of the square crop to extract. Default is 2048.
        patch_size (int): The size of each patch. Default is 256.
    
    Returns:
        results (dict): A dictionary where keys are variable names and values are dictionaries 
                        with keys 'crop' (the 2048x2048 crop) and 'patches' (the 4D array of patches).
    """
    results = {}
    
    # Loop over variables in the netCDF file.
    for var_name, var in nc_file.variables.items():
        try:
            # Get the variable data and squeeze singleton dimensions.
            data = var[:].squeeze()
        except Exception as e:
            print(f"Skipping variable '{var_name}' due to error: {e}")
            continue
        
        # Check for a 2D array with the desired shape.
        if data.ndim == 2 and data.shape == (2240, 2856):
            crop, patches = random_crop_and_partition(data, crop_size, patch_size)
            results[var_name] = {'crop': crop, 'patches': patches}
            print(f"Processed variable '{var_name}': Crop shape {crop.shape}, Patches shape {patches.shape}")
    
    if not results:
        print("No 2D variables with shape (2240, 2856) were found in the file.")
    return results

def plot_crop_and_patches(crop, patches, cmap='viridis'):
    """
    Plots the full crop image alongside its grid of patches with a unified color scale.
    This version displays the images with the top of the array at the top (origin='upper').

    Parameters:
        crop (np.ndarray): A 2D numpy array (e.g., 2048 x 2048) representing the full crop.
        patches (np.ndarray): A 4D numpy array with shape 
                              (num_rows, num_cols, patch_size, patch_size)
                              for example, (8, 8, 256, 256).
        cmap (str): The colormap to use (default 'viridis').
    """
    # Compute global color limits from the crop so that both plots share the same scale.
    vmin, vmax = crop.min(), crop.max()
    
    # Create a figure with two main columns: one for the full crop and one for the patches grid.
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # Left: plot the full crop image.
    ax_full = fig.add_subplot(gs[0])
    im_full = ax_full.imshow(crop, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)
    ax_full.set_title("Full Crop")
    ax_full.axis('off')
    plt.colorbar(im_full, ax=ax_full, fraction=0.046, pad=0.04)
    
    # Right: plot the patches grid.
    n_rows, n_cols, patch_h, patch_w = patches.shape
    gs_patches = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, subplot_spec=gs[1],
        wspace=0.05, hspace=0.05
    )
    
    for i in range(n_rows):
        for j in range(n_cols):
            ax = fig.add_subplot(gs_patches[i, j])
            ax.imshow(patches[i, j], origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_nc_file_two_channels(nc_file, crop_size=2048, patch_size=256):
    """
    Processes a netCDF file and extracts one or both channels among:
      - "ir_brightness_temperature"
      - "visible_reflectance"
    
    For each available channel, a random 2048 x 2048 crop is taken from the original image 
    (expected shape: 2240 x 2856). The crop is then partitioned into non-overlapping 256 x 256 patches.
    
    If both channels are available, returns:
      - crop_array: shape (2, 2048, 2048)
      - patches_array: shape (8, 8, 2, 256, 256)
    
    If only one channel is available, returns:
      - crop_array: shape (1, 2048, 2048)
      - patches_array: shape (8, 8, 1, 256, 256)
    
    If none of the channels is available, returns None.
    """
    required_channels = ["ir_brightness_temperature", "visible_reflectance"]
    available_channels = {}
    
    # Attempt to load each required channel if available.
    for channel in required_channels:
        if channel in nc_file.variables:
            try:
                data = nc_file.variables[channel][:].squeeze()
                available_channels[channel] = data
            except Exception as e:
                print(f"Error reading channel '{channel}': {e}")
    
    if len(available_channels) == 0:
        # None of the required channels is available.
        return None
    
    # Use one channel to determine image dimensions (assumed consistent across channels).
    first_channel_data = next(iter(available_channels.values()))
    expected_shape = (2240, 2856)
    if first_channel_data.shape != expected_shape:
        raise ValueError(f"Expected channel data shape {expected_shape}, but got {first_channel_data.shape}")
    
    # Determine random crop coordinates from the available data.
    H, W = expected_shape
    if H < crop_size or W < crop_size:
        raise ValueError("Channel data is smaller than the desired crop size.")
    max_y = H - crop_size
    max_x = W - crop_size
    start_y = np.random.randint(0, max_y + 1)
    start_x = np.random.randint(0, max_x + 1)
    
    crop_dict = {}
    patches_dict = {}
    for channel, data in available_channels.items():
        # Extract the same crop for each available channel.
        crop_channel = data[start_y:start_y + crop_size, start_x:start_x + crop_size]
        crop_dict[channel] = crop_channel
        
        # Partition the crop into patches.
        num_patches = crop_size // patch_size  # Expected to be 8.
        patches_channel = crop_channel.reshape(num_patches, patch_size, num_patches, patch_size).transpose(0, 2, 1, 3)
        patches_dict[channel] = patches_channel
    
    # Combine channels in a consistent order.
    channels_ordered = [ch for ch in required_channels if ch in available_channels]
    crop_list = [crop_dict[ch] for ch in channels_ordered]
    patches_list = [patches_dict[ch] for ch in channels_ordered]
    
    crop_array = np.stack(crop_list, axis=0)  # Shape: (n_channels, 2048, 2048)
    patches_array = np.stack(patches_list, axis=2)  # Shape: (8, 8, n_channels, 256, 256)
    
    return crop_array, patches_array


# Global lock for multiprocessing checkpoint writes.
global_lock = None

def init_lock(lock):
    global global_lock
    global_lock = lock

def process_single_file(file_path, output_dir, checkpoint_file, crop_size=2048, patch_size=256):
    """
    Processes a single netCDF file:
      - Opens the file.
      - Applies process_nc_file_two_channels.
      - If at least one required channel is available, saves the results as a compressed .npz file 
        in output_dir and records the file in the checkpoint.
      - If none of the channels is available, simply passes.
    
    Returns a tuple (file_path, status).
    """
    base_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, os.path.splitext(base_name)[0] + '.npz')
    
    # Skip if output file already exists.
    if os.path.exists(output_file):
        print(f"Skipping {file_path} (output file exists).")
        return file_path, "skipped"
    
    try:
        with Dataset(file_path, 'r') as nc_file:
            result = process_nc_file_two_channels(nc_file, crop_size, patch_size)
        
        # If none of the required channels is present, just skip saving.
        if result is None:
            print(f"Skipping {file_path} (no required channels found).")
            return file_path, "skipped (no channels)"
        
        crop, patches = result
        
        # Save the resulting arrays.
        np.savez_compressed(output_file, crop=crop, patches=patches)
        
        # Update checkpoint.
        global global_lock
        if global_lock is not None:
            with global_lock:
                with open(checkpoint_file, 'a') as f:
                    f.write(file_path + '\n')
                    
        print(f"Processed {file_path}")
        return file_path, "processed"
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return file_path, f"error: {e}"

def process_directory_tree(input_dir, output_dir, checkpoint_file, n_workers=4, crop_size=2048, patch_size=256):
    """
    Walks through input_dir to locate netCDF files (files ending with '.nc') and processes them
    in parallel using multiprocessing. For each netCDF file, process_nc_file_two_channels is applied.
    The output is saved in output_dir as a .npz file only if at least one required channel is available.
    A checkpoint file is maintained to log processed files so that the run can be resumed after interruption.
    
    Parameters:
      - input_dir (str): Root directory to search for netCDF files.
      - output_dir (str): Directory where output .npz files will be saved.
      - checkpoint_file (str): Path to the checkpoint file for processed file paths.
      - n_workers (int): Number of parallel worker processes.
      - crop_size (int): Size of the square crop. Default is 2048.
      - patch_size (int): Size of the square patches. Default is 256.
      
    Returns:
      - results: A list of tuples (file_path, status) indicating the processing status for each file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load already processed files from the checkpoint.
    processed_files = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_files = {line.strip() for line in f}
    
    # Walk through the directory tree to collect .nc files.
    netcdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.nc'):
                full_path = os.path.join(root, file)
                if full_path not in processed_files:
                    netcdf_files.append(full_path)
    
    print(f"Found {len(netcdf_files)} netCDF files to process.")
    
    # Create a multiprocessing lock.
    lock = multiprocessing.Lock()
    
    # Create a pool of worker processes, each with the shared lock.
    pool = multiprocessing.Pool(processes=n_workers, initializer=init_lock, initargs=(lock,))
    
    # Use functools.partial to pass constant parameters.
    worker_func = functools.partial(
        process_single_file,
        output_dir=output_dir,
        checkpoint_file=checkpoint_file,
        crop_size=crop_size,
        patch_size=patch_size
    )
    
    # Process files in parallel.
    results = pool.map(worker_func, netcdf_files)
    
    pool.close()
    pool.join()
    
    print("Processing complete.")
    return results

def process_file(file_path):
    """
    Load and process a single npz file.
    Uses a context manager with np.load so resources are released promptly.
    Returns the file path along with the processed crops and patches arrays.
    """
    with np.load(file_path) as data:
        crops = data['crop']  # e.g. (2, 2048, 2048) or (1, 2048, 2048)
        patches = data['patches']  # e.g. (8,8,C,256,256)
        patches_reshaped = patches.reshape(-1, patches.shape[-2], patches.shape[-1])
    return file_path, crops, patches_reshaped

def get_next_chunk_index(folder_path, crops_prefix='crops_', ext='.pt'):
    """
    Determine the next available chunk index by listing files that start with crops_
    (e.g. crops_1.pt, crops_2.pt, …) and returning the next index.
    """
    existing = [f for f in os.listdir(folder_path) if f.startswith(crops_prefix) and f.endswith(ext)]
    indices = []
    for fname in existing:
        try:
            idx = int(fname[len(crops_prefix):-len(ext)])
            indices.append(idx)
        except ValueError:
            pass
    return max(indices) + 1 if indices else 1

def flush_chunk(folder_path, partial_crops_list, partial_patches_list):
    """
    Concatenate the partial lists, save them as a new chunk,
    and then clear the lists.
    """
    combined_crops = np.concatenate(partial_crops_list, axis=0)
    combined_patches = np.concatenate(partial_patches_list, axis=0)
    
    chunk_index = get_next_chunk_index(folder_path)
    crops_chunk_name = f"crops_{chunk_index}.pt"
    patches_chunk_name = f"patches_{chunk_index}.pt"
    crops_chunk_path = os.path.join(folder_path, crops_chunk_name)
    patches_chunk_path = os.path.join(folder_path, patches_chunk_name)
    
    # Save the chunks
    torch.save(torch.from_numpy(combined_crops), crops_chunk_path)
    torch.save(torch.from_numpy(combined_patches), patches_chunk_path)
    print(f"Saved chunk {chunk_index}: crops {combined_crops.shape}, patches {combined_patches.shape}")
    
    # Clear memory: clear the lists and delete temporary variables.
    partial_crops_list.clear()
    partial_patches_list.clear()
    del combined_crops, combined_patches
    gc.collect()

def process_npz_files_progressive(folder_path, 
                                  files_per_chunk=5,
                                  crops_partial_name='crops_partial.pt',
                                  patches_partial_name='patches_partial.pt',
                                  checkpoint_filename='checkpoint.txt'):
    """
    Process npz files in parallel, but accumulate data in memory only in chunks.
    Uses a checkpoint file and partial chunk files so that processing can resume.
    """
    checkpoint_path = os.path.join(folder_path, checkpoint_filename)
    
    # Load processed files from checkpoint if it exists
    processed_files = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            processed_files = {line.strip() for line in f if line.strip()}
        print(f"Resuming: {len(processed_files)} files already processed.")
    
    # List all npz files and filter out already processed ones
    all_npz_files = sorted([os.path.join(folder_path, f)
                            for f in os.listdir(folder_path) if f.endswith('.npz')])
    remaining_files = [f for f in all_npz_files if f not in processed_files]
    print(f"Total files: {len(all_npz_files)}. Remaining: {len(remaining_files)}.")
    
    # Paths for partial (incomplete chunk) files
    crops_partial_path = os.path.join(folder_path, crops_partial_name)
    patches_partial_path = os.path.join(folder_path, patches_partial_name)
    
    # Initialize accumulators for the current chunk.
    partial_crops_list = []
    partial_patches_list = []
    partial_file_count = 0
    
    # Try to resume from a saved partial chunk (if available)
    if os.path.exists(crops_partial_path) and os.path.exists(patches_partial_path):
        try:
            existing_crops = torch.load(crops_partial_path).numpy()
            existing_patches = torch.load(patches_partial_path).numpy()
            partial_crops_list.append(existing_crops)
            partial_patches_list.append(existing_patches)
            # Optionally, you might record how many files these correspond to.
            print(f"Loaded partial chunk: crops {existing_crops.shape}, patches {existing_patches.shape}")
            del existing_crops, existing_patches
            gc.collect()
        except Exception as e:
            print(f"Error loading partial chunk: {e}. Starting fresh for partial chunk.")
    
    # Process files in parallel.
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file_path): file_path 
                          for file_path in remaining_files}
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                fname, crops, patches = future.result()
            except Exception as exc:
                print(f"Error processing {file_path}: {exc}")
                continue
            
            # Append the processed arrays to the partial lists
            partial_crops_list.append(crops)
            partial_patches_list.append(patches)
            partial_file_count += 1
            
            # Update the checkpoint file with the newly processed file
            with open(checkpoint_path, 'a') as cp_file:
                cp_file.write(fname + "\n")
            
            print(f"Processed {fname} (added to current chunk; count = {partial_file_count}).")
            
            # Save the current partial chunk to disk so that progress is not lost.
            try:
                combined_crops = np.concatenate(partial_crops_list, axis=0)
                combined_patches = np.concatenate(partial_patches_list, axis=0)
                torch.save(torch.from_numpy(combined_crops), crops_partial_path)
                torch.save(torch.from_numpy(combined_patches), patches_partial_path)
                print(f"Updated partial chunk on disk: crops {combined_crops.shape}, patches {combined_patches.shape}")
                del combined_crops, combined_patches
                gc.collect()
            except Exception as e:
                print(f"Error saving partial chunk: {e}")
            
            # If the accumulated number of files reaches the threshold, flush them as a new chunk.
            if partial_file_count >= files_per_chunk:
                flush_chunk(folder_path, partial_crops_list, partial_patches_list)
                partial_file_count = 0
                
                # Remove the partial chunk files (they have been flushed)
                if os.path.exists(crops_partial_path):
                    os.remove(crops_partial_path)
                if os.path.exists(patches_partial_path):
                    os.remove(patches_partial_path)
    
    # Flush any remaining partial data as a final chunk.
    if partial_file_count > 0:
        flush_chunk(folder_path, partial_crops_list, partial_patches_list)
        if os.path.exists(crops_partial_path):
            os.remove(crops_partial_path)
        if os.path.exists(patches_partial_path):
            os.remove(patches_partial_path)
    else:
        print("No remaining partial data to flush.")

def process_chunk_file(chunk_file_path, file_type,
                         max_deviation_factor=3.0,
                         global_stats=None, global_lock=None,
                         global_mean_deviation_factor=3.0):
    """
    Worker function to process one chunk file.

    Parameters:
      chunk_file_path: Full path of the chunk file (e.g., 'crops_1.pt' or 'patches_2.pt').
      file_type: A string, either 'crop' or 'patch'.
      max_deviation_factor: Local threshold; a slice is discarded if the maximum absolute deviation
                            from its mean exceeds this factor times its standard deviation.
      global_stats: A shared dictionary (via Manager) holding keys "sum", "square_sum", "count"
                    for computing the running mean of accepted slice means.
      global_lock:  A multiprocessing Lock to synchronize updates to global_stats.
      global_mean_deviation_factor: Global threshold; a slice is discarded if its mean deviates
                                    from the current global mean by more than this factor times the
                                    global standard deviation.
                                    
    The function:
      1. Loads the tensor from chunk_file_path.
      2. For each slice (i.e. along the first dimension):
         - Discards it if it contains any NaN values.
         - Computes its own mean and std; if the maximum absolute deviation from its mean is
           larger than max_deviation_factor * std, it is discarded.
         - If global_stats is provided and already contains accepted data (count > 0),
           the slice is rejected if |slice_mean - global_mean| > global_mean_deviation_factor * global_std.
         - If accepted, updates global_stats (under lock) and saves the slice as an individual file.
      3. Removes the original chunk file.
      4. Clears memory and returns the processed chunk_file_path.
    """
    try:
        tensor = torch.load(chunk_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load {chunk_file_path}: {e}")
    
    # Extract chunk index (assumes filename pattern like: crops_3.pt)
    base_name = os.path.basename(chunk_file_path)
    try:
        chunk_index = base_name.split('_')[1].split('.')[0]
    except Exception as e:
        raise ValueError(f"Unexpected filename format for {base_name}: {e}")
    
    num_items = tensor.shape[0]
    valid_count = 0  # Counter for accepted slices (for naming individual files)
    for i in range(num_items):
        item = tensor[i]
        # --- Local checks: discard if any NaN ---
        if torch.isnan(item).any():
            print(f"Skipping {file_type} from chunk {chunk_index} at index {i} due to NaN values.")
            continue
        
        # --- Local statistical check based on the slice's own stats ---
        item_mean = torch.mean(item).item()
        item_std = torch.std(item).item()
        if item_std > 0:
            # Here we compute the maximum absolute deviation from the slice's mean.
            # (Converting torch.max(item - item_mean) to a python float.)
            max_dev = torch.max(torch.abs(item - item_mean)).item()
            if max_dev > max_deviation_factor * item_std:
                print(f"Skipping {file_type} from chunk {chunk_index} at index {i} due to extreme local deviation (max_dev={max_dev:.3f}, threshold={max_deviation_factor * item_std:.3f}).")
                continue

        # --- Global mean check ---
        # If global_stats has been initialized and at least one slice has been accepted:
        if global_stats is not None and global_lock is not None:
            with global_lock:
                count = global_stats["count"]
                if count > 0:
                    g_mean = global_stats["sum"] / count
                    g_var = (global_stats["square_sum"] / count) - (g_mean ** 2)
                    if g_var < 0:
                        g_var = 0.0
                    g_std = g_var ** 0.5
                else:
                    g_mean = item_mean
                    g_std = 0.0
            if count > 0 and g_std > 0:
                if abs(item_mean - g_mean) > global_mean_deviation_factor * g_std:
                    print(f"Skipping {file_type} from chunk {chunk_index} at index {i} due to global mean deviation (slice mean={item_mean:.3f}, global mean={g_mean:.3f}, threshold={global_mean_deviation_factor * g_std:.3f}).")
                    continue

        # --- If passed all checks, update global_stats ---
        if global_stats is not None and global_lock is not None:
            with global_lock:
                global_stats["sum"] += item_mean
                global_stats["square_sum"] += item_mean ** 2
                global_stats["count"] += 1

        # --- Save the accepted slice as an individual file ---
        out_file = os.path.join(os.path.dirname(chunk_file_path),
                                f"{file_type}_{chunk_index}_{valid_count}.pt")
        try:
            torch.save(item, out_file)
        except Exception as e:
            raise RuntimeError(f"Error saving {out_file}: {e}")
        valid_count += 1

    # Remove the original chunk file after processing.
    try:
        os.remove(chunk_file_path)
    except Exception as e:
        raise RuntimeError(f"Error removing {chunk_file_path}: {e}")
    
    # Free memory and force garbage collection.
    del tensor
    gc.collect()
    return chunk_file_path

def process_chunk_files_parallel(folder_path, type_prefix, file_type,
                                 checkpoint_filename,
                                 max_deviation_factor=3.0,
                                 global_stats=None, global_lock=None,
                                 global_mean_deviation_factor=3.0):
    """
    Processes all chunk files of a given type (e.g., "crops" or "patches") in parallel.
    
    Parameters:
      folder_path: Folder where the chunk files are stored.
      type_prefix: The prefix of the chunk files (e.g., "crops" for crops_*.pt).
      file_type: The singular form used for individual file names (e.g., "crop").
      checkpoint_filename: Name of the checkpoint file for this type.
      max_deviation_factor: Local threshold for discarding a slice (as in process_chunk_file).
      global_stats: Shared dictionary holding global statistics (via Manager).
      global_lock: Lock for synchronizing updates to global_stats.
      global_mean_deviation_factor: Threshold for discarding a slice whose mean deviates too much from the global mean.
    
    This function:
      - Reads the checkpoint file to skip already processed chunk files.
      - Lists all chunk files matching type_prefix.
      - Processes the unprocessed chunk files in parallel (each using process_chunk_file).
      - Updates the checkpoint file as each chunk file is processed.
    """
    checkpoint_path = os.path.join(folder_path, checkpoint_filename)
    processed_chunks = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            processed_chunks = {line.strip() for line in f if line.strip()}
        print(f"[{file_type}] Resuming: {len(processed_chunks)} chunk files already processed.")

    # List all chunk files for the given type.
    all_chunk_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.startswith(type_prefix + "_") and f.endswith(".pt")
    ])
    remaining_files = [f for f in all_chunk_files if f not in processed_chunks]
    print(f"[{file_type}] Total chunk files: {len(all_chunk_files)}. Remaining to process: {len(remaining_files)}.")

    with ProcessPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_chunk_file, file_path, file_type,
                            max_deviation_factor, global_stats, global_lock,
                            global_mean_deviation_factor): file_path
            for file_path in remaining_files
        }
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                processed_file = future.result()
                # Update the checkpoint file.
                with open(checkpoint_path, 'a') as cp:
                    cp.write(processed_file + "\n")
                print(f"[{file_type}] Processed and removed chunk file: {processed_file}")
            except Exception as exc:
                print(f"[{file_type}] Chunk file {file_path} generated an exception: {exc}")

def process_all_chunks_to_individual_files(folder_path,
                                           crops = False,
                                           patches = False,
                                           max_deviation_factor=3.0,
                                           global_mean_deviation_factor=3.0):
    """
    Main function to process both crops and patches chunk files.

    For each type, it calls process_chunk_files_parallel with all thresholds and
    with shared global statistics for filtering based on the global mean of accepted slices.
    After processing, each original chunk file is removed and individual files (without NaNs,
    without extreme local deviations, and whose means are close to the global mean) are saved.

    Parameters:
      folder_path: The folder containing the chunk files.
      max_deviation_factor: Local threshold for a slice’s maximum deviation.
      global_mean_deviation_factor: Global threshold for discarding a slice whose mean deviates too much.
    """
    # Create a Manager for shared global statistics.
    manager = multiprocessing.Manager()
    # Initialize the global statistics: we'll track the sum of slice means, sum of squares, and count.
    global_stats = manager.dict({"sum": 0.0, "square_sum": 0.0, "count": 0})
    global_lock = manager.Lock()

    # Process crops if crops=True
    if crops:
        process_chunk_files_parallel(
            folder_path=os.path.join(folder_path, "Crops"),
            type_prefix="crops",          # e.g., crops_1.pt, crops_2.pt, ...
            file_type="crop",             # individual files will be named like crop_1_0.pt, etc.
            checkpoint_filename="processed_crop_chunks.txt",
            max_deviation_factor=max_deviation_factor,
            global_stats=global_stats,
            global_lock=global_lock,
            global_mean_deviation_factor=global_mean_deviation_factor
        )

    # Process patches if patches=True
    if patches:
        process_chunk_files_parallel(
            folder_path=os.path.join(folder_path, "Patches"),
            type_prefix="patches",        # e.g., patches_1.pt, patches_2.pt, ...
            file_type="patch",            # individual files will be named like patch_1_0.pt, etc.
            checkpoint_filename="processed_patch_chunks.txt",
            max_deviation_factor=max_deviation_factor,
            global_stats=global_stats,
            global_lock=global_lock,
            global_mean_deviation_factor=global_mean_deviation_factor
        )

def check_and_remove_file(file_path):
    """
    Attempts to load the tensor file. If loading fails, attempts to remove the file.
    
    Args:
        file_path (str): Full path to the tensor file.
    
    Returns:
        str or None: Returns file_path if the file failed to load (and was removed), or
                     None if the file loaded successfully.
    """
    try:
        # Try loading the tensor file.
        _ = torch.load(file_path)
        return None
    except Exception as e:
        # If loading fails, try to remove the file.
        try:
            os.remove(file_path)
            print(f"Removed file: {file_path}")
        except Exception as remove_e:
            print(f"Failed to remove file {file_path}: {remove_e}")
        # Return the file path as a record of the failure.
        return file_path

def check_and_remove_tensor_files_parallel(root_dir, output_file, ext='.pt'):
    """
    Iterates over files in root_dir with the given extension, attempts to load them using torch.load
    in parallel using available CPUs. Files that fail to load are removed from disk and their names 
    are logged into a text file.
    
    Args:
        root_dir (str): Directory where the tensor files are stored.
        output_file (str): Path to the text file to save the list of files that failed to load.
        ext (str): File extension to check (default is '.pt').
    """
    # List all files with the specified extension.
    file_list = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(ext)]
    print(f"Found {len(file_list)} files with extension '{ext}' in {root_dir}.")
    
    failed_files = []
    
    # Use as many processes as available.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all file checks concurrently.
        futures = {executor.submit(check_and_remove_file, file_path): file_path for file_path in file_list}
        
        # Collect results as they complete.
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                failed_files.append(result)
    
    # Write the names of failed files to the output text file.
    with open(output_file, 'w') as f:
        for file_name in failed_files:
            f.write(file_name + "\n")
    
    print(f"Finished processing files. {len(failed_files)} file(s) failed to load and were removed. See '{output_file}' for details.")

