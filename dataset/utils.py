from netCDF4 import Dataset
import math
import os
import multiprocessing
import functools
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    Process a single npz file: load its 'crop' and 'patches' arrays,
    reshape the patches (flattening the first three dimensions),
    and return the file name along with the processed arrays.
    """
    data = np.load(file_path)
    crops = data['crop']  # Expected shape: (n, 2048, 2048)
    patches = data['patches']  # Expected shape: (8,8,C,256,256) where C is 1 or 2
    patches_reshaped = patches.reshape(-1, patches.shape[-2], patches.shape[-1])
    return file_path, crops, patches_reshaped

def process_npz_files_parallel(folder_path, 
                               crops_filename='crops.pt', 
                               patches_filename='patches.pt', 
                               checkpoint_filename='checkpoint.txt'):
    """
    Process all .npz files in a folder in parallel while maintaining a checkpoint.
    
    - A checkpoint file (a text file) is used to record processed files.
    - If crops.pt and patches.pt exist, they are loaded to resume accumulation.
    - Files already processed (as recorded in the checkpoint) are skipped.
    - After processing each file, the new data is concatenated with the existing data,
      and the tensors are saved.
    """
    # Full paths for checkpoint and tensor files
    checkpoint_path = os.path.join(folder_path, checkpoint_filename)
    crops_file_path = os.path.join(folder_path, crops_filename)
    patches_file_path = os.path.join(folder_path, patches_filename)
    
    # Read checkpoint file if it exists
    processed_files = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            processed_files = set(line.strip() for line in f if line.strip())
        print(f"Found checkpoint: {len(processed_files)} files already processed.")
    
    # Initialize lists for accumulating crops and patches.
    # If the tensor files already exist, load them to resume accumulation.
    crops_list = []
    patches_list = []
    if os.path.exists(crops_file_path) and os.path.exists(patches_file_path):
        try:
            existing_crops = torch.load(crops_file_path).numpy()
            existing_patches = torch.load(patches_file_path).numpy()
            crops_list.append(existing_crops)
            patches_list.append(existing_patches)
            print(f"Loaded existing tensors: crops {existing_crops.shape}, patches {existing_patches.shape}")
        except Exception as e:
            print(f"Could not load existing tensors due to: {e}. Starting fresh.")
    
    # Get list of all .npz files in the folder and filter out the ones already processed.
    all_npz_files = sorted([os.path.join(folder_path, f) 
                            for f in os.listdir(folder_path) if f.endswith('.npz')])
    remaining_files = [f for f in all_npz_files if f not in processed_files]
    print(f"Total npz files: {len(all_npz_files)}; {len(remaining_files)} remaining to process.")
    
    # Process files in parallel using all available CPUs
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file_path): file_path for file_path in remaining_files}
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                file_processed, crops, patches = future.result()
            except Exception as exc:
                print(f"File {file_path} generated an exception: {exc}")
                continue
            
            # Append the new results
            crops_list.append(crops)
            patches_list.append(patches)
            
            # Update checkpoint: record this file as processed
            with open(checkpoint_path, 'a') as cp_file:
                cp_file.write(file_processed + "\n")
            
            # Concatenate all results along the first axis
            all_crops = np.concatenate(crops_list, axis=0)
            all_patches = np.concatenate(patches_list, axis=0)
            
            # Convert to torch tensors and save to disk
            crop_tensor = torch.from_numpy(all_crops)
            patch_tensor = torch.from_numpy(all_patches)
            torch.save(crop_tensor, crops_file_path)
            torch.save(patch_tensor, patches_file_path)
            
            print(f"Processed {file_processed}: crops shape {crop_tensor.shape}, patches shape {patch_tensor.shape}")
