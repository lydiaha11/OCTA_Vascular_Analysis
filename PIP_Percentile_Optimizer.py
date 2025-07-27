#!/usr/bin/env python
# coding: utf-8

"""
MIP Percentile Optimizer
 
This script is designed to pick the best percentile to generate OCTA MIPs with. It requires manual baseline vessel masks and the raw OCTA data. I would reccomend running this code on a small random selection of the dataset which you have segmented, then using the outputed average percentile to run the majority of the scans with MIP_Generator.py

Arguments:
----------
--data_folder         : str (Required)
     Root directory containing subfolders with DICOM files. Each subfolder should correspond to one scan set 
     and a baseline mask.
--save_folder         : str (Required)
     Output directory to save generated MIP images (as .tif files).
--p_start             : float Default = 90)
     Percentile to start search. 
--p_end               : float Default = 100)
    Percentile to end search.
--step                : float Default = 1.0)
    Step size between percentiles. 
--plot                : flag (Optional)

Dependencies:
-------------
Install required packages via pip:
opencv-python pydicom XlsxWriter scikit-image torch matplotlib numpy scipy tqdm

"""

import os
import cv2
import argparse
import pydicom
import xlsxwriter
import skimage.io
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from skimage.filters import gaussian, frangi, meijering, threshold_otsu, sato
from skimage.filters.rank import entropy
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects
from skimage.transform import resize
from skimage.util import img_as_ubyte
from tqdm import tqdm

def data_manager(parent_folder):
    
    """
    Data Manager locates all raw OCT and OCTA files within the specified parent folder, along with their 
    corresponding baseline masks. It applies the `surface_cleaning` function to remove the surface from 
    both OCT and OCTA images.

    Inputs:
    - Path to the parent folder.
      Expected folder structure:

      Parent_Folder/
        └── scan_name/
              ├── scan_x_s.dcm or scan_x_S.dcm   (OCTA)
              ├── scan_x_d.dcm or scan_x_D.dcm   (OCT)
              └── mask_x.png                     (Baseline Mask)

    Outputs:
    - OCT tensor
    - OCTA tensor
    - Spacing metadata tensor
    - Filenames tensor
    - Surface indices tensor
    - Baseline masks tensor
    
    """
   
    all_oct_images = []
    all_octa_images = []
    all_spacings = []
    all_filenames = []
    all_surfaces = []
    all_base_masks = []

    subfolders = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder) # Find subfolders
                  if os.path.isdir(os.path.join(parent_folder, f))]

    if not subfolders:
        raise ValueError(f'No subfolders found in {parent_folder}')

    for folder in tqdm(subfolders, desc='Processing Dicom Pairs'):
        files = os.listdir(folder)
        dcm_files = [f for f in files if f.lower().endswith('.dcm') and not f.startswith('._')] #OCT, OCTA files
        mask_files = [f for f in files if f.lower().endswith('.png') and not f.startswith('._')] # Baseline masks

        # Look for _S.dcm, _s.dcm and _D.dcm, _d.dcm
        intensity_path = next((os.path.join(folder, f) for f in dcm_files if '_S.dcm' in f or '_s.dcm' in f), None)
        flow_path = next((os.path.join(folder, f) for f in dcm_files if '_D.dcm' in f or '_d.dcm' in f), None)
        mask_path = next((os.path.join(folder, f) for f in mask_files if '.png' in f), None)

        if not intensity_path or not flow_path:
            print(f'Missing pair in: {folder}. Found: {intensity_path}, {flow_path}')
            continue

        # Load dicoms
        intensity_dcm = pydicom.dcmread(intensity_path)
        flow_dcm = pydicom.dcmread(flow_path)

        oct_image = torch.tensor(intensity_dcm.pixel_array, dtype=torch.float32).unsqueeze(0) / 255.0
        octa_image = torch.tensor(flow_dcm.pixel_array, dtype=torch.float32).unsqueeze(0) / 255.0

        # Get spacing info
        spacing_x, spacing_y = map(float, intensity_dcm.PixelSpacing) if 'PixelSpacing' in intensity_dcm else (1.0, 1.0)
        spacing_z = float(intensity_dcm.SliceThickness) if 'SliceThickness' in intensity_dcm else 1.0
        spacings = (spacing_x, spacing_y, spacing_z)
        
        # Convert to numpy array
        oct_image_np = oct_image.squeeze().numpy()
        octa_image_np = octa_image.squeeze().numpy()
        
        # Remove everything above the surface
        oct_cleaned, octa_cleaned, surface = surface_cleaning(oct_image_np, octa_image_np, plot=False)
        
        # Convert back to tensor
        oct_image = torch.tensor(oct_cleaned, dtype=torch.float32).unsqueeze(0)
        octa_image = torch.tensor(octa_cleaned, dtype=torch.float32).unsqueeze(0)
        
        # Load baseline masks
        base_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        base_mask = torch.tensor(base_mask).unsqueeze(0)  # (1, H, W)
        
        all_oct_images.append(oct_image)
        all_octa_images.append(octa_image)
        all_spacings.append(spacings)
        all_surfaces.append(surface)
        all_base_masks.append(base_mask)
        all_filenames.append(os.path.basename(folder))
        
    if not all_oct_images:
        raise ValueError('No valid dicom pairs found.')

    oct_tensor = torch.stack(all_oct_images)       # (N, 1, H, W)
    octa_tensor = torch.stack(all_octa_images)     # (N, 1, H, W)
    spacing_tensor = torch.tensor(all_spacings)    # (N, 3)
    all_base_masks = torch.stack(all_base_masks)   # (N, 1, H, W)
    all_surfaces = torch.stack([torch.tensor(surface).unsqueeze(0) for surface in all_surfaces], dim=0) # (N 1, H, W)
    
    return oct_tensor, octa_tensor, spacing_tensor, all_filenames, all_surfaces, all_base_masks

def surface_cleaning(oct_image_np, octa_image_np, plot=False):
    
    """
    The surface_cleaning function detects the skin surface in OCT and OCTA images, sets all pixels above 
    this surface to zero (removing surface hairs and artifacts), and then flattens the image to correct 
    for skin curvature.
    
    Input: 
    - OCT image (numpy array)
    - OCTA image (numpy array)
    - Plot (bool)
    
    Output: 
    - OCT image with surface removed
    - OCTA image with surface removed
    - Surface index (array of surface locations per column)
    
    """
    
    # Transpose to match matlab dimensions
    oct_image_np = np.transpose(oct_image_np, (1, 2, 0)) # Ensure same shape for intensity image
    octa_image_np = np.transpose(octa_image_np, (1, 2, 0)) # (460, 1366, 120)
    
    Nz, Nx, Ny = oct_image_np.shape
    
    # Detect surface
    SurfInd = np.zeros((Nx, Ny), dtype=int)

    for yy in range(Ny):
        
        slice_img = oct_image_np[:, :, yy]

        # Smooth vertically to reduce noise
        smoothed = gaussian_filter1d(slice_img, sigma=2, axis=0)

        # Calculate vertical gradient (axis=0)
        grad = np.gradient(smoothed, axis=0)

        # Find the max gradient for each column
        surface = np.argmax(grad, axis=0)

        SurfInd[:, yy] = surface

    # Plot center slice
    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(oct_image_np[:, :, Ny // 2], cmap='gray')
        plt.plot(range(Nx), SurfInd[:, Ny // 2], 'r-', linewidth=2)
        plt.title('OCT (Center slice)')

        plt.subplot(1, 2, 2)
        plt.imshow(octa_image_np[:, :, Ny // 2], cmap='gray')
        plt.plot(range(Nx), SurfInd[:, Ny // 2], 'r-', linewidth=2)
        plt.title('OCTA (Center slice)')
        plt.show()

    # Surface smoothing (hair removal)
    surf_ind_f = np.round(ndimage.median_filter(SurfInd, size=(60, 30), mode='reflect'))
    surf_idx_comb = np.maximum(SurfInd, surf_ind_f)
    
    oct_cleaned = oct_image_np
    octa_cleaned = octa_image_np
    
    # Set all pixels above the surface to 0
    for yy in range(Ny):
        for xx in range(Nx):
            surface_level = surf_idx_comb[xx, yy]
            oct_cleaned[:surface_level, xx, yy] = 0
            octa_cleaned[:surface_level, xx, yy] = 0
    
    # Flatten intensity and flow images
    surf_idx_neg = -surf_idx_comb

    for yy in range(Ny):
        for xx in range(Nx):
            oct_cleaned[:, xx, yy] = np.roll(oct_cleaned[:, xx, yy], shift=surf_idx_neg[xx, yy], axis=0)
            octa_cleaned[:, xx, yy] = np.roll(octa_cleaned[:, xx, yy], shift=surf_idx_neg[xx, yy], axis=0)
            
    return oct_cleaned, octa_cleaned, SurfInd

def mip_selection(octa_tensor, all_base_masks, all_filenames, spacing_tensor, p_search=(90, 100), step_size=1, batch_size=8, plot=False, save_folder='mip_output'): 
    
    """
    The MIP selection function generates Maximum Intensity Projections (MIPs) across a specified range of 
    percentiles (typically 95–100 recommended). For each percentile, it calculates:

    - Entropy Score: Quantifies image noisiness.
    - Vessel Mask: Made using the Sato filter.
    - Percentile Bias: Biases the score towards a higher vessel scores
    - Quality score: Based on the intersection-over-union (IoU) with the baseline mask, the entropy score, 
    and the percential bias.

    The MIP with the highest quality score is selected for each image. The function also tracks the 
    average best-performing percentile across the dataset.

    Note: A step size smaller than 0.5 is not recommended for datasets with more than 5 images, due 
    to computational cost.

    Inputs:
    - OCTA tensor
    - Baseline masks tensor
    - Filenames tensor
    - Spacing metadata tensor
    - Percentile Search Range(tuple): specifying percentile range to search
    - Step size (float)
    - Batch size (int)
    - Plot (bool)
    - Save folder (str or Path)

    Outputs:
    - MIPs tensor (only best mip for each image)
    - Average best-performing percentile (float)
    """
    
    # Unpack percentile range
    z_start, z_end = p_search

    all_mips = []
    percentiles = list(np.arange(z_start, z_end+step_size, step_size))
    device = octa_tensor.device
    
    with torch.no_grad():
        for i in tqdm(range(0, len(octa_tensor), batch_size), desc='Generating MIPs'):
            batch_images = octa_tensor[i:i+batch_size].to(device)     
            batch_spacing = spacing_tensor[i:i+batch_size].to(device)
            batch_masks = all_base_masks[i:i+batch_size].to(device)
            
            for j in range(batch_images.shape[0]):
                
                image = batch_images[j].cpu().numpy().squeeze(0)
                spacing = batch_spacing[j].cpu().numpy()
                base_mask = batch_masks[j].cpu().numpy().squeeze(0)
                
                x_spacing, y_spacing = spacing[0], spacing[1]
                pixel_area = y_spacing * x_spacing

                best_mip = None
                best_percentile = None
                picked_percentiles = []
                entropy_scores = []
                IoU_scores = []

                for p in percentiles:
                    
                    # Generate MIP
                    mip = mip_slicer(image, p)
                    
                    # Calculate entropy
                    entropy_score = calculate_entropy(mip, local_size=5, plot=False)
                    
                    # Use frangi filter
                    mip_sato = calculate_vessel_score(mip, sigma=(1,7), plot=False)
                    
                    # Threshold frangi output to binary
                    mip_sato_bin = (mip_sato > threshold_otsu(mip_sato)).astype(np.float32)

                    # Resize to match base_mask if needed
                    if mip_sato_bin.shape != base_mask.shape:
                        mip_sato_bin = resize(mip_sato_bin, base_mask.shape, order=0, preserve_range=True, anti_aliasing=False)

                    # Calculate IoU
                    intersection = np.logical_and(mip_sato_bin, base_mask).sum()
                    union = np.logical_or(mip_sato_bin, base_mask).sum()
                    iou_score = intersection / union if union != 0 else 0.0

                    # Append scores
                    entropy_scores.append(entropy_score)
                    IoU_scores.append(iou_score)
                
                # Normalize (z scores)
                e_arr = np.array(entropy_scores)
                i_arr = np.array(IoU_scores)
                
                e_norm = (e_arr - e_arr.mean()) / e_arr.std()
                i_norm = (i_arr - i_arr.mean()) / i_arr.std()

                # Linear percentile bias
                percentiles = np.arange(z_start, z_end+step_size, step_size)
                percentile_bias = 1 - np.exp(-0.05 * (percentiles - 90))
                alpha = 1 # Bias towards smoothing
                beta = 2 # Bias towards higher percentiles
                quality_scores = i_norm - alpha * e_norm + beta * percentile_bias
                
                if plot: 
                    
                    # Plot graph of Scores and Percentile Bias
                    plt.plot(percentiles, i_norm, label='IoU Vessel')
                    plt.plot(percentiles, e_norm, label='Entropy')
                    plt.plot(percentiles, percentile_bias, label='Percentile Bias')
                    plt.plot(percentiles, quality_scores, label='Quality Score', linewidth=2)
                    plt.legend()
                    plt.title('Score Balancing Debug')
                    plt.xlabel('Percentile')
                    plt.show()
                                
                # Pick best mip
                best_idx = np.argmax(quality_scores)
                best_percentile = percentiles[best_idx]
                best_mip = mip_slicer(image, best_percentile)
                                       
                # Create output folder if needed
                filename_base = os.path.splitext(all_filenames[i + j])[0]
                sample_save_dir = os.path.join(save_folder, filename_base)
                os.makedirs(sample_save_dir, exist_ok=True)

                # Save MIP
                mip_path = os.path.join(sample_save_dir, f'mip_{filename_base}.tif')
                skimage.io.imsave(mip_path, best_mip.astype(np.uint8))
                all_mips.append(best_mip)
                
                # Save percentile
                picked_percentiles.append(best_percentile)

                if plot:
                    
                    height, width = best_mip.shape
                    real_size = [0, width * x_spacing, 0, height * y_spacing]
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Plot Best MIP
                    axes[0].imshow(best_mip, extent=real_size, cmap='gray')
                    axes[0].set_title(f'{best_percentile} Percentile MIP')
                    axes[0].axis('on')

                    # Plot histogram
                    axes[1].hist(best_mip.ravel(), bins=100, log=True, color='darkslateblue', edgecolor='black')
                    axes[1].set_title(f'Intensity Histogram - Sample {i+j}')
                    axes[1].set_xlabel('Pixel Intensity')
                    axes[1].set_ylabel('Log Frequency')
                    axes[1].grid(True)

                    plt.tight_layout()
                    plt.show()
                    
    average_percentile = np.mean(picked_percentiles)

    #all_mips = torch.tensor(np.stack(all_mips)).unsqueeze(1)
    return all_mips, average_percentile


def mip_slicer(image, percentile, crop_depth=(34, 90)): 
    
    """
    Generates a Maximum Intensity Projection (MIP) from a cropped OCTA image volume.

    The function crops the input volume along the depth axis using the provided crop range,
    computes the MIP using the given percentile, resizes it to match the original XY dimensions,
    and normalizes the result. 
    
    Input: 
    - OCTA image
    - Crop depth (tuple): specifying depth range to crop before MIP
    - Percentile (float)
    
    Output: 
    - MIP
    
    """
                     
    # Get image height and width                 
    img_height, img_width, img_depth = image.shape # (460, 1366, 120)
    
    # Validate and unpack crop depth
    z_start, z_end = crop_depth
    if z_start < 0 or z_end > img_height:
        raise ValueError('Crop depth is outside of image')
    
    # Initialize volume
    papillary_dermis_3d = np.zeros_like(image, dtype=np.uint8) 
    papillary_dermis_3d = image[z_start:z_end, :, :]  # Crop depth 150-400 microns
                                                
    # Generate percentile mip
    papillary_dermis_mip = np.percentile(papillary_dermis_3d, percentile, axis=0)                
    c, _ = papillary_dermis_mip.shape

    # Upsample rows to make image square
    papillary_dermis_mip = cv2.resize(papillary_dermis_mip, (c, c), interpolation=cv2.INTER_LINEAR)

    # Normalize image
    mip = cv2.normalize(papillary_dermis_mip, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            
    return mip

def calculate_entropy(mip, local_size=3, plot=False):
    
    """
    Calculates the entropy of a Maximum Intensity Projection (MIP) image using a local neighborhood defined 
    by `local_size`. The entropy is computed using a disk-shaped filter, and the final entropy score is 
    the mean value of the resulting entropy map.
    
    Input: 
    - MIP
    - local_size (int): Radius of the disk-shaped structuring element
    - Plot (bool)
    
    Output: 
    - entropy score (float)
    
    """
    
    # Rescale to 0-1 and clip
    mip_scaled = np.clip((mip / 255.0), 0, 1)    
    
    # Convert to 8bit
    mip_8bit = img_as_ubyte(mip_scaled)
    
    # Calculate entropy map
    entropy_map = entropy(mip_8bit, footprint=disk(local_size))
    
    # Calculate entropy score
    entropy_score = entropy_map.mean()
    
    if plot:
        plt.imshow(entropy_map, cmap='viridis')
        plt.title("Entropy Map")
        plt.colorbar()
        plt.show()
    
    return entropy_score

def calculate_vessel_score(mip, sigma=(1, 7), plot=False):
    
    """
    Calculates a vessel score from a Maximum Intensity Projection (MIP) image using the Sato filter.

    The function applies a Sato filter over the specified sigma range with a given scale step. 
    An Otsu threshold is then used to generate a binary vessel mask.

    Input: 
    - MIP
    - Sigma range (tuple): (min_sigma, max_sigma) for the Sato filter
    - Scale step (float): Step size for scales between sigma range
    - Plot (bool)
    
    Output: 
    - vessel_mask (Binary Vessel mask, same size as input MIP)
    
    """
    
    # Scale and clip to 0-1
    mip_scaled = mip.astype(np.float32)
    mip_scaled /= mip_scaled.max()  #
    mip_clipped = np.clip(mip_scaled.astype(np.float32), 0, 1)
    
    # Calculate vessel score
    vesselness = sato(mip_scaled, sigmas=sigma, black_ridges=False) 
    
    # Create vessel mask using otsu threshold method
    threshold = threshold_otsu(vesselness)
    vessel_mask = vesselness > threshold
    
    if plot:
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
        axes[0].imshow(mip_scaled, cmap='gray')
        axes[0].imshow(vesselness, cmap='magma', vmin=0, vmax=np.percentile(vesselness, 99), alpha=0.5)
        axes[0].set_title('Vessels overlayed on MIP')
        
        axes[1].imshow(mip, cmap='gray')
        axes[1].set_title('Sato Vessel Mask')
        
        plt.show()
    
    return vessel_mask


#### Running the code

import sys
import argparse

# Make sure to import or define these functions:
# from your_module import data_manager, mip_selection

def main():
    if len(sys.argv) == 1:
        print("""
====================================
      MIP Percentile Optimizer
====================================
This script helps choose the best percentile to generate MIPs for your OCTA data.
It calculates vessel scores and selects the optimal percentile.

Command-line mode example:
    python script.py --data_folder path/to/data --p_start 90 --p_end 100 --step 1 --plot
    
Parameters:
----------
--data_folder      : Directory containing subfolders of DICOM files and baseline masks.
--save_folder      : Directory to save MIP images (.tif).
--p_start          : Percentile to start search. 
--p_end            : Percentile to end search.
--step             : Step size between percentiles. 
--plot             : Add '--plot' to enable debug visualizations.

Now entering interactive mode...
        """)

        # Interactive prompts
        data_folder = input('Enter the path to the data folder: ').strip()
        p_start = int(input('Enter the start percentile (default 90): ') or 90)
        p_end = int(input('Enter the end percentile (default 100): ') or 100)
        step = float(input('Enter the step size between percentiles (default 1.0): ') or 1.0)
        plot_input = input('Plot intermediate results? (y/n, default n): ').strip().lower()
        plot = plot_input == 'y'
        save_folder = input('Enter folder to save output MIPs (default 'mip_output'): ').strip() or 'mip_output'

        run_pipeline(data_folder, p_start, p_end, step, plot, save_folder)
    
    else:
        # Command-line mode    
        parser = argparse.ArgumentParser(description='Optimize MIP percentile for OCTA scans')
        parser.add_argument('--data_folder', type=str, required=True,
                            help='Path to the folder containing subfolders of DICOMs and baseline masks.')
        parser.add_argument('--p_start', type=int, default=90,
                            help='Start of percentile range (default: 90)')
        parser.add_argument('--p_end', type=int, default=100,
                            help='End of percentile range (default: 100)')
        parser.add_argument('--step', type=float, default=1.0,
                            help='Step size between percentiles (default: 1)')
        parser.add_argument('--plot', action='store_true',
                            help='Whether to plot intermediate results')
        parser.add_argument('--save_folder', type=str, default='mip_output',
                            help='Folder to save output MIPs')

        args = parser.parse_args()
        run_pipeline(args.data_folder, args.p_start, args.p_end, args.step, args.plot, args.save_folder)


def run_pipeline(data_folder, p_start, p_end, step, plot, save_folder):
    print('\nLoading and processing data...')
    oct_tensor, octa_tensor, spacing_tensor, all_filenames, all_surfaces, all_base_masks = data_manager(data_folder)

    print('Running percentile optimization...')
    all_mips, average_percentile = mip_selection(
        octa_tensor, all_base_masks, all_filenames, spacing_tensor,
        p_search=(p_start, p_end),
        step_size=step,
        plot=plot,
        save_folder=save_folder
    )

    print(f'\nDone! Average optimal percentile across dataset: {average_percentile:.2f}')

if __name__ == "__main__":
    main()
