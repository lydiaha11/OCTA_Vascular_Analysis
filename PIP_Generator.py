#!/usr/bin/env python
# coding: utf-8

"""
MIP Generator - Convert DICOM OCTA files to Percentile Intensity Projection Images (MIPs).

This script reads OCTA (Optical Coherence Tomography Angiography) intensity and flow DICOM files from a 
structured folder tree, detects and flattens the skin surface, applies a depth-based crop, and generates 
Percentile MIP images.

Recommended: Use 'MIP Percentile Optimizer' beforehand to determine the optimal percentile value.

Command-line mode example:

python mip_generator.py --root "/path/to/raw_data" \
                        --save "/path/to/output_mips" \
                        --percentile 97.0 \
                        --depth 34 90 \
                        [--plot]

Arguments:
----------
--root         : str (Required)
    Root directory containing subfolders with DICOM files. Each subfolder should correspond to one scan set.
--save         : str (Required)
    Output directory to save generated MIP images (as .tif files).
--percentile   : float (Default = 97.0)
    Percentile value used to generate the intensity projection. Higher values preserve more prominent signals.
--depth        : two integers (Default = 34 90)
    Start and end depth indices for the 3D crop along the z-axis (depth dimension) before projecting.
--plot         : flag (Optional)

Dependencies:
-------------
Install required packages via pip:
pip install opencv-python pydicom scikit-image matplotlib numpy scipy tqdm
"""

import sys
import os
import cv2
import argparse
import pydicom

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from scipy import ndimage
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.morphology import binary_closing, disk, closing
from scipy.ndimage import gaussian_filter1d, median_filter
from tqdm import tqdm

def flatten_michelson_coaks(folder, file, percentile, save_path, crop_depth=(34, 90), plot=False):
    
    """
    Flatten Michelson Coaks converts OCTA files to maximum intensity projections (MIPs). It removes 
    everything above the surface of the skin before flattening the image. A depth crop is taken of the 
    dermis, and then a percentile of the crop is used to generate the MIP. The MIP is then upsampled 
    and normalized. 
    
    Input: 
    - Folder (with raw OCT and OCTA file)
    - Base File Name (str)
    - Percentile (float)
    - Crop Depth (tuple): specifying depth range to crop before MIP
    - Save path (str or path)
    - Plot (bool)
    
    Output: 
    - MIP
    
    """
    
    # Check if the folder is empty
    if not os.listdir(folder):  # If the folder is empty
        raise ValueError(f'No files found in: {folder}')

    # Possible file variations
    intensity_variants = [f'{file}_S.dcm', f'{file}_s.dcm']
    flow_variants = [f'{file}_D.dcm', f'{file}_d.dcm']

    # Find valid intensity and flow files
    intensity_path = next((os.path.join(folder, f) for f in intensity_variants if os.path.exists(os.path.join(folder, f))), None)
    flow_path = next((os.path.join(folder, f) for f in flow_variants if os.path.exists(os.path.join(folder, f))), None)

    # Check if both files exist
    if not intensity_path or not flow_path:
        print(f'Missing file in {folder} for {file}. Found: {intensity_path}, {flow_path}')
        return  # Skip processing this case

    # Read DICOM files
    IntGray = pydicom.dcmread(intensity_path).pixel_array
    FlowGray = pydicom.dcmread(flow_path).pixel_array

    # Ensure correct shape
    IntGray = np.squeeze(IntGray)
    FlowGray = np.squeeze(FlowGray)

    # Transpose to match Matlab dimensions
    IntGray = np.transpose(IntGray, (1, 2, 0))  # Ensure same shape for intensity image
    FlowGray = np.transpose(FlowGray, (1, 2, 0))  # Now (460, 1355, 120)

    Nz, Nx, Ny = IntGray.shape  # Get dimensions
    
    # Detect surface
    SurfInd = np.zeros((Nx, Ny), dtype=int)

    for yy in range(Ny):
        
        slice_img = IntGray[:, :, yy]

        # Smooth vertically to reduce noise
        smoothed = gaussian_filter1d(slice_img, sigma=5, axis=0)

        # Calculate vertical gradient (axis=0)
        grad = np.gradient(smoothed, axis=0)

        # Find the max gradient for each column
        surface = np.argmax(grad, axis=0)

        SurfInd[:, yy] = surface
    
    if plot: 
        
        # Surface Detection Visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(IntGray[:, :, yy], cmap='gray')
        plt.plot(range(Nx), SurfInd[:, yy], 'r-', linewidth=1)
        plt.title('Intensity')

        plt.subplot(1, 2, 2)
        plt.imshow(FlowGray[:, :, yy], cmap='gray')
        plt.plot(range(Nx), SurfInd[:, yy], 'r-', linewidth=1)
        plt.title('Flow')
        
    # Median filter to remove hair
    surf_ind_f = np.round(ndimage.median_filter(SurfInd, size=(60, 30), mode='reflect'))
    surf_ind_comb = np.maximum(SurfInd, surf_ind_f)  # Remove hair
        
    # Flatten intensity and flow images
    surf_index_neg = -surf_ind_comb
    
    for yy in range(Ny):
        for xx in range(Nx):
            IntGray[:, xx, yy] = np.roll(IntGray[:, xx, yy], shift=surf_index_neg[xx, yy], axis=0)
            FlowGray[:, xx, yy] = np.roll(FlowGray[:, xx, yy], shift=surf_index_neg[xx, yy], axis=0)
                        
    # Validate and unpack crop depth
    z_start, z_end = crop_depth
    if z_start < 0 or z_end > Nz:
        raise ValueError(f'Invalid crop range: {z_start}-{z_end} for data with depth {Nz}')

    # Generate and save flow MIP using percentile
    flow_vol_crop = FlowGray[z_start:z_end, :, :] 
    flow_percentile = np.percentile(flow_vol_crop, percentile, axis=0)  # Calculate percentile projection
    c, _ = flow_percentile.shape

    # Upsample rows to make image square
    flow_percentile_square = cv2.resize(flow_percentile, (c, c), interpolation=cv2.INTER_LINEAR)
    tif_file_name_m = os.path.join(save_path, f'{file}_MIP.tif')
    
    # Normalize image
    flow_percentile_square = cv2.normalize(flow_percentile_square, None, 
                                     alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Save MIP as 8bit image
    imageio.imwrite(tif_file_name_m, flow_percentile_square.astype(np.uint8))

    if plot: 
        
        # Display MIP image
        plt.figure()
        plt.imshow(flow_percentile_square, cmap='gray', aspect='equal')
        plt.axis('off')
        plt.show()

def batch_process_flatten_michelson_coaks(root_folder, save_path, percentile, crop_depth=(34, 90), plot=False):
    
    """
    Processes all subfolders in the root directory, running flatten_michelson_coaks for each sample found. 
    Passes output and parameters to flatten_michelson_coaks. 

    Input:
    - root_folder (path)
    - save_path
    - percentile (float)
    - crop_depth (tuple): 
    - plot (bool)
    
    Output: 
    - MIPs
    
    """
    
    os.makedirs(save_path, exist_ok=True)

    subfolders = [sf for sf in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, sf))]

    for subfolder in tqdm(subfolders, desc="Generating MIPs"):
        subfolder_path = os.path.join(root_folder, subfolder)
        
        for file in os.listdir(subfolder_path):
            if file.endswith('_S.dcm'):  # Look for base filename
                base_name = file[:-6]
                flatten_michelson_coaks(
                    folder=subfolder_path,
                    file=base_name,
                    percentile=percentile,
                    save_path=save_path,
                    crop_depth=crop_depth,
                    plot=plot
                )
                
                

if __name__ == "__main__":

    # Show usage description interactively if no arguments are passed
    if len(sys.argv) == 1:
        print("""
====================================
    MIP Generator - OCTA to MIP
====================================

This script processes OCTA (Optical Coherence Tomography Angiography) DICOM files and converts them to 
Percentile Maximum Intensity Projection (MIP) images.

Recommended: Use 'MIP Percentile Optimizer' first to determine optimal percentile for your dataset.

Command-line mode example:
--------------
python mip_generator.py --root "/path/to/raw_data" \\
                        --save "/path/to/output_mips" \\
                        --percentile 97.0 \\
                        --depth 34 90

Parameters:
-----------
--root         : Root directory containing subfolders of DICOM files.
--save         : Directory to save MIP images (.tif).
--percentile   : Float, percentile for MIP (Default: 97.0).
--depth        : Two integers for cropping depth (Default: 34 90).
--plot         : Add '--plot' to enable debug visualizations.

Now entering interactive mode...
        """)
        # Prompt user input
        root = input('Enter path to root folder with DICOM subfolders: ').strip()
        save = input('Enter path to save output MIP images: ').strip()
        percentile = float(input('Enter percentile for MIP (e.g., 97.0): ').strip())
        depth_start = int(input('Enter crop depth START (e.g., 34): ').strip())
        depth_end = int(input('Enter crop depth END (e.g., 90): ').strip())
        plot = input('Enable plotting? (y/n): ').strip().lower() == 'y'

        # Call main function with user input
        batch_process_flatten_michelson_coaks(
            root_folder=root,
            save_path=save,
            percentile=percentile,
            crop_depth=(depth_start, depth_end),
            plot=plot
        )

    else:
        
        # Command-line mode
        parser = argparse.ArgumentParser(description='Generate MIPs from OCTA DICOM folders.')
        parser.add_argument('--root', type=str, required=True, help='Root folder containing subfolders of DICOM files')
        parser.add_argument('--save', type=str, required=True, help='Output folder to save MIP images')
        parser.add_argument('--percentile', type=float, default=97.0, help='Percentile to use for projection')
        parser.add_argument('--depth', nargs=2, type=int, default=[34, 90], help='Crop depth as two integers (start end)')
        parser.add_argument('--plot', action='store_true', help='Enable plotting for debugging')

        args = parser.parse_args()

        batch_process_flatten_michelson_coaks(
            root_folder=args.root,
            save_path=args.save,
            percentile=args.percentile,
            crop_depth=tuple(args.depth),
            plot=args.plot
        )
        print(f'\nDone!')