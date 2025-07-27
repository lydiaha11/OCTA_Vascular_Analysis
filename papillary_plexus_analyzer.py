#!/usr/bin/env python
# coding: utf-8

"""
Papillary Plexus Analyzer - Convert DICOM OCTA files to papillary plexus MIPs

This script reads OCTA (Optical Coherence Tomography Angiography) intensity and flow DICOM files from a 
structured folder tree, detects and flattens the skin surface, applies a depth-based crop based on a segmentation model, and generates Percentile MIP images. The MIPs capillary loops are analyzed. 

Dependencies:
-------------
Install required packages via pip:
pip install opencv-python pydicom scikit-image matplotlib numpy scipy tqdm pandas torch xlsxwriter
"""

import os
import math
import cv2
import pydicom
import argparse
import xlsxwriter
import skimage.io
import torch

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch.nn.functional as F

from scipy import ndimage, signal
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from skimage.filters import gaussian, frangi, meijering, threshold_otsu, sato
from skimage.feature import canny
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, remove_small_objects
from skimage.transform import resize
from tqdm import tqdm

class UNet128(torch.nn.Module):
    
    def __init__(self, out_channels=2, dropout=0.2):
        super().__init__()
        
        def double_convolution(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout2d(p=dropout),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout2d(p=dropout),
            )

        self.down1 = double_convolution(1, 32)
        self.pool1 = torch.nn.MaxPool2d(2)

        self.down2 = double_convolution(32, 64)
        self.pool2 = torch.nn.MaxPool2d(2)

        self.down3 = double_convolution(64, 128)
        self.pool3 = torch.nn.MaxPool2d(2)

        self.bottleneck = double_convolution(128, 256)

        self.up3 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = double_convolution(256, 128)

        self.up2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv2 = double_convolution(128, 64)

        self.up1 = torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv1 = double_convolution(64, 32)

        self.final_conv = torch.nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        l1 = self.down1(x)
        l2 = self.down2(self.pool1(l1))
        l3 = self.down3(self.pool2(l2))

        bottleneck = self.bottleneck(self.pool3(l3))

        up3 = self.up3(bottleneck)
        l3_resized = F.interpolate(l3, size=up3.shape[2:], mode='bilinear', align_corners=False)
        up3 = torch.cat([up3, l3_resized], dim=1)
        up3 = self.upconv3(up3)

        up2 = self.up2(up3)
        l2_resized = F.interpolate(l2, size=up2.shape[2:], mode='bilinear', align_corners=False)
        up2 = torch.cat([up2, l2_resized], dim=1)
        up2 = self.upconv2(up2)

        up1 = self.up1(up2)
        l1_resized = F.interpolate(l1, size=up1.shape[2:], mode='bilinear', align_corners=False)
        up1 = torch.cat([up1, l1_resized], dim=1)
        up1 = self.upconv1(up1)

        out = self.final_conv(F.interpolate(up1, size=x.shape[2:], mode='bilinear', align_corners=False))
        return out


def surface_cleaning(oct_image_np, octa_image_np, plot=False):
    
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
        plt.title('Intensity (Center slice)')

        plt.subplot(1, 2, 2)
        plt.imshow(octa_image_np[:, :, Ny // 2], cmap='gray')
        plt.plot(range(Nx), SurfInd[:, Ny // 2], 'r-', linewidth=2)
        plt.title('Flow (Center slice)')
        plt.show()

    # Surface smoothing
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
    
    return oct_cleaned, octa_cleaned, SurfInd


def data_manager(parent_folder):
    
    all_oct_images = []
    all_octa_images = []
    all_spacings = []
    all_filenames = []
    all_surfaces = []

    subfolders = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder)
                  if os.path.isdir(os.path.join(parent_folder, f))]

    if not subfolders:
        raise ValueError(f'No subfolders found in {parent_folder}')

    for folder in tqdm(subfolders, desc='Processing Dicom Pairs'):
        files = os.listdir(folder)
        dcm_files = [f for f in files if f.lower().endswith('.dcm') and not f.startswith('._')]

        # Look for _S.dcm, _s.dcm and _D.dcm, _d.dcm
        intensity_path = next((os.path.join(folder, f) for f in dcm_files if '_S.dcm' in f or '_s.dcm' in f), None)
        flow_path = next((os.path.join(folder, f) for f in dcm_files if '_D.dcm' in f or '_d.dcm' in f), None)

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

        all_oct_images.append(oct_image)
        all_octa_images.append(octa_image)
        all_spacings.append(spacings)
        all_surfaces.append(surface)
        all_filenames.append(str(os.path.basename(folder)))
        
    if not all_oct_images:
        raise ValueError('No valid dicom pairs found.')

    oct_tensor = torch.stack(all_oct_images)       # (N, 1, H, W)
    octa_tensor = torch.stack(all_octa_images)     # (N, 1, H, W)
    spacing_tensor = torch.tensor(all_spacings)    # (N, 3)
    all_surfaces = torch.stack([torch.tensor(surface).unsqueeze(0) for surface in all_surfaces], dim=0) # (N 1, H, W)
    
    return oct_tensor, octa_tensor, spacing_tensor, all_filenames, all_surfaces


def predict_and_save_masks(model, oct_tensor, all_filenames, save_folder, batch_size=8, plot=False):
    """Processes all images in input_folder in batches and saves masks in output_folder."""
    
    all_masks = []
    
    # Process in batches
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(oct_tensor), batch_size), desc='Predicting Masks'):
            
            batch = oct_tensor[i:i+batch_size].to(device)  # Get batch and move to device
            
            # Get center slice
            center_slice_idx = batch.shape[4] // 2 
            center_slices = batch[:, :, :, :, center_slice_idx]  # (N, 1, 460, 1366)

            # Predict mask
            preds = model(center_slices)
            preds = torch.sigmoid(preds)
            preds = F.interpolate(preds, size=(460, 1366), mode='bilinear', align_corners=False)
            preds = (preds > 0.5).float()
            
            preds = preds.cpu().numpy()  # Move back to CPU
            
            # Process each image in the batch
            for j in range(preds.shape[0]):
                pred_mask = preds[j, 0]  # Get the predicted mask  (460, 1355)
                pred_mask = 1 - pred_mask  # Invert mask
                
                if plot:
                    img_rgb = center_slices[j, 0].cpu().numpy()  # Get the OCT image from the batch
                    img_rgb = np.repeat(img_rgb[None, :, :], 3, axis=0)  # Convert to RGB
                    img_rgb = np.transpose(img_rgb, (1, 2, 0))  # (H, W, 3)

                    # Normalize image for display
                    img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)

                    # Create red mask
                    red_mask = np.zeros_like(img_rgb)
                    red_mask[..., 0] = pred_mask  # Apply mask to red channel

                    alpha = 0.3
                    overlay = (1 - alpha) * img_rgb + alpha * red_mask

                    # Plot original slice and overlayed mask slice 
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                    axes[0].imshow(img_rgb)
                    axes[0].set_title('Original OCT Image')
                    axes[0].axis('off')

                    axes[1].imshow(overlay)
                    axes[1].set_title('OCT with Mask Overlay')
                    axes[1].axis('off')

                    plt.tight_layout()
                    plt.show()
                
                # Get file name
                filename_base = os.path.splitext(all_filenames[i + j])[0]
                sample_save_dir = os.path.join(save_folder, filename_base)

                # Create directory
                os.makedirs(sample_save_dir, exist_ok=True)

                # Save mask
                mask_path = os.path.join(sample_save_dir, f'mask_{filename_base}.tif')
                skimage.io.imsave(mask_path, (pred_mask * 255).astype(np.uint8))
                #print(f'Saved mask: {mask_path}')
                
                all_masks.append(pred_mask)
        
    all_masks = torch.tensor(all_masks).unsqueeze(1)
        
    return all_masks


def find_epidermis_boundaries(model, oct_tensor, all_masks, spacing_tensor, batch_size=8, plot=False):
    
    epidermis_boundaries = []
    all_thicknesses = []
    offset = 0
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(oct_tensor), batch_size), desc='Finding Epidermis Boundaries'):
            
            # Get batches and move to device
            batch_masks = all_masks[i:i+batch_size].to(device)
            batch_images = oct_tensor[i:i+batch_size].to(device)
            batch_spacing = spacing_tensor[i:i+batch_size].to(device)
                        
            for j in range(batch_masks.shape[0]):
                
                mask = batch_masks[j, 0].cpu().numpy()  # (H, W)
                spacing = batch_spacing[j].cpu().numpy()
                
                H, W = mask.shape
                x_spacing, y_spacing = spacing[1], spacing[2] # Get pixel spacing
                pixel_area = y_spacing * x_spacing  # Area of one pixel in mm²
                
                upper = np.zeros(W, dtype=np.int32)
                lower = np.zeros(W, dtype=np.int32)
                        
                valid = mask > 0 # papillary plexus
                
                # Get upper and lower boundaries of mask
                upper = np.where(valid.any(axis=0), valid.argmax(axis=0), -1)
                reversed_valid = valid[::-1]
                last_indices_from_bottom = reversed_valid.argmax(axis=0)
                lower = np.where(valid.any(axis=0), H - 1 - last_indices_from_bottom, -1)

                # Interpolate missing (-1) values
                def interpolate_missing(values):
                    idx = np.where(values != -1)[0]
                    if idx.size == 0:
                        return values  # No missing values
                    return np.interp(np.arange(len(values)), idx, values[idx])

                upper = interpolate_missing(upper)
                lower = interpolate_missing(lower)
                                
                # Calculate original thickness of upper and lower boundaries
                thickness = lower - upper + offset
                orig_upper = upper
                orig_lower = lower
                
                mm_thickness = np.mean(thickness * y_spacing) 
                #print(f'Epidermis Thickness: {mm_thickness :2f}')

                # Shift boundaries
                upper = (upper + 5).astype(np.int32)
                lower = (lower + offset).astype(np.int32)
                
                # Smooth boundaries
                lower = signal.savgol_filter(lower, 10, 1) # window, order of fitted polynomial
                
                # Append
                all_thicknesses.append(mm_thickness)

                upper_tensor = torch.tensor(upper, dtype=torch.int32)
                lower_tensor = torch.tensor(lower, dtype=torch.int32)
                boundary_tensor = torch.stack([upper_tensor, lower_tensor], dim=0)  # (2, W)
                epidermis_boundaries.append(boundary_tensor)

                if plot:
                    # Get center slice
                    center_slice_idx = batch_images.shape[4] // 2 
                    center_slices_img = batch_images[:, :, :, :, center_slice_idx]  # (N, 1, 460, 1366)
                    img = center_slices_img[j, 0].cpu().numpy()
                    
                    # Convert to mm
                    height, width = img.shape
                    real_size = [0, width * x_spacing, height * y_spacing, 0]  # (x_min, x_max, y_min, y_max)
                    
                    # Scale X values
                    x_vals = np.arange(img.shape[1]) * x_spacing  # scaled x-axis

                    # Scale Y values
                    upper_scaled = upper * y_spacing
                    lower_scaled = lower * y_spacing
                    orig_upper_scaled = orig_upper * y_spacing
                    orig_lower_scaled = orig_lower * y_spacing
                    
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Plot shifted boundaries
                    axes[0].imshow(img[::-1, :], cmap='gray', extent=real_size, origin='lower')
                    axes[0].plot(x_vals, upper_scaled, 'r-', label='Upper boundary')
                    axes[0].plot(x_vals, lower_scaled, 'b-', label='Lower boundary')
                    axes[0].set_title('Smoothed Epidermis Boundaries')
                    axes[0].set_xlabel('(mm)')
                    axes[0].set_ylabel('(mm)')
                    axes[0].set_aspect(4)
                    axes[0].axis('on')
                    axes[0].legend()

                    # Plot original boundaries
                    axes[1].imshow(img[::-1, :], cmap='gray', extent=real_size, origin='lower')
                    axes[1].plot(x_vals, orig_upper_scaled, 'r-', label='Original upper boundary')
                    axes[1].plot(x_vals, orig_lower_scaled, 'b-', label='Original lower boundary')
                    axes[1].set_title('Original Epidermis Boundaries')
                    axes[1].set_xlabel('(mm)')
                    axes[1].set_ylabel('(mm)')
                    axes[1].set_aspect(4)
                    axes[1].axis('on')
                    axes[1].legend()
                    
                    plt.tight_layout()
                    plt.show()
                        
    epidermis_boundaries = torch.stack(epidermis_boundaries, dim=0)  # (N, 2, W)
    all_thicknesses = torch.tensor(all_thicknesses).unsqueeze(1)     # (N, 1)
    
    return epidermis_boundaries, all_thicknesses 


def mip_selection(model, octa_tensor, epidermis_boundaries, all_filenames, spacing_tensor, save_folder, batch_size=8, plot=False): 
    
    all_mips = []

    percentiles = list(np.arange(100, 98, -1))
    model.eval()
    device = octa_tensor.device
    
    with torch.no_grad():
        for i in tqdm(range(0, len(octa_tensor), batch_size), desc='Generating MIPs'):
            
            batch_images = octa_tensor[i:i+batch_size].to(device)     
            batch_bounds = epidermis_boundaries[i:i+batch_size].to(device)
            batch_spacing = spacing_tensor[i:i+batch_size].to(device)
            
            for j in range(batch_images.shape[0]):
                
                image = batch_images[j].cpu().numpy().squeeze(0)
                bounds = batch_bounds[j]
                spacing = batch_spacing[j].cpu().numpy()
                
                x_spacing, y_spacing = spacing[0], spacing[1] # Get pixel spacing
                pixel_area = y_spacing * x_spacing  # Area of one pixel in mm²
                
                best_mip = None
                best_percentile = None
                min_small_objects = float('inf')
                min_huge_objects = float('inf')
                mip_stats = []

                for p in percentiles:
                    
                    mip = mip_slicer(image, bounds, p)
                    
                    binary_mip = mip > 25  # All values >25 are objects
                    labeled = label(binary_mip > 25)
                    regions = regionprops(labeled)
                    sizes = [1000 * r.area * pixel_area for r in regions] # convert from pixel to mm to um
                    
                    small_objects = sum(1 for s in sizes if s <= 5)
                    large_objects = sum(1 for s in sizes if 5 < s <= 15)
                    large_objects_size = sum(s for s in sizes if 5 < s <= 15)
                    huge_objects = sum(1 for s in sizes if s > 15)
                    
                    mip_stats.append((p, mip, small_objects, large_objects, large_objects_size, huge_objects))

                # Count max number of large objects
                max_large_size = max(stat[4] for stat in mip_stats if stat[0] == 100)

                for p, mip, small_count, large_count, large_objects_size, huge_count in mip_stats:
                    if (
                        small_count < min_small_objects and 
                        large_count >= 0.65 * max_large_size and 
                        huge_count <= min_huge_objects
                    ):
                        
                        best_mip = mip
                        best_percentile = p
                        min_small_objects = small_count
                        min_huge_objects = huge_count

                if best_mip is None:
                    best_mip = mip_stats[0][1]  # fallback to 100 percentile MIP
                    best_percentile = mip_stats[0][0]
                    
                # Get file name  
                filename = str(all_filenames[i + j])
                filename_base = os.path.splitext(filename)[0]
                sample_save_dir = os.path.join(save_folder, filename_base)

                # Save mip
                mip_path = os.path.join(sample_save_dir, f'mip_{filename_base}.tif')
                skimage.io.imsave(mip_path, best_mip.astype(np.uint16)) # Save mask as 16bit image
                    
                all_mips.append(best_mip)
                
                if plot:
                    # Calculate object sizes for the best mip
                    labeled_final = label(best_mip > 25)
                    regions_final = regionprops(labeled_final)
                    sizes_final = [1000 * r.area * pixel_area for r in regions_final]
                    large_sizes_final = [s for s in sizes_final if 0 < s <= 50]
                    
                    # Normalize the sizes
                    size_array = np.array(sizes_final)
                    size_large_array = np.array(large_sizes_final)
                    
                    norm_sizes = (size_array - size_array.min()) / (size_array.ptp() + 1e-8)  
                    norm_sizes_larges = (size_large_array - size_large_array.min()) / (size_large_array.ptp() + 1e-8)

                    # Normalize sizes
                    size_map = {}
                    size_large_map = {}
                    
                    for region, norm_size in zip(regions_final, norm_sizes):
                        size_map[region.label] = norm_size
                        
                    for region, norm_sizes_large in zip(regions_final, norm_sizes_larges):
                        size_large_map[region.label] = norm_sizes_large
                    
                    # Fill missing labels with default values
                    for region in regions_final:
                        if region.label not in size_map:
                            size_map[region.label] = 0  # Default value
                        if region.label not in size_large_map:
                            size_large_map[region.label] = 0  # Default value
                        
                    # Plot an image colored by size
                    colored_image = np.zeros((*labeled_final.shape, 3), dtype=np.float32)
                    colored_image_large = np.zeros((*labeled_final.shape, 3), dtype=np.float32)
                    cmap = plt.cm.viridis

                    for lbl in size_map:
                        mask = labeled_final == lbl
                        color_size = cmap(size_map[lbl])[:3]  # Color based on size_map
                        color_large = cmap(size_large_map[lbl])[:3]  # Color based on large_map

                        for c in range(3):  # RGB channels
                            colored_image[..., c][mask] = color_size[c]
                            colored_image_large[..., c][mask] = color_large[c]

                    # Plot images
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    height, width = best_mip.shape
                    real_size = [0, width * x_spacing, 0, height * y_spacing]  # (x_min, x_max, y_min, y_max)
                    
                    im0 = axes[0].imshow(colored_image, extent=real_size, aspect='auto')
                    axes[0].set_title(f'All Objects Colored by Size')
                    axes[0].set_xlabel('(mm)')
                    axes[0].set_ylabel('(mm)')
                    axes[0].axis("on")
                    #plt.colorbar(im0, ax=axes[0])
                    
                    im1 = axes[1].imshow(colored_image_large, extent=real_size, aspect='auto')
                    axes[1].set_title(f"Large Objects Colored by Size")
                    axes[1].set_xlabel('(mm)')
                    axes[1].set_ylabel('(mm)')
                    axes[1].axis("on")
                    #plt.colorbar(im1, ax=axes[0])

                    axes[2].imshow(best_mip, cmap='gray',  extent=real_size, aspect='auto')
                    axes[2].set_title(f'Optimal Percentile ({best_percentile}) MIP - Sample {i+j}')
                    axes[2].set_xlabel('(mm)')
                    axes[2].set_ylabel('(mm)')
                    axes[2].axis('on')

                    plt.tight_layout()
                    plt.show()

                    # Plot histograms 
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    axes[0].hist(sizes_final, bins=30, log=True, range=(0,  max(sizes_final)), color='gray', edgecolor='black')
                    axes[0].set_title(f'Object Size Histogram - Sample {i+j}')
                    axes[0].set_xlabel('Object Area (µm)')
                    axes[0].set_ylabel('Log Count')
                    axes[0].grid(True)

                    axes[1].hist(best_mip.ravel(), bins=100, log=True, color='steelblue', edgecolor='black')
                    axes[1].set_title(f'Intensity Histogram - Sample {i+j}')
                    axes[1].set_xlabel('Pixel Intensity')
                    axes[1].set_ylabel('Log Frequency')
                    axes[1].grid(True)
                    
                    plt.tight_layout()
                    plt.show()
                    
    all_mips = torch.tensor(all_mips).unsqueeze(1)

    return all_mips


def mip_slicer(image, bounds, percentile): 
                     
    # Get boundaries and image height and width                 
    img_height, img_width, img_depth = image.shape # (460, 120, 1366)

    upper_y = bounds[0].cpu().numpy()
    lower_y = bounds[1].cpu().numpy() # Lower line y-coordinate

    # Initialize volume
    papillary_dermis_3d = np.zeros_like(image, dtype=np.uint8)
                
    # Masking each slice
    for z in range(img_depth):           
        for x in range(img_width):
            y1 = int(upper_y[x])
            y2 = int(lower_y[x])
                        
            if y1 < y2 and y2 < img_height:
                papillary_dermis_3d[y1:y2, x, z] = image[y1:y2, x, z]
                                                
    # Generate percentile mip
    papillary_dermis_mip = np.percentile(papillary_dermis_3d, percentile, axis=0)                
    c, _ = papillary_dermis_mip.shape

    # Upsample rows to make image square
    papillary_dermis_mip = cv2.resize(papillary_dermis_mip, (c, c), interpolation=cv2.INTER_LINEAR)

    # Normalize image
    mip = cv2.normalize(papillary_dermis_mip, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            
    return mip


def linear_artifact_remover(image):
    
    # Fourier transform
    image_fft = np.fft.fftshift(np.fft.fft2(image))
    
    rows, cols = image.shape
    
    # Vertical artifact mask parameters
    mask_height = 8
    center_offset = 15
    dist_edge = 300
    
    # Define boundaries
    top_row = rows//2 + mask_height//2
    bottom_row = rows//2 - mask_height//2

    l_mask_right_edge = cols//2 + center_offset
    
    r_mask_left_edge = cols//2 + center_offset
    r_mask_right_edge = cols - dist_edge
    
    # Generate vertical mask
    mask = np.ones(image.shape)
    mask[bottom_row:top_row, dist_edge:l_mask_right_edge] = 0
    mask[bottom_row:top_row, r_mask_left_edge:r_mask_right_edge] = 0
    
    # Horizontal mask parameters
    mask_width = 10
    center_offset = 10
    dist_edge = 750
    
    # Define boundaries
    u_top_row = dist_edge
    u_bottom_row = rows//2 + center_offset
    
    b_top_row = rows//2 - center_offset
    b_bottom_row = rows - dist_edge
    
    mask_left_edge = cols//2 - mask_width//2
    mask_right_edge = cols//2 + mask_width//2
    
    # Generate horizontal mask
    mask[u_bottom_row:u_top_row, mask_left_edge:mask_right_edge] = 0
    mask[b_bottom_row:b_top_row, mask_left_edge:mask_right_edge] = 0
    
    # Apply mask
    clean_image_fft = image_fft * mask
    
    # Inverse fourier transform 
    clean_mip = np.fft.ifft2(np.fft.ifftshift(clean_image_fft)).real
    clean_mip = clean_mip > 0.5
    
    """
    # Plot Image and its Fourier Transform
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Image with Artifacts')
    ax[0].set_axis_off()
    ax[1].imshow(np.log(abs(image_fft)), cmap='gray')
    ax[1].set_title('Logarithmic Fourier Transform of the Image')
    ax[1].set_axis_off()
    plt.show()
    
    # Plot cleaned image and its masked fourier transform
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(clean_mip, cmap='gray')
    ax[0].set_title('Cleaned Image')
    ax[0].set_axis_off()
    ax[1].imshow(np.log(abs(clean_image_fft)), cmap='gray')
    ax[1].set_title('Logarithmic Fourier Transform of the Image')
    ax[1].set_axis_off()
    plt.show()
    """
    
    return clean_mip


def identify_capillary_loops(mip, spacing, min_size=50, intensity_threshold = 50, 
                             dot_color='red', dot_alpha=0.5, circularity_threshold=0.7, plot=False):
    
    # Keep a copy of the original image for plotting
    original_image = mip.copy()
    
    # Image spacing information
    x_spacing, y_spacing = spacing[0], spacing[1] # Get pixel spacing
    pixel_area = y_spacing * x_spacing  # Area of one pixel in mm²
    
    # Converting pixels to mm
    height, width = mip.shape
    real_size = [0, width * x_spacing, 0, height * y_spacing]
    
    image = cv2.normalize(mip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image[image < intensity_threshold] = 0
    
    # Convert to binary (0 and 1)   
    binary_image = np.where(image > 0, 1, 0)
    
    # Remove linear artifacts
    cleaned_image = linear_artifact_remover(binary_image)
    
    # Remove small objects
    cleaned_image = remove_small_objects(cleaned_image.astype(bool), min_size=min_size)
    
    # Label connected components
    labeled_image = label(cleaned_image)

    # Create an empty image to store only circular objects
    circular_objects = np.zeros_like(labeled_image)

    # Get properties of labeled regions
    regions = regionprops(labeled_image)

    # Collect coordinates of dots that are sufficiently circular
    dot_coords = []
    for region in regions:
        if region.perimeter == 0:  # Avoid division by zero
            continue
        # Calculate circularity
        circularity = 4 * math.pi * region.area / (region.perimeter ** 2)
        if circularity >= circularity_threshold:
            dot_coords.append(region.centroid)
            # Keep the circular object in the circular_objects image
            circular_objects[labeled_image == region.label] = 1
        # Else: do nothing, effectively removing the non-circular region

    # Convert circular_objects to binary image
    circular_objects = circular_objects > 0
    
    # Plotting
    if plot:
                
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot the image with only circular objects
        axes[0].imshow(circular_objects, cmap='gray')
        axes[0].set_title(f'Identified Dots (min size = {min_size}, circularity ≥ {circularity_threshold})')
        axes[0].axis('off')

        # plot the cleaned image on top of the original image
        axes[1].imshow(original_image, cmap='gray', extent=real_size)
        #axes[1].imshow(circular_objects, cmap='viridis', alpha=dot_alpha, extent=real_size)
        axes[1].set_ylabel('(mm)')
        axes[1].set_xlabel('(mm)')
        axes[1].set_title('MIP with identified dots')
        plt.show()
    
    return dot_coords, circular_objects


def calculate_capillary_metrics(model, all_mips, octa_tensor, all_filenames, spacing_tensor, save_folder, batch_size=8, plot=False):
    
    capillary_metrics = []
    
    plot_mip = plot

    model.eval()
    device = octa_tensor.device
    
    with torch.no_grad():
        for i in tqdm(range(0, len(all_mips), batch_size), desc='Calculating Capillary Metrics'):
            
            batch_mips = all_mips[i:i+batch_size].to(device)
            batch_spacing = spacing_tensor[i:i+batch_size].to(device)

            for j in range(batch_mips.shape[0]):
                
                mip = batch_mips[j].squeeze(0).cpu().numpy()
                
                spacing = batch_spacing[j].squeeze(0).cpu().numpy()
                
                dot_coords, circular_objects = identify_capillary_loops(mip, 
                                                                        spacing, 
                                                                        min_size=30, 
                                                                        intensity_threshold =50,
                                                                        dot_color='red', 
                                                                        dot_alpha=0.5, 
                                                                        circularity_threshold=0.45, 
                                                                        plot=plot_mip)
                
                # Total dots
                total_dots = len(dot_coords)
                #print(f'Total dots: {total_dots}')

                # Calculate the combined dot area
                total_dot_area = np.count_nonzero(circular_objects)
                #print(f'Combined dot area: {total_dot_area :.2f}')
                
                # Average dot size
                avg_dot_size = total_dot_area / total_dots if total_dots > 0 else 0
                avg_diameter = 2 * np.sqrt(avg_dot_size / np.pi) if total_dots > 0 else 0
                #print(f'Average dot diameter: {avg_diameter :.2f}')

                # Calculate dot density
                total_area = mip.shape[0] * mip.shape[1]
                dot_density = total_dot_area / total_area
                #print(f'Dot area ratio: {dot_density :.2f}') 
                
                dot_mask = circular_objects * 255
                
                # Get file name
                filename_base = os.path.splitext(all_filenames[i + j])[0]
                sample_save_dir = os.path.join(save_folder, filename_base)

                # Create directory if it doesn't exist
                os.makedirs(sample_save_dir, exist_ok=True)

                # Save dots mask
                dots_path = os.path.join(sample_save_dir, f'dots_{filename_base}.tif')
                skimage.io.imsave(dots_path, dot_mask.astype(np.uint8))

                
                capillary_metrics.append((total_dots, total_dot_area, avg_diameter, dot_density))
                
    capillary_metrics = torch.tensor(capillary_metrics).unsqueeze(0)
    
    return capillary_metrics


def generate_excel_file(all_filenames, all_thicknesses, capillary_metrics, save_path='capillary_stats.xlsx'):
    
    print('Generating Excel file...')

    # Convert tensors to NumPy
    thicknesses = all_thicknesses.view(-1).cpu().numpy().tolist()
    metrics = capillary_metrics.view(-1, 4).cpu().numpy()
    
    # Ensure filenames are strings
    filenames = [str(f) for f in all_filenames]

    # Define column names
    metric_cols = ['Epidermis_Thickness', 'Total_Dots', 'Total_Dot_Area', 'Average_Dot_Diameter', 'Dot_Density']
    columns = ['Filename'] + metric_cols

    # Assemble rows
    rows = []
    for i in range(len(filenames)):
        row = [filenames[i], thicknesses[i]] + metrics[i].tolist()
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # Save to Excel
    df.to_excel(save_path, index=False)
    print(f'Excel file saved to: {save_path}')


def run_capillary_pipeline(model_path, input_folder, save_folder, excel_path, batch_size=8, device_str='cpu', plot=False):

    # Set device
    device = torch.device("cpu")

    # Load model
    model = UNet128()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Data loading
    oct_tensor, octa_tensor, spacing_tensor, all_filenames, all_surfaces = data_manager(input_folder)

    # Predict masks
    all_masks = predict_and_save_masks(model, oct_tensor, all_filenames, save_folder, batch_size, plot=plot)

    # Find epidermis boundaries
    epidermis_boundaries, all_thicknesses = find_epidermis_boundaries(
            model, oct_tensor, all_masks, spacing_tensor, batch_size, plot=plot)

    # Generate MIPs
    all_mips = mip_selection(model=model, 
                             octa_tensor=octa_tensor, 
                             epidermis_boundaries=epidermis_boundaries, 
                             all_filenames=all_filenames, 
                             spacing_tensor=spacing_tensor, 
                             save_folder=save_folder, 
                             batch_size=batch_size, 
                             plot=plot)

    
    # Calculate metrics
    capillary_metrics = calculate_capillary_metrics(model, 
                                                    all_mips,
                                                    octa_tensor,
                                                    all_filenames, 
                                                    spacing_tensor,
                                                    save_folder,
                                                    batch_size,
                                                    plot=plot)

    # Save results
    generate_excel_file(all_filenames, all_thicknesses, capillary_metrics, save_path=excel_path)


import argparse
import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive Mode
        print("""
=========================================
       Papillary Plexus Analyzer
=========================================

This script processes OCTA DICOM images and calculates capillary metrics 
from the papillary plexus.

Command-line mode example:
--------------------
python run_pipeline.py --model_path "model.pt" \\
                       --input_folder "data/input_dicoms" \\
                       --save_folder "results/" \\
                       --excel_path "results/metrics.xlsx" \\
                       --batch_size 8 \\
                       --device_str cpu

Parameters:
-----------
--model_path   : Path to the .pt file for the trained segmentation model
--input_folder : Folder with DICOM subfolders
--save_folder  : Where to save intermediate and output files
--excel_path   : Output Excel file path
--batch_size   : Inference batch size (default: 8)
--device_str   : "cpu" or "cuda"
--plot         : Add '--plot' to enable debug visualizations, not reccomended with more than 4 OCTA recordings

Now entering interactive mode...
        """)

        model_path = input('Enter model path (.pt): ').strip()
        input_folder = input('Enter path to input DICOM folder: ').strip()
        save_folder = input('Enter path to save outputs: ').strip()
        excel_path = input('Enter path to save Excel file: ').strip()
        batch_size = input('Enter batch size (default 8): ').strip()
        batch_size = int(batch_size) if batch_size else 8
        device_str = input('Device to use ("cpu" or "cuda", default "cpu"): ').strip() or "cpu"
        plot = input('Enable plotting? (y/n): ').strip().lower() == 'y'

        run_capillary_pipeline(
            model_path=model_path,
            input_folder=input_folder,
            save_folder=save_folder,
            excel_path=excel_path,
            batch_size=batch_size,
            device_str=device_str,
            plot=plot
        )

    else:
        # Command-Line Mode
        parser = argparse.ArgumentParser(description="Run Papillary Plexus Analyzer")

        parser.add_argument('--model_path', type=str, required=True)
        parser.add_argument('--input_folder', type=str, required=True)
        parser.add_argument('--save_folder', type=str, required=True)
        parser.add_argument('--excel_path', type=str, required=True)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--device_str', type=str, default='cpu')
        parser.add_argument('--plot', action='store_true')

        args = parser.parse_args()

        run_capillary_pipeline(
            model_path=args.model_path,
            input_folder=args.input_folder,
            save_folder=args.save_folder,
            excel_path=args.excel_path,
            batch_size=args.batch_size,
            device_str=args.device_str,
            plot=args.plot
        )
