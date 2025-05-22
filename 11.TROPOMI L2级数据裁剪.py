import xarray as xr
import numpy as np
import os
from datetime import datetime
import netCDF4 as nc

def clip_tropomi_to_us_region(input_file, output_file):
    """
    Clip TROPOMI NetCDF files to US region and keep only specified variables.
    
    Parameters:
        input_file (str): Path to input NetCDF file
        output_file (str): Path to output clipped NetCDF file
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Define US region boundaries
    US_LON_MIN, US_LON_MAX = -172.4543, -66.9548  # Includes Alaska and mainland
    US_LAT_MIN, US_LAT_MAX = 18.9100, 71.3866     # Wide latitude range for US
    
    print(f"Processing file: {input_file}")
    
    try:
        # Open the NetCDF file with netCDF4
        with nc.Dataset(input_file, 'r') as ncfile:
            # Define the paths to variables we want to extract
            var_info = {
                'nitrogendioxide_total_column': {
                    'path': ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'],
                    'name': 'nitrogendioxide_total_column'
                },
                'nitrogendioxide_stratospheric_column': {
                    'path': ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'],
                    'name': 'nitrogendioxide_stratospheric_column'
                },
                'nitrogendioxide_tropospheric_column': {
                    'path': ['PRODUCT'],
                    'name': 'nitrogendioxide_tropospheric_column'
                },
                'qa_value': {
                    'path': ['PRODUCT'],
                    'name': 'qa_value'
                },
                'latitude': {
                    'path': ['PRODUCT'],
                    'name': 'latitude'
                },
                'longitude': {
                    'path': ['PRODUCT'],
                    'name': 'longitude'
                }
            }
            
            # First extract latitude and longitude for clipping
            try:
                lat_group = ncfile['PRODUCT']
                latitude = lat_group['latitude'][:]
                longitude = lat_group['longitude'][:]
                
                # Print dimensions of latitude to help debugging
                print(f"Latitude dimensions: {lat_group['latitude'].dimensions}")
            except Exception as e:
                print(f"Failed to access latitude/longitude: {str(e)}")
                print(f"Available groups in root: {list(ncfile.groups.keys())}")
                if 'PRODUCT' in ncfile.groups:
                    print(f"Available variables in PRODUCT: {list(ncfile['PRODUCT'].variables.keys())}")
                return False
            
            # Determine the dimensions used in the file
            lat_dims = lat_group['latitude'].dimensions
            
            # Check if we have the expected dimensions
            if len(lat_dims) != 3 or 'time' not in lat_dims:
                print(f"Unexpected dimensions for latitude: {lat_dims}")
                return False
                
            # Create region mask for US
            lat_mask = (latitude >= US_LAT_MIN) & (latitude <= US_LAT_MAX)
            lon_mask = (longitude >= US_LON_MIN) & (longitude <= US_LON_MAX)
            region_mask = lat_mask & lon_mask
            
            # Find valid coordinates based on the dimensions
            # Assuming lat_dims is ('time', 'scanline', 'ground_pixel')
            time_idx, scanline_idx, pixel_idx = np.where(region_mask)
            
            if len(time_idx) == 0:
                print(f"No valid data within US region for {input_file}")
                return False
            
            # Get unique indices for each dimension
            unique_time = np.unique(time_idx)
            unique_scanline = np.unique(scanline_idx)
            unique_pixel = np.unique(pixel_idx)
            
            print(f"Found {len(unique_time)} time, {len(unique_scanline)} scanlines and {len(unique_pixel)} pixels in US region")
            
            # Create a new NetCDF file for the clipped data
            with nc.Dataset(output_file, 'w') as out_nc:
                # First collect all dimensions from variables we need
                all_dimensions = set()
                dimension_sizes = {}
                
                # Identify all dimensions needed across all variables
                for var_name, info in var_info.items():
                    try:
                        # Navigate to the correct group
                        current_group = ncfile
                        for group_name in info['path']:
                            current_group = current_group[group_name]
                        
                        orig_var = current_group[info['name']]
                        dimensions = orig_var.dimensions
                        
                        print(f"Variable {var_name} has dimensions: {dimensions}")
                        
                        # Add to our set of dimensions
                        for dim in dimensions:
                            all_dimensions.add(dim)
                            
                            # Get original dimension size
                            if dim in ncfile.dimensions:
                                dimension_sizes[dim] = len(ncfile.dimensions[dim])
                            elif 'PRODUCT' in ncfile.groups and dim in ncfile['PRODUCT'].dimensions:
                                dimension_sizes[dim] = len(ncfile['PRODUCT'].dimensions[dim])
                    except Exception as e:
                        print(f"Error examining dimensions for {var_name}: {str(e)}")
                
                # Create dimensions in the root of the output file
                for dim_name in all_dimensions:
                    # Special case for dimensions we're clipping
                    if dim_name == 'time':
                        out_nc.createDimension(dim_name, len(unique_time))
                    elif dim_name == 'scanline':
                        out_nc.createDimension(dim_name, len(unique_scanline))
                    elif dim_name == 'ground_pixel':
                        out_nc.createDimension(dim_name, len(unique_pixel))
                    elif dim_name in dimension_sizes:
                        out_nc.createDimension(dim_name, dimension_sizes[dim_name])
                    else:
                        print(f"Warning: Could not determine size for dimension {dim_name}")
                
                # Copy global attributes
                for attr_name in ncfile.ncattrs():
                    out_nc.setncattr(attr_name, ncfile.getncattr(attr_name))
                
                # Add custom attributes
                out_nc.setncattr('preprocessing', 'US region clip')
                out_nc.setncattr('lon_range', f"{US_LON_MIN}~{US_LON_MAX}")
                out_nc.setncattr('lat_range', f"{US_LAT_MIN}~{US_LAT_MAX}")
                out_nc.setncattr('clip_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                
                # Create PRODUCT group in output file
                product_group = out_nc.createGroup('PRODUCT')
                
                # Create SUPPORT_DATA group in PRODUCT
                support_data_group = product_group.createGroup('SUPPORT_DATA')
                
                # Create DETAILED_RESULTS group in SUPPORT_DATA
                detailed_results_group = support_data_group.createGroup('DETAILED_RESULTS')
                
                # Copy time coordinate variable if it exists in source
                if 'time' in ncfile.variables:
                    time_var = ncfile.variables['time']
                    out_time_var = out_nc.createVariable('time', time_var.dtype, ('time',))
                    out_time_var[:] = time_var[unique_time]
                    
                    # Copy attributes
                    for attr_name in time_var.ncattrs():
                        if attr_name != '_FillValue':
                            out_time_var.setncattr(attr_name, time_var.getncattr(attr_name))
                
                # Extract and store variables
                for var_name, info in var_info.items():
                    try:
                        # Navigate to the correct group
                        current_group = ncfile
                        for group_name in info['path']:
                            current_group = current_group[group_name]
                        
                        # Get original variable
                        orig_var = current_group[info['name']]
                        
                        # Determine target group for this variable
                        target_group = out_nc
                        for group_name in info['path']:
                            if group_name == 'PRODUCT':
                                target_group = product_group
                            elif group_name == 'SUPPORT_DATA':
                                target_group = support_data_group
                            elif group_name == 'DETAILED_RESULTS':
                                target_group = detailed_results_group
                        
                        # Create the variable in the output file
                        out_var = target_group.createVariable(
                            info['name'], 
                            orig_var.dtype, 
                            orig_var.dimensions,
                            zlib=True,
                            complevel=5
                        )
                        
                        # Copy data with appropriate slicing based on dimensions
                        if orig_var.dimensions == ('time', 'scanline', 'ground_pixel'):
                            # Handle 3D arrays
                            sliced_data = orig_var[unique_time][:, unique_scanline][:, :, unique_pixel]
                            out_var[:] = sliced_data
                        elif 'time' in orig_var.dimensions and 'scanline' in orig_var.dimensions:
                            # Handle 2D arrays with time and scanline
                            out_var[:] = orig_var[unique_time][:, unique_scanline]
                        elif 'time' in orig_var.dimensions:
                            # Handle arrays with time dimension
                            out_var[:] = orig_var[unique_time]
                        else:
                            # Copy as-is for other arrays
                            out_var[:] = orig_var[:]
                        
                        # Copy variable attributes
                        for attr_name in orig_var.ncattrs():
                            if attr_name != '_FillValue':  # Skip _FillValue as it's set during createVariable
                                out_var.setncattr(attr_name, orig_var.getncattr(attr_name))
                            
                        print(f"Successfully clipped and saved variable: {var_name}")
                            
                    except Exception as e:
                        print(f"Error processing variable {var_name}: {str(e)}")
                        # Continue with other variables
                
            print(f"Successfully saved clipped file to: {output_file}")
            return True
            
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_all_files():
    """Process all NetCDF files in the source directory"""
    source_dir = r"M:\TROPOMI_S5P\NO2\L2"
    
    # Get all NetCDF files in the source directory
    nc_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.nc') and not file.endswith('_clip.nc'):  # Skip already clipped files
                nc_files.append(os.path.join(root, file))
    
    print(f"Found {len(nc_files)} NetCDF files to process")
    
    # Process each file
    success_count = 0
    for i, input_file in enumerate(nc_files):
        # Create temp file name and final file name
        original_path = os.path.dirname(input_file)
        filename = os.path.basename(input_file)
        basename, ext = os.path.splitext(filename)
        
        # Create a temporary file first to avoid issues if processing fails
        temp_output = os.path.join(original_path, f"{basename}_temp{ext}")
        final_output = os.path.join(original_path, f"{basename}_clip{ext}")
        
        print(f"[{i+1}/{len(nc_files)}] Processing: {filename}")
        
        # Process the file to a temporary location first
        if clip_tropomi_to_us_region(input_file, temp_output):
            try:
                # Remove original file
                os.remove(input_file)
                # Rename temporary file to final name
                os.rename(temp_output, final_output)
                success_count += 1
                print(f"Original file replaced with: {os.path.basename(final_output)}")
            except Exception as e:
                print(f"Error replacing original file: {str(e)}")
                # If renaming fails, at least try to clean up the temporary file
                try:
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                except:
                    pass
    
    print(f"Processing complete. Successfully clipped and replaced {success_count} out of {len(nc_files)} files.")

if __name__ == "__main__":
    process_all_files()