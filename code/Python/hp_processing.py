import pyart
import glob
import sys
import numpy as np
import os
import gc
import logging
import argparse
import subprocess
from datetime import datetime
from csu_radartools import csu_fhc

# Mappings for CSU Summer, Winter, and Py-ART classifications to HydroPhase (hp)
csu_summer_to_hp = np.array([0, 1, 1, 2, 2, 4, 2, 3, 3, 3, 1])
csu_winter_to_hp = np.array([0, 2, 2, 2, 2, 4, 3, 1])
pyart_to_hp = np.array([0, 2, 2, 1, 3, 1, 2, 4, 4, 3])

def setup_logging(output_dir):
    log_file = os.path.join(output_dir, "hp_processing.log")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s %(message)s', filemode='w')
    return log_file

def unprocessed_files(files, output_dir):
    unprocessed = []
    for file in files:
        basename = os.path.basename(file)
        output_file = basename.replace('gucxprecipradarcmacppiS2.c1', 'gucxprecipradarcmacppihpS2.c1')
        output_path = os.path.join(output_dir, output_file)
        if not os.path.exists(output_path):
            unprocessed.append(file)
    return unprocessed


def read_radar(file, sweep=None):
    radar = pyart.io.read(file)
    if sweep is not None:
        radar = radar.extract_sweeps([sweep])
    return radar


def classify_summer(radar):
    logging.info("Running CSU Summer classification")
    dbz = radar.fields['corrected_reflectivity']['data']
    zdr = radar.fields['corrected_differential_reflectivity']['data']
    kdp = radar.fields['corrected_specific_diff_phase']['data']
    rhv = radar.fields['RHOHV']['data']
    rtemp = radar.fields['sounding_temperature']['data']
    scores = csu_fhc.csu_fhc_summer(dz=dbz, zdr=zdr, rho=rhv, kdp=kdp, use_temp=True, band='X', T=rtemp)
    return csu_summer_to_hp[scores]

def classify_winter(radar):
    logging.info("Running CSU Winter classification")
    dz = np.ma.masked_array(radar.fields['DBZ']['data'])
    zdr = np.ma.masked_array(radar.fields['ZDR']['data'])
    kd = np.ma.masked_array(radar.fields['PHIDP']['data'])
    rh = np.ma.masked_array(radar.fields['RHOHV']['data'])
    sn = np.ma.masked_array(radar.fields['signal_to_noise_ratio']['data'])
    rtemp = radar.fields['sounding_temperature']['data']
    azimuths = radar.azimuth['data']
    heights_km = radar.fields['height']['data'] / 1000
    hcawinter = csu_fhc.run_winter(
        dz=dz, zdr=zdr, kdp=kd, rho=rh, azimuths=azimuths, sn_thresh=-30,
        expected_ML=2.0, sn=sn, T=rtemp, heights=heights_km, nsect=36,
        scan_type=radar.scan_type, verbose=False, use_temp=True, band='S', return_scores=False
    )
    return csu_winter_to_hp[hcawinter]

def classify_pyart(radar):
    logging.info("Running Py-ART classification")
    radar.instrument_parameters['frequency'] = {'long_name': 'Radar frequency', 'units': 'Hz', 'data': [9.2e9]}
    hydromet_class = pyart.retrieve.hydroclass_semisupervised(
        radar,
        refl_field="corrected_reflectivity",
        zdr_field="corrected_differential_reflectivity",
        kdp_field="filtered_corrected_specific_diff_phase",
        rhv_field="RHOHV",
        temp_field="sounding_temperature",
    )
    return pyart_to_hp[hydromet_class['data']]

def add_classification_to_radar(classified_data, radar, field_name, description):
    logging.info(f"Adding field: {field_name} to radar obj")
    fill_value = -32768
    masked_data = np.ma.asanyarray(classified_data)
    masked_data.mask = masked_data == fill_value
    dz_field = 'DBZ' if 'winter' in field_name else 'corrected_reflectivity'
    if hasattr(radar.fields[dz_field]['data'], 'mask'):
        masked_data.mask = np.logical_or(masked_data.mask, radar.fields[dz_field]['data'].mask)
        fill_value = radar.fields[dz_field]['_FillValue']
    field_dict = {
        'data': masked_data,
        'units': '',
        'long_name': description,
        'standard_name': 'hydrometeor phase',
        '_FillValue': fill_value,
        "valid_min": 0,
        "valid_max": 4,
        "classification_description": "0: Unclassified, 1:Liquid, 2:Frozen, 3:High-Density Frozen, 4:Melting",
    }
    radar.add_field(field_name, field_dict, replace_existing=True)

def filter_fields(radar):
    fields = ['corrected_reflectivity', 'corrected_differential_reflectivity',
              'corrected_specific_diff_phase', 'RHOHV', 'sounding_temperature',
              'hp_semisupervised', 'hp_fhc_summer', 'hp_fhc_winter']
    radar.fields = {k: radar.fields[k] for k in fields if k in radar.fields}
    return radar


# taken from MAx's script for gridding for Squire
# Setup a Helper Function and Configure our Grid
def compute_number_of_points(extent, resolution):
    """
    Create a helper function to determine number of points
    """
    return int((extent[1] - extent[0])/resolution)


def grid_radar(radar,
               x_grid_limits=(-20_000.,20_000.),
               y_grid_limits=(-20_000.,20_000.),
               z_grid_limits = (500.,5_000.),
               grid_resolution = 250,
               
               ):
    """
    Grid the radar using some provided parameters
    """
    
    x_grid_points = compute_number_of_points(x_grid_limits, grid_resolution)
    y_grid_points = compute_number_of_points(y_grid_limits, grid_resolution)
    z_grid_points = compute_number_of_points(z_grid_limits, grid_resolution)
    
    grid = pyart.map.grid_from_radars(radar,
                                      grid_shape=(z_grid_points,
                                                  y_grid_points,
                                                  x_grid_points),
                                      grid_limits=(z_grid_limits,
                                                   y_grid_limits,
                                                   x_grid_limits),
                                      method='nearest'
                                     )
    return grid.to_xarray()

def subset_lowest_vertical_level(ds, additional_fields=["corrected_reflectivity"]):
    """
    Filter the dataset based on the lowest vertical level
    """
    hp_fields = [var for var in list(ds.variables) if "hp" in var] + additional_fields
    
    # Create a new 4-d height field
    ds["height_expanded"] = (ds.z * (ds[hp_fields[0]]/ds[hp_fields[0]])).fillna(5_000)
    
    # Find the minimum height index
    min_index = ds.height_expanded.argmin(dim='z',
                                          skipna=True)
    
    # Subset our hp fields based on this new index
    subset_ds = ds[hp_fields].isel(z=min_index)
    
    return subset_ds


def process_files(files, year, month, scheme, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    for file in files:
        logging.info(f"Processing file: {file} with scheme={scheme}")
        radar = read_radar(file)
        if scheme == 'summer':
            add_classification_to_radar(classify_summer(radar), radar, 'hp_fhc_summer', 'HydroPhase from CSU Summer')
        elif scheme == 'winter':
            add_classification_to_radar(classify_winter(radar), radar, 'hp_fhc_winter', 'HydroPhase from CSU Winter')
        add_classification_to_radar(classify_pyart(radar), radar, 'hp_semisupervised', 'HydroPhase from Py-ART')
        filter_fields(radar)

        ds = grid_radar(radar)
        out_ds = subset_lowest_vertical_level(ds)
        # Ge tonly available fields
        available_fields = list(out_ds.data_vars.keys())
        out_ds.attrs['field_names'] = ', '.join(available_fields)

        out_ds.attrs['radar_name'] = 'gucxprecipradar'
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        system_info = subprocess.check_output(['uname', '-n'], encoding='utf-8').strip()
        out_ds.attrs['history'] = f"created by Bhupendra Raut on {current_time} on: {system_info}"

        out_ds.attrs['attributions'] = "This data is collected by the ARM Climate Research facility. Radar system is operated by the radar engineering team radar@arm.gov and the data is processed by the precipitation radar products team."
        out_ds.attrs['vap_name'] = 'hp'
        out_ds.attrs['process_version'] = "HP v1.0"
        out_ds.attrs['known_issues'] = 'CMAC issues like, false phidp jumps, and some snow below melting layer, may affect classification. The Semisupervised method and fussy logic methods do not agree very well near melting layer.'
        out_ds.attrs['input_datastream'] = 'xprecipradarcmacppi'
        out_ds.attrs['developers'] = "Bhupendra Raut, ANL; Robert Jackson, ANL; Zachary Sherman, ANL; Maxwell Grover, ANL; Joseph OBrien, ANL"
        out_ds.attrs['datastream'] = "gucxprecipradarhpS2.c1"
        out_ds.attrs['platform_id'] = "xprecipradarhp"
        out_ds.attrs['dod_version'] = "xprecipradarhp-c1-1.0"
        out_ds.attrs['doi'] = "xxxxxx"
        out_ds.attrs['command_line'] = " ".join(sys.argv) # command that ran  the script.
        output_file = os.path.join(output_dir, os.path.basename(file).replace('gucxprecipradarcmacppiS2.c1', 'gucxprecipradarcmacppihpS2.c1'))
        out_ds.to_netcdf(output_file)
        logging.info(f"Saved file to {output_file}")
        del radar
        del ds
        del out_ds
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Process radar files for a given year, month, and classification scheme")
    parser.add_argument("year", type=str, help="Year of the data (YYYY)")
    parser.add_argument("month", type=str, help="Month of the data (MM)")
    parser.add_argument("--data_dir", type=str, default="/gpfs/wolf2/arm/atm124/world-shared/gucxprecipradarcmacS2.c1/ppi/",
                         help="data directory without year month")
    parser.add_argument("--output_dir", type=str, default= '/gpfs/wolf2/arm/atm124/proj-shared/gucxprecipradarhpS2.c1/',
                        help="outputdirectory without year month")
    parser.add_argument("--season", type=str, choices=['summer', 'winter'], required=True, help="CSU classification scheme to use (summer or winter)")
    parser.add_argument("--rerun", action='store_true', help="If set, process all files again (even if already processed)")

    args = parser.parse_args()
    year, month, season, rerun = args.year, args.month, args.season, args.rerun
    base_path = args.data_dir
    files = sorted(glob.glob(f"{base_path}{year}{month}/gucxprecipradarcmacppiS2.c1.*.nc"))
    output_dir = f'{args.output_dir}{year}{month}'
    files_to_process = files if rerun else unprocessed_files(files, output_dir)

    logging.info(f"Starting processing for year={year}, month={month}, season={season}")
    logging.info(f"Found {len(files_to_process)} files to process.")


    if files_to_process:
        process_files(files_to_process, year, month, season, output_dir)
    else:
        logging.info(f"No files to process. All files already processed or directory empty.")

if __name__ == "__main__":
    main()
