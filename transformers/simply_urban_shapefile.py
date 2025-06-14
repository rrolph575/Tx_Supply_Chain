
# Simplify urban file
# ## https://github.nrel.gov/cobika/DLR
# Specific definition is here:
# https://github.nrel.gov/ReEDS/ReEDS-2.0_Input_Processing/blob/01ffe2c3e5ebc187059dd2cff6ad52f85100e09b/zones/make_maps.py#L135-L170


import os
import geopandas as gpd



def get_urban(
    #url='https://www2.census.gov/geo/tiger/TIGER2024/UAC20/tl_2024_us_uac20.zip',
    ifile_urban_shp,
    simplify=20000,  # 2000 , 20000 is 12 miles
    crs='ESRI:102008',
    #projpath='Data/',
):
    """
    Args:
        url (str): URL for data.
            Parent site: https://www.census.gov/cgi-bin/geo/shapefiles/index.php
        simplify (float [m]): Amount by which to simplify the geometry
    """
    ### Download and unzip the file
    local_path = ifile_urban_shp

    ### Load and convert to CRS
    urban = (
        gpd.read_file(local_path)
        .to_crs(crs)
        .astype({'INTPTLAT20':float, 'INTPTLON20':float})
    )

    ### Subset to contiguous US
    latlon_bounds = {'latmin':20, 'latmax':50, 'lonmin':-130, 'lonmax':-60}

    urban = urban.loc[
        (urban.INTPTLAT20 >= latlon_bounds['latmin'])
        & (urban.INTPTLAT20 <= latlon_bounds['latmax'])
        & (urban.INTPTLON20 <= latlon_bounds['lonmax'])
        & (urban.INTPTLON20 >= latlon_bounds['lonmin'])
    ].copy()

    ### Simplify the geometry to speed up mapping
    urban['geometry'] = urban.simplify(simplify, preserve_topology=True).buffer(0.)
    #urban['geometry'] = urban.simplify(tolerance=0.1,simplify, preserve_topology=True).buffer(0.)

    return urban