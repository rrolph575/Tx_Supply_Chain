import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import site
import geopandas as gpd

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

dlrpath = os.path.realpath(os.path.join(os.path.dirname(__file__),'..'))
site.addsitedir(dlrpath)
import dlr

reedspath = os.path.expanduser('../ReEDS-2.0/')

hifldpath = '/projects/atm/DLR/hifld_data_conus_only_20240926'
if not os.path.exists(hifldpath):
    hifldpath = os.path.join(dlrpath, 'data', 'hifld_data_conus_only_20240926')


FOOT_PER_METER = 3.28084
DEFAULTS = {
    'wind_speed': 0.61, # m/s
    'wind_direction': 90, # degrees from parallel to conductor
    'air_temperature': 40 + dlr.C2K, # K
    'conductor_temperature': 75 + dlr.C2K, # K
    'ghi': 1000, # W/m^2
    'emissivity': 0.8,
    'absorptivity': 0.8,
}

def determine_quarterly_season(day_of_year):
    if day_of_year in range(81, 173):
        return 'spring'
    elif day_of_year in range(173, 265):
        return 'summer'
    elif day_of_year in range(265, 356):
        return 'fall'
    else:
        return 'winter'
    
def determine_biannual_season(month):
    if month in [1, 2, 3, 4, 11, 12]:
        return 'winter'
    else:
        return 'summer'
    
def determine_biannual_season_long_summer(month):
    if month in [1, 2, 12]:
        return 'winter'
    else:
        return 'summer'

def clean_hifld_voltages(dfhifld):
    updated_hifld = dfhifld.copy()

    volt_class_rep_voltage_map = {
        'UNDER 100': 69,
        '100-161': 115,
        '220-287': 230,
        '345': 345,
        '500': 500,
        '735 AND ABOVE': 765,
        'NOT AVAILABLE': np.nan,
        'DC': np.nan
    }
    def get_closest_representative_voltage(voltage):
        if np.isnan(voltage):
            return np.nan
        else:
            return min(volt_class_rep_voltage_map.values(), key=lambda x:abs(x - voltage))

    # Replace any negative voltages with an inferred value based on the volt class, or NaN
    # if volt class doesn't specify a kV range
    updated_hifld.loc[updated_hifld.VOLTAGE < 0, "VOLTAGE"] = (
        updated_hifld.loc[updated_hifld.VOLTAGE < 0, "VOLT_CLASS"].map(volt_class_rep_voltage_map)
    )
    # Add a column representing the representative voltage closest to the line's real voltage
    updated_hifld["rep_volt"] = updated_hifld["VOLTAGE"].apply(lambda x: get_closest_representative_voltage(x))
    
    return updated_hifld
    
def get_lines_and_dlrs(path='.', weather_year=2012):
    dflines = clean_hifld_voltages(gpd.read_file(hifldpath))
    dlrs = pd.read_hdf(os.path.join(path,"dlrs_all_lines.h5"), key=str(weather_year))

    # Filter out any lines for which we couldn't compute line rating data
    dflines = (
        dflines.loc[dflines.ID.isin(dlrs.columns)]
        .copy()
        .reset_index(drop=True)
    )

    # Add ISO information for each line
    hierarchy = pd.read_csv(
        os.path.join(reedspath,'inputs','hierarchy.csv')
    ).drop_duplicates().rename(columns={'ba':'r'}).set_index('r')
    hierarchy = hierarchy.loc[hierarchy.country.str.lower()=='usa'].copy()

    dfba = gpd.read_file(os.path.join(reedspath,'inputs','shapefiles','US_PCA')).set_index('rb')
    for col in hierarchy:
        dfba[col] = dfba.index.map(hierarchy[col])

    dfmap = {}
    for col in ['interconnect','transreg','transgrp','st','country']:
        dfmap[col] = dfba.copy()
        dfmap[col].geometry = dfmap[col].buffer(0.)
        dfmap[col] = dfmap[col].dissolve(col)

    regions = dfmap['transreg'].index
    _overlaps = {}
    for region in regions:
        _overlaps[region] = dflines.intersection(dfmap['transreg'].loc[region, 'geometry']).length
    overlaps = pd.concat(_overlaps, axis=1, names='transreg')
    main_region = (
        overlaps.stack().rename('overlap')
        .sort_values().reset_index().drop_duplicates('level_0', keep='last')
        .set_index('level_0')
    )
    main_region.loc[main_region.overlap == 0, 'transreg'] = '_none'
    dflines = dflines.merge(main_region[["transreg"]], left_index=True, right_index=True)

    return dflines, dlrs

def compute_weather_year_mean(dfs_by_year, weather_years):
    dfs_sum = pd.DataFrame()
    for weather_year in weather_years:
        year_adj = 2024 - weather_year
        
        weather_year_df = dfs_by_year[weather_year]
        weather_year_df["hour_of_year"] = (
            weather_year_df.index + pd.offsets.DateOffset(years=year_adj)
        )
    
        if not dfs_sum.empty:
            dfs_sum = pd.concat([dfs_sum, weather_year_df])
        else:
            dfs_sum = weather_year_df

        dfs_sum = dfs_sum.groupby("hour_of_year").sum().reset_index()

    dfs_mean = (
        dfs_sum.set_index(dfs_sum["hour_of_year"])
        .drop(columns="hour_of_year")
        .div(len(weather_years))
    )

    return dfs_mean.loc[~((dfs_mean.index.month == 2) & (dfs_mean.index.day == 29))]


def assign_line_to_region(dflines, dfregions, label='region'):
    """
    """
    ## If a line overlaps with at least one region, assign it to the most overlapping region
    regions = dfregions.index
    _overlaps = {}
    for region in regions:
        _overlaps[region] = dflines.intersection(dfregions[region]).length
    overlaps = pd.concat(_overlaps, axis=1, names=label)
    main_region = (
        overlaps.stack().rename('overlap')
        .sort_values().reset_index().drop_duplicates('ID', keep='last')
        .set_index('ID')
    )
    main_region.loc[main_region.overlap == 0, label] = '_none'
    _dflines = dflines.merge(main_region[[label]], left_index=True, right_index=True)
    ## Also record lines that cross between regions
    _dflines[f'multi_{label}'] = overlaps.replace(0,np.nan).apply(
        lambda row: ','.join(row.dropna().index.tolist()),
        axis=1,
    )
    ## For unmapped lines, map them to the closest region
    ids_unmapped = _dflines.loc[_dflines[label] == '_none'].index
    for ID in ids_unmapped:
        _dflines.loc[ID,label] = (
            dfregions.distance(_dflines.loc[ID, 'geometry']).nsmallest(1).index[0])

    return _dflines


def calculate_zlr(
    dfhifld=None,
    regional_air_temp=True,
    regional_wind=False,
    regional_irradiance=False,
    regional_conductor_temp=False,
    regional_emissivity=False,
    regional_absorptivity=False,
    aggfunc='representative',
    minimal=False,
):
    """Calculate seasonal line ratings using regional_assumptions.csv
    Inputs
    ------
    regional_{}:
        If True, use value from regional_assumptions.csv in all seasons and regions
        If False, use default value from helpers.DEFAULTS in all seasons and regions
        If numeric, use the provided numeric value in all seasons and regions
        If dictionary with ('summer', 'winter') as keys, use those in all regions
    regional_air_temp: °Celsius
    regional_conductor_temp: °Celsius
    regional_wind: m/s

    Outputs
    -------
    pd.DataFrame: Copy of dfhifld with ZLR_summer and ZLR_winter added
    """
    regional_assumptions = pd.read_csv(
        os.path.join(dlrpath, 'data', 'regional_assumptions.csv')
    )
    ## Fill missing winter values with summer values
    fill_columns = [
        'ambient_temp_{}_celsius',
        'windspeed_{}_fps',
        'solar_radiation_{}_watts_per_square_meter',
    ]
    for col in fill_columns:
        regional_assumptions.loc[
            regional_assumptions[col.format('winter')].isnull(),
            col.format('winter')
        ] = regional_assumptions.loc[
            regional_assumptions[col.format('winter')].isnull(),
            col.format('summer')
        ]
    ## Convert feet per second to meters per second
    for col in ['windspeed_summer_fps','windspeed_winter_fps']:
        regional_assumptions[col.replace('fps','mps')] = regional_assumptions[col] / FOOT_PER_METER
    ## Aggregate
    if aggfunc == 'representative':
        regional_assumptions = regional_assumptions.loc[
            regional_assumptions.representative == 1
        ].set_index('transreg')
    else:
        regional_assumptions = regional_assumptions.groupby('transreg').agg(aggfunc)
    assert (regional_assumptions.index.value_counts() == 1).all()

    ### Assign lines to ISOs (called transreg in ReEDS)
    dfzones = gpd.read_file(
        os.path.join(dlrpath,'data','reeds_maps.gpkg'), layer='transreg',
    ).set_index('transreg').geometry
    ## If a line overlaps with at least one zone, assign it to the most overlapping zone
    regions = dfzones.index
    _overlaps = {}
    for region in regions:
        _overlaps[region] = dfhifld.intersection(dfzones[region]).length
    overlaps = pd.concat(_overlaps, axis=1, names='transreg')
    main_region = (
        overlaps.stack().rename('overlap')
        .sort_values().reset_index().drop_duplicates('ID', keep='last')
        .set_index('ID')
    )
    main_region.loc[main_region.overlap == 0, 'transreg'] = '_none'
    _dfhifld = dfhifld.merge(main_region[["transreg"]], left_index=True, right_index=True)
    ## For unmapped lines, map them to the closest region
    ids_unmapped = _dfhifld.loc[_dfhifld.transreg == '_none'].index
    for ID in ids_unmapped:
        _dfhifld.loc[ID,'transreg'] = (
            dfzones.distance(_dfhifld.loc[ID, 'geometry']).nsmallest(1).index[0])

    for season in ['summer', 'winter']:
        ### Get the appropriate regional data
        if regional_air_temp is True:
            _dfhifld[f'temperature_{season}'] = _dfhifld.transreg.map(
                regional_assumptions[f'ambient_temp_{season}_celsius'] + dlr.C2K)
        elif regional_air_temp is False:
            _dfhifld[f'temperature_{season}'] = DEFAULTS['air_temperature']
        elif isinstance(regional_air_temp, dict):
            _dfhifld[f'temperature_{season}'] = regional_air_temp[season] + dlr.C2K
        else:
            _dfhifld[f'temperature_{season}'] = regional_air_temp + dlr.C2K

        if regional_wind is True:
            _dfhifld[f'windspeed_{season}'] = _dfhifld.transreg.map(
                regional_assumptions[f'windspeed_{season}_mps'])
            _dfhifld['windangle'] = _dfhifld.transreg.map(
                regional_assumptions['windangle_deg'])
        elif regional_wind is False:
            _dfhifld[f'windspeed_{season}'] = DEFAULTS['wind_speed']
            _dfhifld['windangle'] = DEFAULTS['wind_direction']
        else:
            _dfhifld[f'windspeed_{season}'] = regional_wind
            _dfhifld['windangle'] = DEFAULTS['wind_direction']

        if regional_irradiance is True:
            _dfhifld[f'ghi_{season}'] = _dfhifld.transreg.map(
                regional_assumptions[f'solar_radiation_{season}_watts_per_square_meter'])
        elif regional_irradiance is False:
            _dfhifld[f'ghi_{season}'] = DEFAULTS['ghi']
        else:
            _dfhifld[f'ghi_{season}'] = regional_irradiance

        if regional_conductor_temp is True:
            _dfhifld['conductor_temp'] = _dfhifld.transreg.map(
                regional_assumptions['conductor_acsr_temp_celsius'] + dlr.C2K)
        elif regional_conductor_temp is False:
            _dfhifld['conductor_temp'] = DEFAULTS['conductor_temperature']
        else:
            _dfhifld['conductor_temp'] = regional_conductor_temp

        if regional_emissivity is True:
            _dfhifld['emissivity'] = _dfhifld.transreg.map(
                regional_assumptions['emissivity'])
        elif regional_emissivity is False:
            _dfhifld['emissivity'] = DEFAULTS['emissivity']
        else:
            _dfhifld['emissivity'] = regional_emissivity

        if regional_absorptivity is True:
            _dfhifld['absorptivity'] = _dfhifld.transreg.map(
                regional_assumptions['absorptivity'])
        elif regional_absorptivity is False:
            _dfhifld['absorptivity'] = DEFAULTS['absorptivity']
        else:
            _dfhifld['absorptivity'] = regional_absorptivity

        ### Calculate the rating
        _dfhifld[f'ZLR_{season}'] = dlr.compute_DLR(
            windspeed=_dfhifld[f'windspeed_{season}'],
            wind_conductor_angle=_dfhifld['windangle'],
            T_ambient_air=_dfhifld[f'temperature_{season}'],
            solar_ghi=_dfhifld[f'ghi_{season}'],
            T_conductor=_dfhifld['conductor_temp'],
            diameter_conductor=_dfhifld.diameter,
            R=_dfhifld.AC_R_75C,
            emissivity_conductor=_dfhifld.emissivity,
            a_s=_dfhifld.absorptivity,
        )

    if minimal:
        outcols = ['transreg', 'ZLR_summer', 'ZLR_winter']
    else:
        outcols = (
            ['transreg', 'conductor_temp', 'emissivity', 'absorptivity', 'windangle']
            + [
                i+f'_{season}' for i in ['temperature', 'windspeed', 'ghi', 'ZLR']
                for season in ['summer','winter']
            ]
        )

    return _dfhifld[outcols]


def get_hifld(
    min_kv=115,
    max_miles=50,
    calc_slr=True,
    calc_zlr=False,
    within_poly=None,
    regional_air_temp=True,
    regional_wind=False,
    regional_irradiance=False,
    regional_conductor_temp=False,
    regional_emissivity=False,
    regional_absorptivity=False,
    aggfunc='representative',
    hifld_ids=slice(None),
    slr_kwargs={},
):
    """
    Inputs
    ------
    min_kv: Minimum voltage to include in results.
        ≥115 kV lines = HV/EHV transmission system (ANSI C84.1-2020)
        (see chat with Jarrad + Greg on 20240909)
    """
    ### Get HIFLD
    _dfhifld = clean_hifld_voltages(
        gpd.read_file(hifldpath)
    ).set_index('ID').loc[hifld_ids]
    ## Downselect to US
    dfhifld = _dfhifld.loc[
        (_dfhifld.bounds.maxy <= 1.4e6)
        & (_dfhifld.bounds.miny >= -1.8e6)
        & (_dfhifld.bounds.minx >= -2.5e6)
        & (_dfhifld.bounds.maxx <= 2.5e6)
        ## Remove DC
        & (_dfhifld.VOLT_CLASS != 'DC')
        ## Remove underground
        & (_dfhifld.TYPE.map(lambda x: 'UNDERGROUND' not in x))
    ].copy()
    ## Clip to provided polygon
    if within_poly is not None:
        dfhifld.geometry = dfhifld.intersection(within_poly)
    ## Remove lines below cutoff voltage and above maximum length
    dfhifld['length_miles'] = dfhifld.length / 1609.344
    dfhifld = dfhifld.loc[
        (dfhifld.VOLTAGE >= min_kv)
        & (dfhifld.length_miles <= max_miles)
    ]
    ### Calculate SLR and ZLR if desired
    if calc_slr:
        dfhifld['SLR'] = dlr.compute_DLR(
            diameter_conductor=dfhifld.diameter,
            R=dfhifld.AC_R_75C,
            **slr_kwargs,
        )

    if calc_zlr:
        dfhifld = dfhifld.merge(
            calculate_zlr(
                dfhifld=dfhifld,
                regional_air_temp=regional_air_temp,
                regional_wind=regional_wind,
                regional_irradiance=regional_irradiance,
                regional_conductor_temp=regional_conductor_temp,
                regional_emissivity=regional_emissivity,
                regional_absorptivity=regional_absorptivity,
                aggfunc=aggfunc,
            ),
            left_index=True, right_index=True, how='left',
        )

    return dfhifld


def get_lines_and_ratings(
    data_meas='DLR',
    data_base='ALR',
    path_to_meas=None,
    path_to_base=None,
    min_kv=115,
    max_miles=50,
    within_poly=None,
    years=range(2007,2014),
    tz='Etc/GMT+6',
    verbose=1,
    output='percent_diff',
    errors='warn',
    dropna=True,
    hifld_ids=slice(None),
    slr_kwargs={},
):
    """
    Inputs
    ------
    min_kv: Minimum voltage to include in results.
        ≥115 kV lines = HV/EHV transmission system (ANSI C84.1-2020)
        (see chat with Jarrad + Greg on 20240909)
    """
    # ### Settings for testing
    # data_meas = 'DLR'
    # data_base = 'ALR'
    # path_to_meas = '/Users/pbrown/Projects/ATM/DLR/results/dlrs_all_lines.h5'
    # path_to_base = '/Users/pbrown/Projects/ATM/DLR/results/aalrs_all_lines.h5'
    ### Get HIFLD
    dfhifld = get_hifld(
        min_kv=min_kv, max_miles=max_miles,
        calc_slr=True, within_poly=within_poly,
        calc_zlr=(True if data_base == 'ZLR' else False),
        slr_kwargs=slr_kwargs,
    )
    ids_hifld = dfhifld.index.values

    ### Get results
    if output != 'percent_diff':
        raise NotImplementedError("only output='percent_diff' is currently supported")
    _years = [years] if isinstance(years, int) else years
    _years = tqdm(_years) if verbose else _years
    if data_base is None:
        dfin = pd.concat(
            {
                year: (
                    pd.read_hdf(path_to_meas, key=str(year))
                    .reindex(columns=ids_hifld).dropna(axis=1, how='all')[hifld_ids]
                )
                for year in _years
            }, names=('year',),
        ### Drop year and switch to output timezone
        ).reset_index(level='year', drop=True).tz_convert(tz)
    else:
        dfin = pd.concat(
            {
                year: (
                    (
                        pd.read_hdf(path_to_meas, key=str(year))
                        .reindex(columns=ids_hifld).dropna(axis=1, how='all')[hifld_ids]
                    ) / (
                        pd.read_hdf(path_to_base, key=str(year))
                        .reindex(columns=ids_hifld).dropna(axis=1, how='all')[hifld_ids]
                    )
                    - 1
                ) * 100
                for year in _years
            }, names=('year',),
        ### Drop year and switch to output timezone
        ).reset_index(level='year', drop=True).tz_convert(tz)

    ### Subsets
    results_ids = dfin.columns
    unused_from_hifld = [c for c in results_ids if c not in dfhifld.index]
    missing_from_results = [c for c in dfhifld.index if c not in results_ids]
    print('results_ids:', len(results_ids))
    print('missing_from_results:', len(missing_from_results))
    print('unused_from_hifld:', len(unused_from_hifld))
    ## Drop lines missing from results
    dfhifld.drop(missing_from_results, inplace=True, errors='ignore')
    ## Drop lines missing or filtered from hifld
    dfin.drop(columns=unused_from_hifld, inplace=True, errors='ignore')
    print('after dropping lines filtered from hifld:', dfin.shape[1])
    ## Drop lines with missing data
    if dropna:
        dfin.dropna(axis=1, how='any', inplace=True)
    print('after dropping lines with missing data:', dfin.shape[1])

    if dfin.shape[1] != dfhifld.shape[0]:
        err = f"{dfin.shape[1]} results but {dfhifld.shape[0]} lines"
        print('WARNING:', err)
    if errors in ['warn','ignore']:
        return dfhifld, dfin
    else:
        raise IndexError(err)


def make_zlr_timeseries(
    dfhifld,
    time_index,
    winter_months=['Dec','Jan','Feb'],
):
    """Make timeseries of ratings using ZLR from dfhifld
    """
    months = range(1,13)
    monthabbrevs = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthnames = [
        'January',' February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December',
    ]
    month2num = {
        **dict(zip(monthabbrevs, months)),
        **dict(zip(monthnames, months)),
    }
    winter = [month2num.get(m,m) for m in winter_months]
    summer_times = time_index[~time_index.month.isin(winter)]
    winter_times = time_index[time_index.month.isin(winter)]

    dfout = pd.concat({
        **{t: dfhifld.ZLR_summer for t in summer_times},
        **{t: dfhifld.ZLR_winter for t in winter_times},
    }, axis=1).T.sort_index(axis=0)

    return dfout