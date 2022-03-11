#!/usr/bin/env python
# coding: utf-8

import logging
import warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

import geodata
import xarray as xr
import rioxarray as rxr
import pandas as pd
import geopandas as gpd
import numpy as np
import math
import pickle
import statsmodels.api as sm

import pyproj
import shapely
from pyproj import Proj
from osgeo import gdal
from shapely.geometry import shape
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.ops import transform
from shapely.ops import cascaded_union

#packages that requires users to install externally
from functools import partial
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import geoplot
import mapclassify
import matplotlib as mpl
from plotly.subplots import make_subplots
import plotly.graph_objects as go

## CUTOUT
cutout = geodata.Cutout(name = "india-2018-era5-wind-solar-hourly",
                        module = "era5",
                        weather_data_config = "wind_solar_hourly",
                        xs = slice(68, 98),
                        ys = slice(6, 37),
                        years=slice(2018, 2018),
                        months=slice(1,12))


jobs_cdf = sm.distributions.ECDF(districts['Coal Job %'].dropna())
pensions_cdf = sm.distributions.ECDF(districts['Total Pensioners'].dropna())
dmf_cdf = sm.distributions.ECDF(districts[districts['DMF USD'] > 0]['DMF USD'].dropna())
csr_cdf = sm.distributions.ECDF(districts[districts['CSR USD'] > 0]['CSR USD'].dropna())

def return_edges(cdf):
    bin_edges = []
    bin_edges.append(cdf.x[0])

    for lst in np.array_split(cdf.x, 3):
        bin_edges.append(lst[len(lst) - 1])

    return bin_edges

def return_groups(val, bin_edges):
    if val == 0:
        return 0
    elif val >= bin_edges[2] and val <= bin_edges[3]:
        return 3
    elif val >= bin_edges[1] and val <= bin_edges[2]:
        return 2
    elif val >= bin_edges[0] and val <= bin_edges[1]:
        return 1
    else:
        return None

def return_JPCD_sum(weights):
        # Bins
    jobs_edges = return_edges(jobs_cdf)
    pensions_edges = return_edges(pensions_cdf)
    dmf_edges = return_edges(csr_cdf)
    csr_edges = return_edges(dmf_cdf)

    # Assign labels to equal percentile groups
    job_groups = districts['Coal Job %'].apply(lambda x: return_groups(x, jobs_edges))
    pension_groups = districts['Total Pensioners'].apply(lambda x: return_groups(x, pensions_edges))
    csr_groups = districts['CSR USD'].apply(lambda x: return_groups(x, csr_edges))
    dmf_groups = districts['DMF USD'].apply(lambda x: return_groups(x, dmf_edges))

    # Create new columns
    JPCD_sum = districts.assign(
        J = job_groups,
        P = pension_groups,
        D = dmf_groups,
        C = csr_groups
    )
    JPCD_sum = JPCD_sum[['State Name', 'District Name', 'Geometry', 'J', 'P', 'D', 'C']]

    # Prepare for weighted sum
    JPCD_sum['J'] = JPCD_sum['J'].apply(lambda x: x*weights[0])
    JPCD_sum['P'] = JPCD_sum['P'].apply(lambda x: x*weights[1])
    JPCD_sum['C'] = JPCD_sum['C'].apply(lambda x: x*weights[2])
    JPCD_sum['D'] = JPCD_sum['D'].apply(lambda x: x*weights[3])

    # Create Rank Score by summing
    JPCD_sum['Score'] = JPCD_sum[['J', 'P', 'D', 'C']].sum(axis = 1)
    scores_cdf = sm.distributions.ECDF(JPCD_sum[JPCD_sum['Score'] > 0]['Score'])
    scores_edges = return_edges(scores_cdf)
    JPCD_sum['Rank'] = JPCD_sum['Score'].apply(lambda x: return_groups(x, scores_edges))

    # export to pickle
    filehandler = open("JPCD_sum.pickle","wb")
    pickle.dump(JPCD_sum, filehandler)
    filehandler.close()

    return JPCD_sum


def coarse(ori, tar, threshold = 0):
    '''
    This function will reindex the original xarray dataset according to the coordiantes of the target.

    There might be a bias for lattitudes and longitudes. The bias are normally within 0.01 degrees.
    In order to not lose too much data, a threshold for bias in degree could be given.
    When threshold = 0, it means that the function is going to find the best place with smallest bias.

    '''
    lat_multiple = round(((tar.lat[1] - tar.lat[0]) / (ori.lat[1] - ori.lat[0])).values.tolist())
    lon_multiple = round(((tar.lon[1] - tar.lon[0]) / (ori.lon[1] - ori.lon[0])).values.tolist())

    lat_start = find_intercept(ori.lat, tar.lat, (lat_multiple - 1) //  2)
    lon_start = find_intercept(ori.lon, tar.lon, (lon_multiple - 1) //  2)
    coarsen = ori.isel(lat = slice(lat_start, None), lon = slice(lon_start, None)).coarsen(
        dim = {'lat': lat_multiple, 'lon' : lon_multiple},
        side = {'lat': 'left', 'lon': 'left'},
        boundary = 'pad', ).mean()
    out = coarsen.reindex_like(tar, method = 'nearest')
    return out


def shpreader_path(theme):
    '''
    Given a theme such as 'admin_1_states_provinces' or 'admin_0_countries', download the dbf/shp/shx data from
    https://www.naturalearthdata.com/downloads/10m-cultural-vectors/;

    If the files have been successfully downloaded, return the path to these files that get_shapes() read;
    '''
    return shpreader.natural_earth(resolution='10m', category='cultural',
                                name = theme)


def get_prov_of(country_name):
    '''Take in a country name, return all the shapes of its provinces'''
    reader = shpreader.Reader(shpreader_path('admin_1_states_provinces')) #get provinces path
    provs = []
    prov_names = []
    for r in reader.records():
        if r.attributes['admin'] == country_name:
            provs.append(r.geometry)
            prov_names.append(r.attributes['name'])
    d = {'State': prov_names, 'Geometry': provs}

    return gpd.GeoDataFrame(d, crs = 'EPSG:4326')

country_path = shpreader_path('admin_0_countries')
prov_path = shpreader_path('admin_1_states_provinces')
provinces = get_prov_of('India')


def grid_coordinates(ds):
    xs, ys = np.meshgrid(ds.coords["lon"], ds.coords["lat"])
    return np.asarray((np.ravel(xs), np.ravel(ys))).T


def grid_cells(ds):
    coords = grid_coordinates(ds)
    span = (coords[len(ds.coords["lon"])+1] - coords[0]) / 2
    hs = np.hstack((coords - span, coords + span))
    out = np.array([None]*len(hs))
    for i in range(len(hs)):
        d = box(*hs[i])
        out[i] = d
    return out


def indicatormatrix(ds, shapes, shapes_proj='latlong'):
    return geodata.gis.compute_indicatormatrix(grid_cells(ds), shapes, shapes_proj, shapes_proj)


def shapes_on_ds(ds, profile, shape, true_area = False, t = None, get_df = False, m_name = 'PM2.5 (µg/m³)', m_size = (10, 5), m_color = "bone_r"):
    """
    Plot a map of the dataArray region, with only PM25 data of the specified targets at time = t
    Return the dataArray containing only the regions from the shape after multiplying with indicator matrix,
    the indicator matrix itself, names of the location/shape, with an optional dataframe contains features in the shape;
    //to do: write documentation
    """
    #re-size the dataArray
    lats, lons = [], []
    lat_diff = (ds.lat[1] - ds.lat[0]) * 2
    lon_diff = (ds.lon[1] - ds.lon[0]) * 2
    for sh in shape:
        shb = sh.bounds
        lats.append(shb[1])
        lats.append(shb[3])
        lons.append(shb[0])
        lons.append(shb[2])
    min_lat, max_lat = min(lats) - lat_diff, max(lats) + lat_diff
    min_lon, max_lon = min(lons) - lon_diff, max(lons) + lon_diff
    ds = ds.where((ds.lat >= min_lat) & (ds.lat <= max_lat),
                  drop=True).where((ds.lon >= min_lon) & (ds.lon <= max_lon), drop=True)
    #the next 10 lines are for stacking the shape data together, reflecting in the area the cutout covers
    try:
        geo_shapes = gpd.GeoSeries([r.geometry for r in shape])
    except:
        geo_shapes = gpd.GeoSeries(shape)
    indicator_matrix = np.zeros(len(indicatormatrix(ds, geo_shapes)[0].toarray()))
    for i in range(len(geo_shapes)):
        indicator_matrix_each = indicatormatrix(ds, geo_shapes)[i]
        #turn it into a 1D array which we can stack on
        indicator_matrix = indicator_matrix + indicator_matrix_each.toarray()

    indicator_matrix = xr.DataArray(indicator_matrix.reshape(len(ds.lat), len(ds.lon)), dims=['lat','lon'],
                                       coords=[ds.lat, ds.lon])
    #filter out the non-specified province part
    im = indicator_matrix.copy()
    im_return = indicator_matrix.copy()
    #find correct time point to plot
    if t != None:
        #set up the axes
        fig = plt.figure(figsize=m_size)
        ax = fig.add_subplot(projection=ccrs.PlateCarree());
        ax.set_extent([ds.lon.min(), ds.lon.max(), ds.lat.min(), ds.lat.max()])
        if isinstance(t, int):
            im.values = (im.values > 0) * ds.isel(time=t)
        else:
            im.values = (im.values > 0) * ds.sel(time=t)
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        #ax.set_title("{} Amount at time: {}".format(ds.name, t))
        im.plot(ax=ax, cmap= m_color)
    if true_area:
        area = im_area(indicator_matrix)
        total_area = area.sum()
        indicator_matrix.values = np.array(area).reshape(len(ds.lat), len(ds.lon))
    #ds = ds.drop(['lat', 'lon']).rename({'x': 'lon','y': 'lat'})
    if profile == 'wind':
        indicator_matrix = indicator_matrix.expand_dims({'time':ds.time})
    #this line below contributes to the longer runtime of the method, it is
    #compute the product of original dataArray with the indicator matrix at
    #each time point
    for i in range(len(ds)):
        ds[i] *= indicator_matrix[i]
    output = [ds, im_return]
    if true_area: output.append(total_area)
    if get_df: output.append(pd.DataFrame([shape[i].attributes for i in range(len(shape))]))
    return output

def batch_ds(ds, profile, shapes, time):
    xarrays_prov = {state:0 for state in shapes.State.values}
    for i in range(len(shapes.index)):
        df, im = shapes_on_ds(ds, profile, [shapes.iloc[i]['Geometry']], t=time)
        xarrays_prov[shapes.iloc[i]['State']] = df
    return xarrays_prov


def calc_dist(lat1, lon1, lat2, lon2):
    from math import sin, cos, sqrt, atan2, radians
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def calc_area(lat, lat_higher, lon, lon_higher):
    '''
    Turn the area grid into trapezoid and calculate area
    example intput: [35, 35.5, 115, 115.625]
    '''

    #generate trapezoid
    top_len = calc_dist(lat_higher, lon, lat_higher, lon_higher)
    bottom_len = calc_dist(lat, lon, lat, lon_higher)
    side = calc_dist(lat, lon, lat_higher, lon)

    #calculate the height of the trapezoid
    height = (side ** 2 - ((top_len - bottom_len) / 2) ** 2) ** 0.5

    area = (1/2) * (top_len + bottom_len) * height
    return round(area,2)


def calc_prov_mask(xs_provs):
    total_areas = {state:[] for state in xs_provs}
    for xs_prov in xs_provs:
        df = xs_provs[xs_prov].to_dataframe().reset_index(drop = False)
        co_prov = df[df['CF'] > 0][['lat','lon']].values
        areas = []
        for i in co_prov:
            areas.append(calc_area(i[0], i[0]+0.5, i[1], i[1]+0.625))
        total_areas[xs_prov] = areas
    return total_areas


def make_figures(df, profile, areas, col = 'CF'):
    df = df[df[col] > 0]
    df['areas'] = areas
    if profile == 'wind':
        df['cap'] = df.areas * 3.9 / 1000
    else:
        df['cap'] = df.areas * 27.7 / 1000
    df = df.sort_values(by = col, ascending = False)
    df['cumsum'] = df['cap'].cumsum()
    return df.reset_index(drop=True)


def batch_figures(xr_provs, profile, provs, col = 'CF'):
    dfs = {state:[] for state in xr_provs}
    areas = calc_prov_mask(xr_provs)
    for xr_prov in xr_provs:
        df = xr_provs[xr_prov].to_dataframe().reset_index(drop=False)
        df = make_figures(df, profile, areas[xr_prov], col)
        dfs[xr_prov] = df

    fig = make_subplots(rows=3, cols=1, subplot_titles=(
        "High {} Capacity Supply Curves for Provinces in India".format(profile),
        "Low {} Capacity Supply Curves for Provinces in India".format(profile),
        "Overall {} Capacity Supply Curves for Provinces in India".format(profile)
    ))

    if profile == 'solar':
        thresh = 0.15
    else:
        thresh = 0.1

    for xr_prov in xr_provs:
        if xr_prov in dfs.keys() and len(dfs[xr_prov][col] > 0):

            # dfs[xr_prov]['cumsum'] -= dfs[xr_prov]['cumsum'][0]

            if dfs[xr_prov][col].iloc[0] > thresh:
                fig.add_trace(
                    go.Scatter(
                        x=dfs[xr_prov]['cumsum'],
                        y=dfs[xr_prov][col],
                        mode='lines',
                        name=xr_prov,
                        showlegend=False),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=dfs[xr_prov]['cumsum'],
                        y=dfs[xr_prov][col],
                        mode='lines',
                        name=xr_prov,
                        showlegend=False),
                    row=2, col=1
                )

            fig.add_trace(
                go.Scatter(
                    x=dfs[xr_prov]['cumsum'],
                    y=dfs[xr_prov][col],
                    mode='lines',
                    name=xr_prov),
                row=3, col=1
            )

    # Update xaxis properties
    fig.update_xaxes(title_text="Cumulative Capacity (GW)", row=1, col=1, range=[0,500])
    fig.update_xaxes(title_text="Cumulative Capacity (GW)", row=2, col=1, range=[0,500])
    fig.update_xaxes(title_text="Cumulative Capacity (GW)", row=3, col=1, range=[0,500])

    # Update yaxis properties
    fig.update_yaxes(title_text="Capacity Factor (%)", row=1, col=1, range=[0,0.4])
    fig.update_yaxes(title_text="Capacity Factor (%)", row=2, col=1, range=[0,0.2])
    fig.update_yaxes(title_text="Capacity Factor (%)", row=3, col=1, range=[0,0.4])

    fig.update_layout(height=1200, width=800)

    return dfs, fig


proj_wgs84 = Proj('+proj=longlat +datum=WGS84')
def geodesic_point_buffer(lat, lon, km):
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(km * 1000)  # distance in metres
    return Polygon(transform(project, buf).exterior.coords[:])


def restricted_case(rxc, profile, mines, plants, case, status, radius, age, mine_cap=0, plant_cap=0):
    # User filteration
    mines = mines[mines['Coal Output (Annual, Mmt)'] > mine_cap]
    plants = plants[plants['Capacity (MW)'] > plant_cap]
    mines = mines[mines['Mine Age'] > age]
    plants = plants[plants['Plant Age'] > age]

    if status == 'Operating':
        plants = plants[plants['Status'].isin([status])]
        mines = mines[mines['Status'] == status]
    elif status == 'Announced':
        plants = plants[plants['Status'].isin([status])]
        mines = mines[mines['Status'] != 'Operating']
    else:
        plants = plants[plants['Status'].isin(['Operating', 'Announced'])]

    # Different cases
    if case == 'plants':
        if len(plants) == 0:
            coal_mask = plants[['State']]
            coal_mask['Geometry'] = np.nan
        else:
            plants['Geometry'] = plants.apply(lambda x: geodesic_point_buffer(x['Latitude'], x['Longitude'], radius), axis=1)
            plant_points = plants[['State', 'Geometry']]
            coal_mask = plant_points
    elif case == 'mines':
        if len(mines) == 0:
            coal_mask = mines[['State']]
            coal_mask['Geometry'] = np.nan
        else:
            mines['Geometry'] = mines.apply(lambda x: geodesic_point_buffer(x['Latitude'], x['Longitude'], radius), axis=1)
            mine_points = mines[['State', 'Geometry']]
            coal_mask = mine_points
    else:
        plants['Geometry'] = plants.apply(lambda x: geodesic_point_buffer(x['Latitude'], x['Longitude'], radius), axis=1)
        plant_points = plants[['State', 'Geometry']]
        mines['Geometry'] = mines.apply(lambda x: geodesic_point_buffer(x['Latitude'], x['Longitude'], radius), axis=1)
        mine_points = mines[['State', 'Geometry']]
        coal_mask = mine_points.append(plant_points)

    xarrays_coal = batch_ds(rxc['CF'], profile, coal_mask, None)
    restricted_dfs, r_fig = batch_figures(xarrays_coal, profile, coal_mask['State'])

    return restricted_dfs, r_fig

wind_t_22 = {
    'Tamil Nadu' : 11.9,
    'Gujarat' : 8.8,
    'Rajasthan' : 8.6,
    'Andhra Pradesh' : 8.1,
    'Maharashtra' : 7.6,
    'Karnataka' : 6.2,
    'Madhya Pradesh' : 6.2,
    'Telangana' : 2.6,
    'Jammu and Kashmir' : 0,
    'Arunachal Pradesh' : 0,
    'West Bengal' : 0,
    'Sikkim' : 0,
    'Uttaranchal' : 0,
    'Uttar Pradesh' : 0,
    'Bihar' : 0,
    'Nagaland' : 0,
    'Manipur' : 0,
    'Mizoram' : 0,
    'Tripura' : 0,
    'Meghalaya' : 0,
    'Punjab' : 0,
    'Himachal Pradesh' : 0,
    'Daman and Diu' : 0,
    'Goa' : 0,
    'Kerala' : 0,
    'Puducherry' : 0,
    'Lakshadweep' : 0,
    'Andaman and Nicobar' : 0,
    'Himachal Pradesh': 0,
    'Delhi' : 0,
    'Dadra and Nagar Haveli' : 0,
    'Chandigarh' : 0,
    'Chhattisgarh' : 0,
    'Haryana' : 0,
    'Jharkhand' : 0,
    'Orissa' : 0
}

wind_t_30 = {
    'Karnataka' : 15.416,
    'Tamil Nadu' : 13.796,
    'Gujarat' : 13.677,
    'Rajasthan' : 13.130,
    'Madhya Pradesh' : 12.474,
    'Maharashtra' : 11.410,
    'Telangana' : 8.482,
    'Andhra Pradesh' : 8.1,
    'Haryana' : 5.220,
    'Chhattisgarh' : 5.171,
    'Jharkhand' : 4.938,
    'Uttar Pradesh' : 4.781,
    'West Bengal' : 3.0,
    'Bihar' : 3.0,
    'Punjab' : 2.0,
    'Uttaranchal' : 0.860,
    'Kerala' : 0.788,
    'Delhi' : 0.096,
    'Goa' : 0.002,
    'Jammu and Kashmir' : 0,
    'Arunachal Pradesh' : 0,
    'Sikkim' : 0,
    'Nagaland' : 0,
    'Manipur' : 0,
    'Mizoram' : 0,
    'Tripura' : 0,
    'Meghalaya' : 0,
    'Himachal Pradesh' : 0,
    'Daman and Diu' : 0,
    'Puducherry' : 0,
    'Lakshadweep' : 0,
    'Andaman and Nicobar' : 0,
    'Himachal Pradesh': 0,
    'Dadra and Nagar Haveli' : 0,
    'Chandigarh' : 0,
    'Orissa' : 0
}

solar_t_22 = {
    'Maharashtra' : 11.926,
    'Uttar Pradesh' : 10.697,
    'Andhra Pradesh' : 9.834,
    'Tamil Nadu' : 8.884,
    'Gujarat' : 8.020,
    'Rajasthan' : 5.762,
    'Karnataka' : 5.697,
    'Madhya Pradesh' : 5.675,
    'West Bengal' : 5.336,
    'Punjab' : 4.772,
    'Haryana' : 4.142,
    'Delhi' : 2.762,
    'Bihar' : 2.493,
    'Orissa' : 2.377,
    'Jharkhand' : 1.995,
    'Kerala' : 1.870,
    'Jammu and Kashmir' : 1.155,
    'Uttaranchal' : 0.9,
    'Himachal Pradesh': 0.776,
    'Dadra and Nagar Haveli' : 0.449,
    'Goa' : 0.358,
    'Puducherry' : 0.246,
    'Daman and Diu' : 0.199,
    'Meghalaya' : 0.161,
    'Chandigarh' : 0.153,
    'Chhattisgarh' : 0.153,
    'Manipur' : 0.105,
    'Tripura' : 0.105,
    'Mizoram' : 0.072,
    'Nagaland' : 0.061,
    'Telangana' : 0.05831,
    'Arunachal Pradesh' : 0.039,
    'Sikkim' : 0.036,
    'Andaman and Nicobar' : 0.027,
    'Lakshadweep' : 0.004,
    'Himachal Pradesh' : 0
}

solar_t_30 = {
    'Tamil Nadu' : 18.449,
    'Rajasthan' : 18.042,
    'Gujarat' : 14.26,
    'Karnataka' : 13.338,
    'Telangana' : 12.971,
    'Madhya Pradesh' : 12.923,
    'Maharashtra' : 12.506,
    'Uttar Pradesh' : 11.500,
    'Orissa' : 10.971,
    'Bihar' : 10.431,
    'Andhra Pradesh' : 9.834,
    'Chhattisgarh' : 9.483,
    'West Bengal' : 8.077,
    'Arunachal Pradesh' : 6.870,
    'Punjab' : 6.0,
    'Jharkhand' : 5.277,
    'Haryana' : 5.040,
    'Uttaranchal' : 4.0,
    'Delhi' : 2.762,
    'Kerala' : 2.625,
    'Jammu and Kashmir' : 1.546,
    'Goa' : 1.025,
    'Himachal Pradesh': 0.776,
    'Himachal Pradesh' : 0.500,
    'Dadra and Nagar Haveli' : 0.449,
    'Puducherry' : 0.246,
    'Daman and Diu' : 0.199,
    'Meghalaya' : 0.161,
    'Chandigarh' : 0.153,
    'Manipur' : 0.105,
    'Tripura' : 0.105,
    'Mizoram' : 0.072,
    'Nagaland' : 0.061,
    'Sikkim' : 0.036,
    'Andaman and Nicobar' : 0.027,
    'Lakshadweep' : 0.004
}

targets_22 = {key: wind_t_22.get(key, 0) + solar_t_22.get(key, 0)
          for key in wind_t_22.keys()}

targets_30 = {key: wind_t_30.get(key, 0) + solar_t_30.get(key, 0)
          for key in wind_t_30.keys()}

def plot_cost_state(unrestricted_dfs, restricted_dfs, state, t_22, t_30):

    unrestricted_dfs[state]['cumsum'] -= unrestricted_dfs[state]['cumsum'][0]
    restricted_dfs[state]['cumsum'] -= restricted_dfs[state]['cumsum'][0]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=unrestricted_dfs[state]['cumsum'],
            y=unrestricted_dfs[state]['CF'],
            name='Unrestricted Case')
    )

    if state in restricted_dfs.keys():
        fig.add_trace(
            go.Scatter(
                x=restricted_dfs[state]['cumsum'],
                y=restricted_dfs[state]['CF'],
                name='Restricted Case')
        )

    fig.add_vline(x=t_22[state], line_dash="dot", annotation_text="Target 2022", annotation_position="bottom right")
    fig.add_vline(x=t_30[state], line_dash="dot", annotation_text="Target 2030", annotation_position="bottom right")

    fig.update_xaxes(title_text="Cumulative Capacity (GW)", range=[0, 50])
    fig.update_yaxes(title_text="Capacity Factor (%)", range=[0,0.4])

    return fig
