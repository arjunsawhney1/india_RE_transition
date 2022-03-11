import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import xarray as xr
import pickle
import india_RE_analysis_python as irea

def main():
    profiles = ['wind', 'solar']
    states = ['Jammu and Kashmir', 'Punjab', 'Haryana', 'Rajasthan', 'Delhi',
    'Chandigarh', 'Himachal Pradesh', 'Uttar Pradesh', 'Uttaranchal', 'Goa',
    'Gujarat', 'Madhya Pradesh', 'Chhattisgarh', 'Maharashtra', 'Daman and Diu',
    'Dadra and Nagar Haveli', 'Andhra Pradesh', 'Telangana', 'Karnataka',
    'Kerala', 'Tamil Nadu', 'Puducherry', 'Bihar', 'Jharkhand', 'Orissa',
    'West Bengal', 'Sikkim', 'Assam', 'Manipur', 'Meghalaya', 'Nagaland',
    'Mizoram', 'Tripura', 'Arunachal Pradesh', 'Lakshadweep',
    'Andaman and Nicobar']
    projects = ['plants', 'mines', 'all']
    coal_status = ['Operating', 'Announced', 'Operating + Announced']

    # Load in and cache data
    mines = load_mines("June 2021 Global Coal Mine Tracker.csv")
    plants = load_plants("July 2021 Global Coal Plant Tracker.csv")

    # About Text
    st.title("Planning for High-Penetration Futures of Renewable Energy in \
    India")
    st.subheader("Created by Arjun Sawhney")
    st.subheader("")
    # Button to render plots
    render = st.button("Render Supply Curves")
    st.subheader("")
    # user interaction elements
    profile = st.selectbox("Select Profile:", profiles, index=1)
    state = st.selectbox("Select State:", states, index=11)
    project = st.selectbox("Filter by Coal Project:", projects, index=2)
    status = st.selectbox("Filter by Coal Project Status:", coal_status, index=2)
    radius = st.slider("Select Coal Project Radius (km):", min_value=25, max_value=100, value=50)
    age = st.slider("Filter by Coal Project Age:", min_value=0, max_value=40)

    if render:
        xarray_ds_wind, xarray_ds_solar, districts = load_start()

        res_path = 'res_{}.nc'.format(profile)
        prov_path = 'prov_{}.pickle'.format(profile)

        res_xarray_coarse = xr.open_dataset(res_path)

        with open(prov_path, 'rb') as handle:
            xarrays_prov = pickle.load(handle)

        ur_fig = get_unrestricted_plot(prov_path, profile)
        with open('unrestricted_{}.pickle'.format(profile), 'rb') as handle:
            unrestricted_dfs = pickle.load(handle)

        r_fig = get_restricted_plot(res_path, profile, mines, plants, project, status, radius, age)
        with open('restricted_{}.pickle'.format(profile), 'rb') as handle:
            restricted_dfs = pickle.load(handle)

        if len(restricted_dfs) != 0:
            st.subheader("State Supply Curve")
            st.plotly_chart(get_state_plot(profile, unrestricted_dfs,
            restricted_dfs, state))
            st.subheader("Unrestricted Supply Curve")
            st.plotly_chart(ur_fig)
            st.subheader("Restricted Supply Curve")
            st.plotly_chart(r_fig)
        else:
            st.subheader("No Announced {}".format(project))
            st.subheader("Unrestricted Supply Curve")
            st.plotly_chart(ur_fig)


@st.cache(allow_output_mutation=True)
def load_plants(path):
    plants = pd.read_csv(path, encoding="ISO-8859-1")
    plants = plants[plants['Country'] == 'India']
    plants = plants[plants['Latitude'].isna() == False]
    plants = plants[plants['Longitude'].isna() == False]
    plants = plants[plants['Status'].isin(['operating'])]
    # 'announced', 'cancelled', 'construction', 'permitted', 'pre-permit', 'retired', 'shelved'])]
    plants = plants.drop_duplicates('ParentID')
    plants = plants[['Plant', 'Subnational unit (province, state)', 'Status', 'Latitude', 'Longitude',
                     'Capacity (MW)', 'Annual CO2 (million tonnes / annum)', 'Year']]
    plants = plants.rename(columns={'Subnational unit (province, state)': 'State', 'Year': 'Plant Age'})
    plants['Capacity (MW)'] = plants['Capacity (MW)'].astype(float)
    plants['Plant Age'] = plants['Plant Age'].apply(lambda x: 2021 - float(x))

    return plants


@st.cache(allow_output_mutation=True)
def load_mines(path):
    mines = pd.read_csv(path, encoding="ISO-8859-1")
    mines = mines[mines['Country'] == 'India']
    mines = mines[mines['Status'] == 'Operating'] # 'Proposed'
    mines = mines[mines['Latitude'].isna() == False]
    mines = mines[mines['Longitude'].isna() == False]
    mines = mines.drop_duplicates('Mine ID')
    mines = mines[['Mine Name', 'Status', 'Status Detail', 'State, Province', 'Coal Output (Annual, Mt)', 'Mine Type',
                   'Latitude', 'Longitude', 'Opening Year']]
    mines = mines[mines['Opening Year'] != 'TBD']
    mines['Mine Age'] = mines['Opening Year'].apply(lambda x: 2021 - float(x))
    mines = mines.drop(columns=['Opening Year'])

    return mines


@st.cache(suppress_st_warning=True, hash_funcs={dict: lambda _: None})
def load_start():
    # load from pickle before first run
    file = open("xarray_ds_wind.pickle",'rb')
    xarray_ds_wind = pickle.load(file)
    file.close()

    file = open("xarray_ds_solar.pickle",'rb')
    xarray_ds_solar = pickle.load(file)
    file.close()

    file = open("districts.pickle",'rb')
    districts = pickle.load(file)
    file.close()

    return xarray_ds_wind, xarray_ds_solar, districts


@st.cache(hash_funcs={dict: lambda _: None})
def get_restricted_plot(res_path, profile, mines, plants,
project, status, radius, age):
    res_xarray_coarse = xr.open_dataset(res_path)

    restricted_dfs, r_fig = irea.restricted_case(res_xarray_coarse, profile, mines, plants, project, status, radius, age)

    with open('restricted_{}.pickle'.format(profile), 'wb') as handle:
        pickle.dump(restricted_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return r_fig


@st.cache(hash_funcs={dict: lambda _: None})
def get_state_plot(profile, unrestricted_dfs, restricted_dfs, state):
    if profile == 'wind':
        fig = irea.plot_cost_state(unrestricted_dfs,
        restricted_dfs, state, irea.wind_t_22, irea.wind_t_30)
        fig.update_layout(width = 800, height=800,
        title='{} Wind Capacity Supply Curves Unrestricted vs. Restricted'.format(state))
    else:
        fig = irea.plot_cost_state(unrestricted_dfs,
        restricted_dfs, state, irea.solar_t_22, irea.solar_t_30)
        fig.update_layout(width = 800, height=800,
        title='{} Solar Capacity Supply Curves Unrestricted vs. Restricted'.format(state))

    return fig

if __name__ == '__main__':
   main()
