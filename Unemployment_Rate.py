import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import seaborn as sns

# Function to load data with caching
@st.cache_data
def load_data():
    # Load data
    world = gpd.read_file('data/110m_cultural/ne_110m_admin_0_countries.shp')
    unemployment = pd.read_csv('data/aged_15plus_unemployment_rate_percent.csv')
    unemployment_15_24 = pd.read_csv('data/aged_15_24_unemployment_rate_percent.csv')
    unemployment_25_54 = pd.read_csv('data/aged_25_54_unemployment_rate_percent.csv')
    unemployment_55_64 = pd.read_csv('data/aged_55_64_unemployment_rate_percent.csv')
    unemployment_female = pd.read_csv('data/females_aged_15plus_unemployment_rate_percent.csv')
    unemployment_male = pd.read_csv('data/males_aged_15plus_unemployment_rate_percent.csv')
    unemployment_female_15_24 = pd.read_csv('data/females_aged_15_24_unemployment_rate_percent.csv')
    unemployment_female_25_54 = pd.read_csv('data/females_aged_25_54_unemployment_rate_percent.csv')
    unemployment_female_55_64 = pd.read_csv('data/females_aged_55_64_unemployment_rate_percent.csv')
    unemployment_male_15_24 = pd.read_csv('data/males_aged_15_24_unemployment_rate_percent.csv')
    unemployment_male_25_54 = pd.read_csv('data/males_aged_25_54_unemployment_rate_percent.csv')
    unemployment_male_55_64 = pd.read_csv('data/males_aged_55_64_unemployment_rate_percent.csv')
    
    # Strip whitespace
    world['ADMIN'] = world['ADMIN'].str.strip()
    unemployment['country'] = unemployment['country'].str.strip()
    unemployment_15_24['country'] = unemployment_15_24['country'].str.strip()
    unemployment_25_54['country'] = unemployment_25_54['country'].str.strip()
    unemployment_55_64['country'] = unemployment_55_64['country'].str.strip()
    unemployment_female['country'] = unemployment_female['country'].str.strip()
    unemployment_male['country'] = unemployment_male['country'].str.strip()
    unemployment_female_15_24['country'] = unemployment_female_15_24['country'].str.strip()
    unemployment_female_25_54['country'] = unemployment_female_25_54['country'].str.strip()
    unemployment_female_55_64['country'] = unemployment_female_55_64['country'].str.strip()
    unemployment_male_15_24['country'] = unemployment_male_15_24['country'].str.strip()
    unemployment_male_25_54['country'] = unemployment_male_25_54['country'].str.strip()
    unemployment_male_55_64['country'] = unemployment_male_55_64['country'].str.strip()

    # Drop rows with a lot of NaNs
    unemployment = unemployment.drop(unemployment.columns[1:54], axis=1)
    unemployment_15_24 = unemployment_15_24.drop(unemployment_15_24.columns[1:53], axis=1)
    unemployment_25_54 = unemployment_25_54.drop(unemployment_25_54.columns[1:53], axis=1)
    unemployment_55_64 = unemployment_55_64.drop(unemployment_55_64.columns[1:53], axis=1)
    unemployment_female = unemployment_female.drop(unemployment_female.columns[1:53], axis=1)
    unemployment_male = unemployment_male.drop(unemployment_male.columns[1:53], axis=1)
    unemployment_female_15_24 = unemployment_female_15_24.drop(unemployment_female_15_24.columns[1:53], axis=1)
    unemployment_female_25_54 = unemployment_female_25_54.drop(unemployment_female_25_54.columns[1:53], axis=1)
    unemployment_female_55_64 = unemployment_female_55_64.drop(unemployment_female_55_64.columns[1:53], axis=1)
    unemployment_male_15_24 = unemployment_male_15_24.drop(unemployment_male_15_24.columns[1:53], axis=1)
    unemployment_male_25_54 = unemployment_male_25_54.drop(unemployment_male_25_54.columns[1:53], axis=1)
    unemployment_male_55_64 = unemployment_male_55_64.drop(unemployment_male_55_64.columns[1:53], axis=1)

    # Add a column containing the average rate over all years
    unemployment['average'] = unemployment.iloc[:, 1:].mean(axis=1)
    unemployment_15_24['average'] = unemployment_15_24.iloc[:, 1:].mean(axis=1)
    unemployment_25_54['average'] = unemployment_25_54.iloc[:, 1:].mean(axis=1)
    unemployment_55_64['average'] = unemployment_55_64.iloc[:, 1:].mean(axis=1)
    unemployment_female['average'] = unemployment_female.iloc[:, 1:].mean(axis=1)
    unemployment_male['average'] = unemployment_male.iloc[:, 1:].mean(axis=1)
    unemployment_female_15_24['average'] = unemployment_female_15_24.iloc[:, 1:].mean(axis=1)
    unemployment_female_25_54['average'] = unemployment_female_25_54.iloc[:, 1:].mean(axis=1)
    unemployment_female_55_64['average'] = unemployment_female_55_64.iloc[:, 1:].mean(axis=1)
    unemployment_male_15_24['average'] = unemployment_male_15_24.iloc[:, 1:].mean(axis=1)
    unemployment_male_25_54['average'] = unemployment_male_25_54.iloc[:, 1:].mean(axis=1)
    unemployment_male_55_64['average'] = unemployment_male_55_64.iloc[:, 1:].mean(axis=1)

    # Add a column containing the average rate over years 2006-2017
    unemployment_2006_2020 = unemployment.drop(unemployment.columns[1:7], axis=1)
    unemployment_2006_2020 = unemployment_2006_2020.drop(unemployment_2006_2020.columns[16:], axis=1)
    unemployment['average_2006_2020'] = unemployment_2006_2020.iloc[:, 1:].mean(axis=1)
    unemployment_2000_2017 = unemployment.drop(unemployment.columns[19:], axis=1)
    unemployment['average_2000_2017'] = unemployment_2000_2017.iloc[:, 1:].mean(axis=1)

    # Applying country mapping to the world DataFrame
    country_mapping = { 'Guinea-Bissau': 'Guinea-Bissau', 'eSwatini': 'Eswatini', 'Republic of the Congo': 'Congo, Rep.', 'Somalia': 'Somalia', 'Central African Republic': 'Central African Republic', 'Eritrea': 'Eritrea', 'Republic of Serbia': 'Serbia', 'Czechia': 'Czech Republic', 'Democratic Republic of the Congo': 'Congo, Dem. Rep.', 'North Korea': 'North Korea', 'Somaliland': 'Somaliland', 'The Bahamas': 'Bahamas', 'United Arab Emirates': 'UAE', 'French Southern and Antarctic Lands': 'French Southern and Antarctic Lands', 'Laos': 'Lao', 'United Kingdom': 'UK', 'Slovakia': 'Slovak Republic', 'Ivory Coast': "Cote d'Ivoire", 'Antarctica': 'Antarctica', 'United States of America': 'USA', 'East Timor': 'Timor-Leste', 'Northern Cyprus': 'Northern Cyprus', 'United Republic of Tanzania': 'Tanzania','Western Sahara': 'Western Sahara', 'Kyrgyzstan': 'Kyrgyz Republic', 'Falkland Islands': 'Falkland Is (Malvinas)' }
    world['ADMIN'] = world['ADMIN'].replace(country_mapping)

    # Add prefix to columns
    unemployment = unemployment.add_prefix('unemp_')
    unemployment_15_24 = unemployment_15_24.add_prefix('u_15_24_')
    unemployment_25_54 = unemployment_25_54.add_prefix('u_25_54_')
    unemployment_55_64 = unemployment_55_64.add_prefix('u_55_64_')
    unemployment_female = unemployment_female.add_prefix('u_female_15plus_')
    unemployment_male = unemployment_male.add_prefix('u_male_15plus_')
    unemployment_female_15_24 = unemployment_female_15_24.add_prefix('u_female_15_24_')
    unemployment_female_25_54 = unemployment_female_25_54.add_prefix('u_female_25_54_')
    unemployment_female_55_64 = unemployment_female_55_64.add_prefix('u_female_55_64_')
    unemployment_male_15_24 = unemployment_male_15_24.add_prefix('u_male_15_24_')
    unemployment_male_25_54 = unemployment_male_25_54.add_prefix('u_male_25_54_')
    unemployment_male_55_64 = unemployment_male_55_64.add_prefix('u_male_55_64_')

    # Merge data
    world = world.merge(unemployment, how='left', left_on='ADMIN', right_on='unemp_country')
    world = world.merge(unemployment_15_24, how='left', left_on='ADMIN', right_on='u_15_24_country')
    world = world.merge(unemployment_25_54, how='left', left_on='ADMIN', right_on='u_25_54_country')
    world = world.merge(unemployment_55_64, how='left', left_on='ADMIN', right_on='u_55_64_country')
    world = world.merge(unemployment_female, how='left', left_on='ADMIN', right_on='u_female_15plus_country')
    world = world.merge(unemployment_male, how='left', left_on='ADMIN', right_on='u_male_15plus_country')
    world = world.merge(unemployment_female_15_24, how='left', left_on='ADMIN', right_on='u_female_15_24_country')
    world = world.merge(unemployment_female_25_54, how='left', left_on='ADMIN', right_on='u_female_25_54_country')
    world = world.merge(unemployment_female_55_64, how='left', left_on='ADMIN', right_on='u_female_55_64_country')
    world = world.merge(unemployment_male_15_24, how='left', left_on='ADMIN', right_on='u_male_15_24_country')
    world = world.merge(unemployment_male_25_54, how='left', left_on='ADMIN', right_on='u_male_25_54_country')
    world = world.merge(unemployment_male_55_64, how='left', left_on='ADMIN', right_on='u_male_55_64_country')

    # Dissolve geometries by continent
    world_continents = world.dissolve(by='CONTINENT')
    # Calculate the average unemployment, happiness, and education by continent
    unemployment_continents = world.dropna(subset=['unemp_average', 'unemp_average_2006_2020', 'unemp_average_2000_2017'])
    unemployment_continents = unemployment_continents.groupby('CONTINENT')[['unemp_average', 'unemp_average_2006_2020', 'unemp_average_2000_2017',
                                                                            'u_female_15_24_average', 'u_female_25_54_average', 'u_female_55_64_average',
                                                                            'u_male_15_24_average', 'u_male_25_54_average', 'u_male_55_64_average']].mean()
    unemployment_15_24_continents = world.dropna(subset=['u_15_24_average'])
    unemployment_15_24_continents = unemployment_15_24_continents.groupby('CONTINENT')['u_15_24_average'].mean()
    unemployment_25_54_continents = world.dropna(subset=['u_25_54_average'])
    unemployment_25_54_continents = unemployment_25_54_continents.groupby('CONTINENT')['u_25_54_average'].mean()
    unemployment_55_64_continents = world.dropna(subset=['u_55_64_average'])
    unemployment_55_64_continents = unemployment_55_64_continents.groupby('CONTINENT')['u_55_64_average'].mean()
    unemployment_female_continents = world.dropna(subset=['u_female_15plus_average'])
    unemployment_female_continents = unemployment_female_continents.groupby('CONTINENT')['u_female_15plus_average'].mean()
    unemployment_male_continents = world.dropna(subset=['u_male_15plus_average'])
    unemployment_male_continents = unemployment_male_continents.groupby('CONTINENT')['u_male_15plus_average'].mean()

    # Merge the dissolved geometries with the average unemployment and happiness data
    world_continents = world_continents.drop(columns=['unemp_average', 'unemp_average_2006_2020', 'unemp_average_2000_2017',
                                                    'u_15_24_average', 'u_25_54_average', 'u_55_64_average',
                                                    'u_female_15plus_average', 'u_male_15plus_average', 'u_female_15_24_average', 'u_female_25_54_average', 'u_female_55_64_average',
                                                    'u_male_15_24_average', 'u_male_25_54_average', 'u_male_55_64_average'])
    world_continents = world_continents.merge(unemployment_continents, on='CONTINENT')
    world_continents = world_continents.merge(unemployment_15_24_continents, on='CONTINENT')
    world_continents = world_continents.merge(unemployment_25_54_continents, on='CONTINENT')
    world_continents = world_continents.merge(unemployment_55_64_continents, on='CONTINENT')
    world_continents = world_continents.merge(unemployment_female_continents, on='CONTINENT')
    world_continents = world_continents.merge(unemployment_male_continents, on='CONTINENT')
    
    # Filter for Europe
    oceania = world[world['CONTINENT'] == 'Oceania']
    africa = world[world['CONTINENT'] == 'Africa']
    north_america = world[world['CONTINENT'] == 'North America']
    asia = world[world['CONTINENT'] == 'Asia']
    south_america = world[world['CONTINENT'] == 'South America']
    europe = world[world['CONTINENT'] == 'Europe']

    # List of continents and custom axis limits
    continents = [africa, asia, europe, north_america, oceania, south_america]
    continent_names = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    xlims = [(-30, 70), (20, 150), (-30, 40), (-170, -10), (100, 180), (-110, -10)]
    ylims = [(-40, 40), (-10, 60), (35, 75), (0, 80), (-50, 0), (-60, 20)]

    return (world, world_continents, unemployment, unemployment_15_24, unemployment_25_54, unemployment_55_64, 
            unemployment_female, unemployment_male, unemployment_female_15_24, unemployment_female_25_54, 
            unemployment_female_55_64, unemployment_male_15_24, unemployment_male_25_54, unemployment_male_55_64,
            oceania, africa, north_america, asia, south_america, europe, continents, continent_names, xlims, ylims)

world, world_continents, unemployment, unemployment_15_24, unemployment_25_54, unemployment_55_64, unemployment_female, unemployment_male, unemployment_female_15_24, unemployment_female_25_54, unemployment_female_55_64, unemployment_male_15_24, unemployment_male_25_54, unemployment_male_55_64, oceania, africa, north_america, asia, south_america, europe, continents, continent_names, xlims, ylims = load_data()










# Streamlit page
st.title('Unemployment Rate')
st.write("Explore global and continental unemployment trends over the past two decades. Visualize and analyze the variability of unemployment rates within continents, and compare trends across different countries and demographic groups.")

st.header('Global and Continental Unemployment')
st.write("This map provides a comprehensive overview of unemployment rates around the world over the past two decades. The varying colors represent different ranges of average unemployment rates. This visual tool is essential for understanding the global landscape of unemployment and identifying patterns.")

# Add a radio button to choose the view type
view_type = st.radio(
    'Select the view type:',
    ('Country', 'Continent'),
    key='radio_1'
)

# Add a selectbox to choose the gender
gender_option = st.selectbox(
    'Select the gender to display the unemployment rate:',
    ('All', 'Female', 'Male'),
    key='gender_option_1'
)

# Add a selectbox to choose the age group
age_group = st.selectbox(
    'Select the age group to display the unemployment rate:',
    ('All', '15-24', '25-54', '55-64'),
    key='age_group_1'
)

# Set the column name based on the user's selection
if gender_option == 'All':
    if age_group == 'All':
        column = 'unemp_average'
    elif age_group == '15-24':
        column = 'u_15_24_average'
    elif age_group == '25-54':
        column = 'u_25_54_average'
    else:
        column = 'u_55_64_average'
else:
    prefix = f'u_{gender_option.lower()}_'
    if age_group == 'All':
        column = f'{prefix}15plus_average'
    else:
        column = f'{prefix}{age_group.replace("-", "_")}_average'

# Set the title based on the selection
if gender_option == 'All' and age_group == 'All':
    title_suffix = ''
elif gender_option == 'All':
    title_suffix = f', Aged {age_group}'
elif age_group == 'All':
    title_suffix = f', {gender_option}s'
else:
    title_suffix = f', {gender_option}s Aged {age_group}'

# Logic to display the map based on view type
if view_type == 'Country':
    # Add a selectbox to choose the specific continent for country view
    continent_option = st.selectbox(
        'Select the continent to display:',
        ('Global',) + tuple(continent_names),
        key='slect_continent'
    )

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    if continent_option == 'Global':
        world.plot(column=column, ax=ax, legend=True,
                   cmap='OrRd', edgecolor='black', missing_kwds={"color": "lightgrey", "label": "No data"},
                   scheme='natural_breaks', k=5, legend_kwds={'title': "Unemployment Rate (%)"})
        plt.title(f'Global Average Unemployment Rates by Country{title_suffix} (2000 - 2020)')
    else:
        # Find the index of the selected continent
        idx = continent_names.index(continent_option)
        continent = continents[idx]
        continent.plot(column=column, ax=ax, legend=True,
                       cmap='OrRd', edgecolor='black', missing_kwds={"color": "lightgrey", "label": "No data"},
                       scheme='natural_breaks', k=5, legend_kwds={'title': "Unemployment Rate (%)"})
        plt.title(f'Average Unemployment Rates in {continent_option}{title_suffix} (2000 - 2020)')
        ax.set_xlim(xlims[idx])
        ax.set_ylim(ylims[idx])
else:
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world_continents.plot(column=column, ax=ax, legend=True,
                          cmap='OrRd', edgecolor='black', missing_kwds={"color": "lightgrey", "label": "No data"},
                          scheme='natural_breaks', k=5, legend_kwds={'title': "Unemployment Rate (%)"})
    plt.title(f'Global Average Unemployment Rates by Continent{title_suffix} (2000 - 2020)')

# Remove the axis numbers
ax.set_xticks([])
ax.set_yticks([])

# Display the plot in Streamlit
st.pyplot(fig)










st.header('Unemployment Rate Variability within Continents')
st.write("This box plot illustrates the variability of unemployment rates within different continents for a selected year. The boxes represent the interquartile range of unemployment rates, while the lines and outliers show the range and anomalies. This graph is useful for comparing the spread and central tendency of unemployment rates within continents, highlighting disparities and commonalities in economic conditions.")

year2 = st.slider('Select the year', min_value=2000, max_value=2020, value=2020, key='slider_2')

# Filter data for the year 2020 and all genders, 15+
unemployment_df = unemployment[['unemp_country', f'unemp_{year2}']].dropna()
unemployment_df.columns = ['Country', 'Unemployment Rate']

# Merge with world data to get continent information
world_df = world[['ADMIN', 'CONTINENT']].merge(unemployment_df, how='inner', left_on='ADMIN', right_on='Country')

# Plotting the box plot
fig, ax = plt.subplots(figsize=(12, 6))

# Create the box plot
world_df.boxplot(column='Unemployment Rate', by='CONTINENT', ax=ax, grid=False,  medianprops=dict(color=plt.get_cmap('tab10')(1)))

# Set the title and labels
ax.set_title(f'Unemployment Rate Variability within Continents in {year2}')
ax.set_xlabel('Continent')
ax.set_ylabel('Unemployment Rate (%)')
plt.suptitle('')

# Display the plot in Streamlit
st.pyplot(fig)










st.header('Unemployment Trends Heatmap')
st.write("This heatmap tracks unemployment rate trends over two decades for each continent and the countries within a continent. The color intensity indicates the severity of unemployment rates. This visual tool allows to spot temporal patterns and shifts in unemployment, offering insights into long-term economic developments and cycles.")
continent_names_heatmap = ['Asia', 'Europe', 'North America', 'Oceania', 'South America']

# Add a selectbox to choose comparison type
comparison_type = st.selectbox(
    'Select the comparison type:',
    ('Continents', 'Countries within a Continent'),
    key='comparison_type'
)

if comparison_type == 'Continents':
    # Prepare data for heatmap comparing continents
    unemployment_with_continent = world[['ADMIN', 'CONTINENT']].merge(unemployment, how='right', left_on='ADMIN', right_on='unemp_country')
    unemployment_heatmap_data = unemployment_with_continent.drop(['ADMIN', 'unemp_country', 'unemp_average', 'unemp_average_2006_2020', 'unemp_average_2000_2017'], axis=1)
    unemployment_heatmap_data = unemployment_heatmap_data.melt(id_vars=['CONTINENT'], var_name='Year', value_name='Unemployment Rate')
    heatmap_data = unemployment_heatmap_data.pivot_table(index='CONTINENT', columns='Year', values='Unemployment Rate')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='coolwarm', ax=ax, cbar_kws={'label': 'Unemployment Rate (%)'})
    
    xticks_labels = [label.replace('unemp_', '') for label in heatmap_data.columns]
    ax.set_xticklabels(xticks_labels, rotation=45, ha='right')

    ax.set_title('Unemployment Rate Trends by Continent (2000-2020)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Continent')

    st.pyplot(fig)

else:
    # Add a selectbox to choose the specific continent
    continent_option = st.selectbox(
        'Select the continent to display:',
        continent_names_heatmap,
        key='continent_option'
    )

    # Prepare data for heatmap comparing countries within the selected continent
    selected_continent_data = world[world['CONTINENT'] == continent_option]
    unemployment_with_country = selected_continent_data[['ADMIN', 'CONTINENT']].merge(unemployment, how='right', left_on='ADMIN', right_on='unemp_country')

    # Filter out countries with NaN values between 2000 and 2020
    columns_to_check = [f'unemp_{year}' for year in range(2000, 2021)]
    unemployment_with_country = unemployment_with_country.dropna(subset=columns_to_check)

    unemployment_heatmap_data_country = unemployment_with_country.drop(['CONTINENT', 'unemp_country', 'unemp_average', 'unemp_average_2006_2020', 'unemp_average_2000_2017'], axis=1)
    unemployment_heatmap_data_country = unemployment_heatmap_data_country.melt(id_vars=['ADMIN'], var_name='Year', value_name='Unemployment Rate')
    heatmap_data_country = unemployment_heatmap_data_country.pivot_table(index='ADMIN', columns='Year', values='Unemployment Rate')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data_country, cmap='coolwarm', ax=ax, cbar_kws={'label': 'Unemployment Rate (%)'})
    
    xticks_labels_country = [label.replace('unemp_', '') for label in heatmap_data_country.columns]
    ax.set_xticklabels(xticks_labels_country, rotation=45, ha='right')

    ax.set_title(f'Unemployment Rate Trends in {continent_option} by Country (2000-2020)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Country')

    st.pyplot(fig)










st.header('Unemployment Trends by Country')
st.write("This line graph displays the unemployment rate trends for selected countries over a twenty-year period. By plotting the data points year by year, we can observe how unemployment rates have fluctuated within specific countries, identifying periods of economic growth or decline.")

# Select only the right columns
unemp_columns = [col for col in world.columns if col.startswith('unemp_')]
filtered_df = world[['ADMIN'] + unemp_columns]
columns_to_drop = filtered_df.columns[[1, -3, -2, -1]]
filtered_df = filtered_df.drop(columns=columns_to_drop)

# only include countries that have at least 3 datapoints
filtered_df = filtered_df[filtered_df.isnull().sum(axis=1) <= 18]

country_options = ['Fiji', 'Tanzania', 'Canada', 'USA', 'Kazakhstan', 'Indonesia', 'Argentina', 'Chile', 'Kenya', 'Dominican Republic', 'Russia', 'Bahamas', 'Norway', 'Timor-Leste', 'South Africa', 'Mexico', 'Uruguay', 'Brazil', 'Bolivia', 'Peru', 'Colombia', 'Panama', 'Costa Rica', 'Nicaragua', 'Honduras', 'El Salvador', 'Guatemala', 'Belize', 'Venezuela', 'Suriname', 'France', 'Ecuador', 'Puerto Rico', 'Jamaica', 'Zimbabwe', 'Botswana', 'Namibia', 'Senegal', 'Mali', 'Benin', 'Niger', 'Nigeria', 'Cameroon', 'Togo', 'Ghana', "Cote d'Ivoire", 'Liberia', 'Zambia', 'Malawi', 'Angola', 'Israel', 'Lebanon', 'Palestine', 'Tunisia', 'Algeria', 'Jordan', 'UAE', 'Qatar', 'Kuwait', 'Iraq', 'Vanuatu', 'Cambodia', 'Thailand', 'Myanmar', 'Vietnam', 'South Korea', 'Mongolia', 'India', 'Bangladesh', 'Bhutan', 'Pakistan', 'Afghanistan', 'Kyrgyz Republic', 'Iran', 'Syria', 'Armenia', 'Sweden', 'Belarus', 'Ukraine', 'Poland', 'Austria', 'Hungary', 'Moldova', 'Romania', 'Lithuania', 'Latvia', 'Estonia', 'Germany', 'Bulgaria', 'Greece', 'Turkey', 'Albania', 'Croatia', 'Switzerland', 'Luxembourg', 'Belgium', 'Netherlands', 'Portugal', 'Spain', 'Ireland', 'New Zealand', 'Australia', 'Sri Lanka', 'Taiwan', 'Italy', 'Denmark', 'UK', 'Iceland', 'Azerbaijan', 'Georgia', 'Philippines', 'Malaysia', 'Brunei', 'Slovenia', 'Finland', 'Slovak Republic', 'Czech Republic', 'Japan', 'Paraguay', 'Saudi Arabia', 'Cyprus', 'Morocco', 'Egypt', 'Uganda', 'Rwanda', 'Bosnia and Herzegovina', 'North Macedonia', 'Serbia', 'Montenegro', 'Kosovo', 'Trinidad and Tobago']

# Multiselect to choose multiple countries to highlight
highlight_countries = st.multiselect(
    'Select the countries to display:',
    country_options,
    default = 'Spain'
)

if highlight_countries:
    # Filter the DataFrame for the rows with the selected countries
    filtered_df = filtered_df[filtered_df['ADMIN'].isin(highlight_countries)]

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define a color map
    color_map = plt.get_cmap('tab10')

    # Loop through each selected country and plot its data
    for idx, country in enumerate(highlight_countries):
        country_df = filtered_df[filtered_df['ADMIN'] == country]
        # Transpose the DataFrame for plotting
        transposed_df = country_df.T
        transposed_df.columns = ['Unemployment Rate']
        transposed_df = transposed_df.iloc[1:]

        # Convert the unemployment rate column to numeric
        transposed_df['Unemployment Rate'] = pd.to_numeric(transposed_df['Unemployment Rate'], errors='coerce')

        # Convert the index to a more readable format
        transposed_df.index = transposed_df.index.str.replace(r'unemp_|u_\w+_\w+_', '', regex=True)

        # Interpolate the DataFrame to fill missing values for the line plot
        interpolated_df = transposed_df.interpolate()

        # Plot the interpolated line
        color = color_map(idx % 10)  # Get a color from the color map
        ax.plot(interpolated_df.index, interpolated_df['Unemployment Rate'], color=color, linewidth=2.5, label=country)
        # Plot the original data points
        ax.plot(transposed_df.index, transposed_df['Unemployment Rate'], 'o', color=color)

    ax.set_title('Unemployment Rate Trends by Country')
    ax.set_xlabel('Year')
    ax.set_ylabel('Unemployment Rate (%)')
    ax.set_xticks(range(len(transposed_df.index)))
    ax.set_xticklabels(transposed_df.index, rotation=45)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend()

    # Display the plot in a Streamlit app
    st.pyplot(fig)
else:
    st.write("Please select at least one country to display the plot.")
    
    
    







st.header('Unemployment by Gender and Age Group')
st.write("This bar chart breaks down the unemployment rates by gender and age group within a selected country for a specific year. The graph highlights differences in unemployment rates between males and females across various age categories. This detailed demographic analysis is important for understanding how unemployment affects different segments of the population, providing insights into labor market inequalities and targeted policy needs.")

countries = ['Canada', 'USA', 'Chile', 'Russia', 'South Africa', 'Mexico', 'Uruguay', 'Costa Rica', 'France', 'Israel', 'South Korea', 'Sweden', 'Poland', 'Austria', 'Germany', 'Bulgaria', 'Greece', 'Turkey', 'Switzerland', 'Netherlands', 'Portugal', 'Spain', 'New Zealand', 'Australia', 'Taiwan', 'Italy', 'Denmark', 'Finland', 'Slovak Republic', 'Czech Republic', 'Japan', 'Cyprus']
# Add a selectbox to choose the specific continent for country view
country = st.selectbox(
    'Select the country to display:',
    countries,
    key = 'select_country_1'
)

year = st.slider('Select the year', min_value=2000, max_value=2020, value=2020, key='slider_1')

# Extract the relevant columns for the bar graph
columns_to_plot = {
    'Female (15-24)': f'u_female_15_24_{year}',
    'Male (15-24)': f'u_male_15_24_{year}',
    'Female (25-54)': f'u_female_25_54_{year}',
    'Male (25-54)': f'u_male_25_54_{year}',
    'Female (55-64)': f'u_female_55_64_{year}',
    'Male (55-64)': f'u_male_55_64_{year}'
}

# Filter the data
df = world[world['ADMIN'] == country]

# Extract the values
values = [df[columns_to_plot[label]].values[0] for label in columns_to_plot.keys()]

# Define categories and subcategories
categories = ['15-24', '25-54', '55-64']
subcategories = ['Female', 'Male']

# Prepare the data for grouped bar chart
data = np.array(values).reshape(3, 2).T

# Create a grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Define the bar width
bar_width = 0.35

# Define positions for the bars
positions = np.arange(len(categories))

# Get the colors from the tab10 colormap
cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(len(subcategories))]

# Plot each subcategory with the appropriate color
for i, (subcategory, color) in enumerate(zip(subcategories, colors)):
    ax.bar(positions + i * bar_width, data[i], bar_width, label=subcategory, color=color)

# Set the title and labels
ax.set_title(f'Unemployment Rate in {country} in {year} by Gender and Age Group')
ax.set_xlabel('Age Group')
ax.set_ylabel('Unemployment Rate (%)')
ax.set_xticks(positions + bar_width / 2)
ax.set_xticklabels(categories)
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)