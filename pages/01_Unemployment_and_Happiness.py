import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Function to load data with caching
@st.cache_data
def load_data():
    # Load data
    world = gpd.read_file('data/110m_cultural/ne_110m_admin_0_countries.shp')
    unemployment = pd.read_csv('data/aged_15plus_unemployment_rate_percent.csv')
    happiness = pd.read_csv('data/hapiscore_whr.csv')
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
    happiness['country'] = happiness['country'].str.strip()
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
    happiness = happiness.drop(happiness.columns[1:2], axis=1)
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
    happiness['average'] = happiness.iloc[:, 1:].mean(axis=1)
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
    happiness_2006_2020 = happiness.drop(happiness.columns[16:], axis=1)
    unemployment['average_2006_2020'] = unemployment_2006_2020.iloc[:, 1:].mean(axis=1)
    happiness['average_2006_2020'] = happiness_2006_2020.iloc[:, 1:].mean(axis=1)
    unemployment_2000_2017 = unemployment.drop(unemployment.columns[19:], axis=1)
    unemployment['average_2000_2017'] = unemployment_2000_2017.iloc[:, 1:].mean(axis=1)

    # Applying country mapping to the world DataFrame
    country_mapping = { 'Guinea-Bissau': 'Guinea-Bissau', 'eSwatini': 'Eswatini', 'Republic of the Congo': 'Congo, Rep.', 'Somalia': 'Somalia', 'Central African Republic': 'Central African Republic', 'Eritrea': 'Eritrea', 'Republic of Serbia': 'Serbia', 'Czechia': 'Czech Republic', 'Democratic Republic of the Congo': 'Congo, Dem. Rep.', 'North Korea': 'North Korea', 'Somaliland': 'Somaliland', 'The Bahamas': 'Bahamas', 'United Arab Emirates': 'UAE', 'French Southern and Antarctic Lands': 'French Southern and Antarctic Lands', 'Laos': 'Lao', 'United Kingdom': 'UK', 'Slovakia': 'Slovak Republic', 'Ivory Coast': "Cote d'Ivoire", 'Antarctica': 'Antarctica', 'United States of America': 'USA', 'East Timor': 'Timor-Leste', 'Northern Cyprus': 'Northern Cyprus', 'United Republic of Tanzania': 'Tanzania','Western Sahara': 'Western Sahara', 'Kyrgyzstan': 'Kyrgyz Republic', 'Falkland Islands': 'Falkland Is (Malvinas)' }
    world['ADMIN'] = world['ADMIN'].replace(country_mapping)

    # Add prefix to columns
    unemployment = unemployment.add_prefix('unemp_')
    happiness = happiness.add_prefix('happi_')
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
    world = world.merge(happiness, how='left', left_on='ADMIN', right_on='happi_country')
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
    happiness_continents = world.dropna(subset=['happi_average', 'happi_average_2006_2020'])
    happiness_continents = happiness_continents.groupby('CONTINENT')[['happi_average', 'happi_average_2006_2020']].mean()
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
    world_continents = world_continents.drop(columns=['unemp_average', 'unemp_average_2006_2020', 'unemp_average_2000_2017', 'happi_average_2006_2020',
                                                    'happi_average', 'u_15_24_average', 'u_25_54_average', 'u_55_64_average',
                                                    'u_female_15plus_average', 'u_male_15plus_average', 'u_female_15_24_average', 'u_female_25_54_average', 'u_female_55_64_average',
                                                    'u_male_15_24_average', 'u_male_25_54_average', 'u_male_55_64_average'])
    world_continents = world_continents.merge(unemployment_continents, on='CONTINENT')
    world_continents = world_continents.merge(happiness_continents, on='CONTINENT')
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
    xlims = [(-20, 60), (20, 150), (-30, 40), (-170, -10), (100, 180), (-90, -30)]
    ylims = [(-40, 40), (-10, 60), (35, 75), (0, 80), (-50, 0), (-60, 20)]

    return (world, world_continents, unemployment, happiness, unemployment_15_24, unemployment_25_54, 
            unemployment_55_64, unemployment_female, unemployment_male, unemployment_female_15_24, 
            unemployment_female_25_54, unemployment_female_55_64, unemployment_male_15_24, 
            unemployment_male_25_54, unemployment_male_55_64,
            oceania, africa, north_america, asia, south_america, europe, continents, continent_names, xlims, ylims)

(world, world_continents, unemployment, happiness, unemployment_15_24, unemployment_25_54, unemployment_55_64, unemployment_female, unemployment_male, unemployment_female_15_24, unemployment_female_25_54, unemployment_female_55_64, unemployment_male_15_24, unemployment_male_25_54, unemployment_male_55_64, oceania, africa, north_america, asia, south_america, europe, continents, continent_names, xlims, ylims) = load_data()










# Streamlit page
st.title('Unemployment and Happiness')
st.write("Investigate the relationship between unemployment rates and happiness scores globally and over time. Examine how economic conditions correlate with the well-being of populations across various countries and historical periods.")

st.header('Global and Continental Happiness')
st.write("This map illustrates the average happiness scores around the world over a fourteen-year period. Different colors indicate various ranges of happiness levels. This visualization is essential for understanding the global distribution of happiness and recognizing geographical patterns in well-being.")

# Add a radio button to choose the view type
view_type = st.radio(
    'Select the view type:',
    ('Country', 'Continent')
)

# Logic to display the map based on view type
if view_type == 'Country':
    # Add a selectbox to choose the specific continent for country view
    continent_option = st.selectbox(
        'Select the continent to display:',
        ('Global',) + tuple(continent_names)
    )

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    if continent_option == 'Global':
        world.plot(column='happi_average_2006_2020', ax=ax, legend=True,
                   cmap='OrRd', edgecolor='black', missing_kwds={"color": "lightgrey", "label": "No data"},
                   scheme='natural_breaks', k=5, legend_kwds={'title': "Happiness Score (WHR)"})
        plt.title('Global Average Happiness Scores by Country (2006 - 2020)')
    else:
        # Find the index of the selected continent
        idx = continent_names.index(continent_option)
        continent = continents[idx]
        continent.plot(column='happi_average_2006_2020', ax=ax, legend=True,
                       cmap='OrRd', edgecolor='black', missing_kwds={"color": "lightgrey", "label": "No data"},
                       scheme='natural_breaks', k=5, legend_kwds={'title': "Happiness Score (WHR)"})
        plt.title(f'Average Happiness Scores in {continent_option} by Country (2006 - 2020)')
        ax.set_xlim(xlims[idx])
        ax.set_ylim(ylims[idx])
else:
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world_continents.plot(column='happi_average_2006_2020', ax=ax, legend=True,
                          cmap='OrRd', edgecolor='black', missing_kwds={"color": "lightgrey", "label": "No data"},
                          scheme='natural_breaks', k=5, legend_kwds={'title': "Happiness Score (WHR)"})
    plt.title('Global Average Happiness Scores by Continent (2006 - 2020)')

# Remove the axis numbers
ax.set_xticks([])
ax.set_yticks([])

# Display the plot in Streamlit
st.pyplot(fig)










st.header('Unemployment vs. Happiness Across Countries')
st.write("This scatter plot examines the relationship between unemployment rates and happiness scores for various countries in a given year. Each point on the graph represents a country, with its position reflecting its unemployment rate and happiness score. This graph is valuable for exploring how economic conditions, specifically unemployment, might correlate with the overall happiness of a country's population.")

year = st.slider('Select the year', min_value=2006, max_value=2020, value=2020)

# Create scatter plot based on selected year
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(world[f'unemp_{year}'], world[f'happi_{year}'], color='#FF9745')
ax.set_title(f'Unemployment Rates (Aged 15+) vs. Happiness Scores in {year} Across Countries')
ax.set_xlabel('Unemployment Rate (%)')
ax.set_ylabel('Happiness Score (WHR)')
ax.grid(True)

# Display the plot in the Streamlit app
st.pyplot(fig)










# Streamlit app
st.header('Unemployment vs. Happiness Over Time')
st.write("This scatter plot focuses on the relationship between unemployment rates and happiness scores over time within a selected country. Each point represents a different year, showing how changes in unemployment rates have corresponded with fluctuations in happiness scores. This graph is useful for analyzing long-term trends and understanding how economic factors may impact the well-being of a nation's citizens over an extended period.")

country_names = ['Canada', 'USA', 'Kazakhstan', 'Uzbekistan', 'Indonesia', 'Argentina', 'Chile', 'Dominican Republic', 'Russia', 'South Africa', 'Mexico', 'Uruguay', 'Brazil', 'Bolivia', 'Peru', 'Colombia', 'Panama', 'Costa Rica', 'Honduras', 'El Salvador', 'France', 'Ecuador', 'Israel', 'Palestine', 'Jordan', 'Thailand', 'South Korea', 'Kyrgyz Republic', 'Sweden', 'Ukraine', 'Poland', 'Austria', 'Hungary', 'Moldova', 'Romania', 'Lithuania', 'Latvia', 'Estonia', 'Germany', 'Greece', 'Turkey', 'Albania', 'Croatia', 'Belgium', 'Netherlands', 'Portugal', 'Spain', 'Ireland', 'New Zealand', 'Australia', 'Sri Lanka', 'Taiwan', 'Italy', 'Denmark', 'UK', 'Georgia', 'Philippines', 'Malaysia', 'Slovenia', 'Finland', 'Japan', 'Paraguay', 'Saudi Arabia', 'Cyprus', 'Egypt', 'North Macedonia', 'Serbia']

# Add a selectbox to choose the specific continent for country view
country_option = st.selectbox(
    'Select the country to display:',
    country_names
)
    
filtered_df = world[world['ADMIN'] == country_option]
columns_to_keep = ['ADMIN'] + [col for col in filtered_df.columns if col.startswith('unemp_')]
filtered_df = filtered_df[columns_to_keep]
filtered_df = filtered_df.iloc[:, 7:-3]
transposed_df = filtered_df.transpose()
transposed_df.columns = transposed_df.iloc[0]
unemp = transposed_df[1:]

filtered_df = world[world['ADMIN'] == country_option]
columns_to_keep = ['ADMIN'] + [col for col in filtered_df.columns if col.startswith('happi_')]
filtered_df = filtered_df[columns_to_keep]
filtered_df = filtered_df.iloc[:, 1:-4]
transposed_df = filtered_df.transpose()
transposed_df.columns = transposed_df.iloc[0]
happi = transposed_df[1:]

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(unemp, happi, color='#FF9745')
ax.set_title(f'Unemployment Rates (Aged 15+) vs. Happiness Scores in {country_option} Over Time')
ax.set_xlabel('Unemployment Rate (%)')
ax.set_ylabel('Happiness Score (WHR)')
ax.grid(True)
st.pyplot(fig)