import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("NBA Player Stats Explorer")

st.write("""
This app performs simple webscraping of NBA player stats data!
* **Python libraries:** base64, pandas, streamlit 
* **Datasource:** [Basketball-reference.com](https://www.basketball-reference.com/)
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox("Year", list(reversed(range(1950, 2022))))


# web-scraping of NBA player stats
@st.cache
def load_data(year):
    # url = "https://www.basketball-reference.com/leagues/NBA_" + str(
    #     year) + "_per_game.html"
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(
        year) + "_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)
    raw = raw.fillna(0)
    playerStats = raw.drop(['Rk'], axis=1)
    return playerStats


playerStats = load_data(selected_year)

# Sidebar - Position Selection
sorted_unique_team = sorted(playerStats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team,
                                       sorted_unique_team)

#Sidebar - Position Selection
sorted_unique_pos = sorted(playerStats.Pos.unique())
selected_pos = st.sidebar.multiselect("Position", sorted_unique_pos,
                                      sorted_unique_pos)

# Filtering data
df_selected_team = playerStats[(playerStats.Tm.isin(selected_team))
                               & (playerStats.Pos.isin(selected_pos))]

st.header("Displayer Player Stats of Selected Team(s)")
st.write("Data Dimension: " + str(df_selected_team.shape[0]) +
         " players and " + str(df_selected_team.shape[1]) + " attributes.")
st.dataframe(df_selected_team)


# Download NBA player stats data
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode(
    )  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV file</a>'
    return href


st.markdown(get_table_download_link(df_selected_team), unsafe_allow_html=True)

# heatmap
if st.button("Intercorrelation Heatmap"):
    st.header("Intercorrelation Matrix Heatmap")
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)

    st.pyplot(f)