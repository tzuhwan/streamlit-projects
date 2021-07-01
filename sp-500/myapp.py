import pandas as pd
import base64
import seaborn as sns
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

st.title('S&P 500 App')

st.markdown("""
This app retrieves the lsit of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date).
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, yfinance
* **Data source: ** [Wikipedia](https://www.wikipedia.org)
""")

st.sidebar.header("User Input Features")


# scrap the s&p500 table using pandas
@st.cache
def load_data():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = pd.read_html(url, header=0)
    return html[0]


# load data into df
df = load_data()

#sector categories for user to choose from - side bar
sector = df.groupby("GICS Sector")
sorted_sector_unique = sorted(df["GICS Sector"].unique())
selected_sectors = st.sidebar.multiselect("Sector", sorted_sector_unique)

# Filtering data
df_selected_sector = df[df["GICS Sector"].isin(selected_sectors)]

# companies stock prices that user wants to see from sector
companies_in_selected_sector = sorted(df_selected_sector["Symbol"].unique())
selected_companies = st.sidebar.multiselect("Company",
                                            companies_in_selected_sector)

st.header("Display Companies in Selected Sector")
st.write("Data Dimension: " + str(df_selected_sector.shape[0]) + " rows and " +
         str(df_selected_sector.shape[1]) + " columns")
st.dataframe(df_selected_sector)


# Download data in csv format
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV file</a>'
    return href


st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

#selected companies' stock price data from yfinance


def downloadStockPrice(companies):
    if companies:
        return yf.download(
            tickers=list(selected_companies),
            period="ytd",
            interval="1d",
            group_by="ticker",
            auto_adjust="True",
            prepost="True",
            threads="True",
            proxy=None,
        )


def price_plot(symbol, data):
    if len(selected_companies) == 1:
        df = pd.DataFrame(data.Close)
    else:
        df = pd.DataFrame(data[symbol].Close)

    fig, ax = plt.subplots()
    df['Date'] = df.index
    plt.fill_between(df.Date, df.Close, color="skyblue", alpha=0.3)
    plt.plot(df.Date, df.Close, color="skyblue", alpha=0.3)
    plt.xticks(rotation=90)
    plt.title(symbol, fontweight="bold")
    plt.xlabel('Date', fontweight="bold")
    plt.ylabel("Closing Price", fontweight="bold")
    return st.pyplot(fig)


if st.button("Show Selected Companies Stock Prices"):
    data = downloadStockPrice(selected_companies)
    for company in selected_companies:
        price_plot(company, data)