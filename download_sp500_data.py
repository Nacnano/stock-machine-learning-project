# %%
import bs4 as bs
import requests
import yfinance as yf

# %%
resp = requests.get("http://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
soup = bs.BeautifulSoup(resp.text, "lxml")
table = soup.find("table", {"class": "wikitable sortable"})

tickers = []

for row in table.findAll("tr")[1:]:
    ticker = row.findAll("td")[0].text
    tickers.append(ticker)

tickers = [s.replace("\n", "") for s in tickers]
tickers.remove("CEG")
tickers.remove("BRK.B")
tickers.remove("BF.B")
tickers.sort()
tickers.append("SPX")

for ticker in tickers:
    data = yf.download(ticker)
    data.to_csv(f"data/{ticker}.csv")
