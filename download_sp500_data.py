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

tickers.sort()
tickers.append('^GSPC')


tickers_file_path = "tickers_sp500.txt"
with open(tickers_file_path, "w") as f:
    for ticker in tickers:
        f.write(ticker + "\n")

# Save to local
for ticker in tickers:
    data = yf.download(ticker)
    data.to_csv(f"data/{ticker}.csv")
