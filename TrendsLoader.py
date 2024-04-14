from pytrends.request import TrendReq
import os

def download_google_trends_data(search_terms, root_dir="./data/Trends"):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    pytrends = TrendReq(hl='en-US', tz=360)
    
    for term in search_terms:
        pytrends.build_payload(kw_list=[term], timeframe='all')
        data = pytrends.interest_over_time()
        data.to_csv(f"{root_dir}/{term}.csv")
        print(f"CSV data for '{term}' downloaded successfully.")

# Examples search terms
search_terms = ["ios", "andriod", "iphone"]
download_google_trends_data(search_terms)
