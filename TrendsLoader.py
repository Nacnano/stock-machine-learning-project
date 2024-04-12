from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# TODO: Find a way to login using google account (need two factor auth) 
def download_google_trends_data(search_terms):
    driver = webdriver.Chrome()
    
    try:
        for term in search_terms:
            driver.get(f'https://trends.google.com/trends/explore?date=all&q={term}')
            
            download_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '/html/body/div[3]/div[2]/div/md-content/div/div/div[1]/trends-widget/ng-include/widget/div/div/div/widget-actions/div/button[1]'))
            )
            
            download_button.click()
            print(f"CSV data for '{term}' downloaded successfully.")
            
            time.sleep(1) 
    finally:
        driver.quit()

search_terms = ["iphone", "ios", "andriod"]
download_google_trends_data(search_terms)
