import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib.parse

def fetch_live_jobs(role, location="India"):
    encoded_role = urllib.parse.quote(role)
    url = f"https://www.linkedin.com/jobs/search?keywords={encoded_role}&location={location}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        job_cards = soup.find_all('div', class_='base-card')
        
        jobs = []
        for card in job_cards[:5]:
            title = card.find('h3', class_='base-search-card__title').text.strip()
            company = card.find('h4', class_='base-search-card__subtitle').text.strip()
            link = card.find('a', class_='base-card__full-link')['href']
            jobs.append({"Title": title, "Company": company, "URL": link})
        return pd.DataFrame(jobs)
    except:
        return pd.DataFrame()