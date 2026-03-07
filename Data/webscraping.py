import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import html, json

all_headlines = []
headers = {"User-Agent": "Mozilla/5.0"}

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.thehindu.com/news/national/"
}

all_headlines = []

# ----------- MAIN PAGES (1–9) -----------
base_url = "https://www.thehindu.com/news/national/?page="

for i in range(1, 10):   # pages 1–9
    url = base_url + str(i)
    print("Scraping main page:", url)

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup.select("h3.title.big a"):
        text = tag.get_text(strip=True)
        if len(text) > 20:
            all_headlines.append(text)

    time.sleep(2)


# ----------- AJAX FRAGMENT PAGES (10+) -----------
fragment_url = "https://www.thehindu.com/news/national/fragment/showmoredesked?page="

for i in range(10, 251):   # pages after 9
    url = fragment_url + str(i)
    print("Scraping fragment page:", url)

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup.select("h3.title.big a"):
        text = tag.get_text(strip=True)
        if len(text) > 20:
            all_headlines.append(text)

    time.sleep(2)


base_url = "https://indianexpress.com/section/political-pulse/"

for i in range(1, 251):

    if i == 1:
        url = base_url
    else:
        url = base_url + f"page/{i}/"

    print("Scraping:", url)

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find ALL JSON-LD scripts
    scripts = soup.find_all("script", type="application/ld+json")

    for script in scripts:
        try:
            data = json.loads(script.string)

            # We only want the ItemList one
            if isinstance(data, dict) and data.get("@type") == "ItemList":

                for item in data.get("itemListElement", []):
                    headline = item.get("name")
                    if headline:
                        all_headlines.append(headline)

        except Exception:
            continue

    time.sleep(2)

all_headlines = [html.unescape(h) for h in all_headlines]
all_headlines = list(set(all_headlines))
print("Total unique headlines:", len(all_headlines))

df = pd.DataFrame({
    "headline": all_headlines,
})

df.to_csv("dataset.csv", index=False)