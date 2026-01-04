import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

#Configurations
TARGET_URL = "https://www.propertyfinder.ae/en/rent/properties-for-rent.html"

#Creating user agent to mimic browser behavior
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Referer": "https://www.google.com/"
}

def get_data():
    listings_data = []

    #Scraping  first 10 pages
    for page in range(1, 11):
        print(f"Scraping page {page}...")

        #Building URL for each page
        url = f"{TARGET_URL}?page={page}"

        try:
            response = requests.get(url, headers=headers, timeout=10)

            #Checking if request was blocked
            if response.status_code != 200:
                print(f"Error on loading page {page}: Staus Code {response.status_code}")
                break

            soup = BeautifulSoup(response.content, "html.parser")

            #Fetching cards
            cards = soup.find_all("article", attrs={"data-testid": "property-card"})
            print(f"Found{len(cards)} listings on page {page}.")

            if len(cards) == 0:
                print("DEBUG: No cards found. Saving HTML to debug.html for inspection.")
                with open("debug.html", "w", encoding="utf-8") as f:
                    f.write(soup.prettify())
                break

            for card in cards:
                try:
                    #Extracting data
                    title_tag = card.find("h2") or card.find("h3")
                    title = title_tag.text.strip() if title_tag else "N/A"

                    price_tag = card.find(attrs={"data-testid": "property-card-price"})
                    # Clean the price text (remove "AED/year" and commas)
                    price = price_tag.text.strip().replace("AED/year", "").replace(",", "").strip() if price_tag else "N/A"

                    loc_tag = card.find(attrs={"data-testid": "property-card-location"})
                    location = loc_tag.text.strip() if loc_tag else "N/A"

                    # 2. Detailed Specs (The "Gold" data)
                    type_tag = card.find(attrs={"data-testid": "property-card-type"})
                    prop_type = type_tag.text.strip() if type_tag else "N/A"

                    bed_tag = card.find(attrs={"data-testid": "property-card-spec-bedroom"})
                    bedrooms = bed_tag.text.strip() if bed_tag else "0"

                    bath_tag = card.find(attrs={"data-testid": "property-card-spec-bathroom"})
                    bathrooms = bath_tag.text.strip() if bath_tag else "0"

                    area_tag = card.find(attrs={"data-testid": "property-card-spec-area"})
                    # Clean area text (remove "sqft")
                    area = area_tag.text.strip().lower().replace("sqft", "").replace(",", "").strip() if area_tag else "0"

                    # Store in list
                    listings_data.append({
                        "Title": title,
                        "Price_AED": price,
                        "Location": location,
                        "Type": prop_type,
                        "Bedrooms": bedrooms,
                        "Bathrooms": bathrooms,
                        "Area_SqFt": area
                    })
                except Exception as e:
                    print(f"Error extracting data from a card: {e}")
                    continue
            
            #Random sleep to mimic human behavior
            sleep_time = random.uniform(3,6)
            print(f"Sleeping for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
    
    #Saving data to CSV
    if listings_data:
        df = pd.DataFrame(listings_data)
        df.to_csv("uae_rents.csv", index=False)
        print(f"\n Scraping completed. {len(listings_data)} listings saved to uae_rents.csv")
        print(df.head())
    else:
        print("Failed to retrieve data. Check debug.html for more details.")

if __name__ == "__main__":
    get_data()