import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_sustainability_guide():
    url = "https://goodonyou.eco/material-guide/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    print(f"Connecting to {url}...")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print("Failed to reach the site.")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Good On You organizes materials in specific sections
    # We are looking for the 'Preferred' vs 'Avoid' labels
    materials_data = []
    
    # We find all headings which usually represent the material names
    for heading in soup.find_all(['h2', 'h3', 'strong']):
        name = heading.get_text().strip().lower()
        
        # Filter for actual material names (short strings)
        if 3 < len(name) < 25:
            # Simple logic: If it's in the first half of the page, it's usually 'Preferred'
            # (We will refine this later, but for now, let's get the names)
            materials_data.append(name)

    # Clean up the list (remove duplicates)
    unique_materials = list(set(materials_data))
    
    # Save to our data folder
    df_materials = pd.DataFrame(unique_materials, columns=['material_name'])
    df_materials.to_csv('data/scraped_materials.csv', index=False)
    
    print(f"✅ Success! Saved {len(unique_materials)} materials to data/scraped_materials.csv")
    return unique_materials

# Execute
if __name__ == "__main__":
    scrape_sustainability_guide()