import requests
from bs4 import BeautifulSoup
import time
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RozeeScraper:
    def __init__(self, base_url="https://www.rozee.pk/job/jsearch/q/all"):
        self.base_url = base_url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
    def fetch_page(self, page_number):
        url = f"{self.base_url}?page={page_number}"
        try:
            # Respect rate limits and add random delay
            time.sleep(random.uniform(1.0, 3.0))
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            html = response.text

            # Rozee may return challenge/skeleton markup to bots instead of real jobs.
            if "chlng=y" in html or "panel-effect" in html:
                logging.warning(
                    "Rozee returned challenge/skeleton content on page %s; skipping page.",
                    page_number
                )
                return None

            return response.content
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching page {page_number}: {e}")
            return None

    def parse_jobs(self, html_content):
        if not html_content:
            return []
            
        soup = BeautifulSoup(html_content, 'html.parser')
        jobs = []
        
        # Note: The actual HTML structure needs to be verified based on Rozee.pk's current DOM
        job_cards = soup.find_all('div', class_='job') # Adjust class based on actual structure
        
        for card in job_cards:
            try:
                title = card.find('h3').text.strip() if card.find('h3') else None
                company = card.find('div', class_='cname').text.strip() if card.find('div', class_='cname') else None
                location = card.find('div', class_='city').text.strip() if card.find('div', class_='city') else None
                
                # Keep only complete rows so downstream dedup/load remains valid.
                if title and company and location:
                    jobs.append({
                        "title": title,
                        "company": company,
                        "location": location,
                        "source": "Rozee.pk"
                    })
            except Exception as e:
                logging.warning(f"Failed to parse a job card: {e}")
                
        return jobs

    def scrape(self, num_pages=5):
        all_jobs = []
        for page in range(1, num_pages + 1):
            logging.info(f"Scraping page {page}...")
            html = self.fetch_page(page)
            jobs = self.parse_jobs(html)
            all_jobs.extend(jobs)
            
        logging.info(f"Successfully scraped {len(all_jobs)} jobs.")
        return all_jobs

if __name__ == "__main__":
    scraper = RozeeScraper()
    scraped_data = scraper.scrape(num_pages=2)
    # In a real scenario, this data would be returned to the Prefect flow
    # and saved to a database or intermediate storage.
    print(f"Sample jobs: {scraped_data[:2]}")
