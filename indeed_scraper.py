import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from datetime import datetime
import re

class IndeedScraper:
    def __init__(self):
        self.base_url = "https://www.indeed.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def get_job_listings(self, query="software engineer", location="United States", num_pages=10):
        all_jobs = []
        
        for page in range(num_pages):
            url = f"{self.base_url}/jobs?q={query}&l={location}&start={page*10}"
            try:
                response = self._make_request(url)
                if response:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    job_cards = soup.find_all('div', class_='job_seen_beacon')
                    
                    for job in job_cards:
                        job_data = self._extract_job_data(job)
                        if job_data:
                            all_jobs.append(job_data)
                    
                    # Respect rate limiting
                    time.sleep(random.uniform(1, 3))
                    
            except Exception as e:
                print(f"Error scraping page {page}: {str(e)}")
                continue
                
        return pd.DataFrame(all_jobs)
    
    def _make_request(self, url):
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            return None
    
    def _extract_job_data(self, job_card):
        try:
            title = job_card.find('h2', class_='jobTitle').text.strip()
            company = job_card.find('span', class_='companyName').text.strip()
            location = job_card.find('div', class_='companyLocation').text.strip()
            
            # Extract job description
            job_desc = job_card.find('div', class_='job-snippet')
            description = job_desc.text.strip() if job_desc else ""
            
            # Extract salary if available
            salary = job_card.find('div', class_='salary-snippet')
            salary = salary.text.strip() if salary else ""
            
            # Extract job type (full-time, contract, etc.)
            job_type = job_card.find('div', class_='metadata')
            job_type = job_type.text.strip() if job_type else ""
            
            # Extract posting date
            date = job_card.find('span', class_='date')
            date = date.text.strip() if date else ""
            
            return {
                'title': title,
                'company': company,
                'location': location,
                'description': description,
                'salary': salary,
                'job_type': job_type,
                'date': date,
                'scraped_date': datetime.now().strftime('%Y-%m-%d')
            }
        except Exception as e:
            print(f"Error extracting job data: {str(e)}")
            return None

def main():
    scraper = IndeedScraper()
    
    # Scrape software engineering jobs
    print("Starting job scraping...")
    jobs_df = scraper.get_job_listings(
        query="software engineer",
        location="United States",
        num_pages=10  # Adjust based on your needs
    )
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'indeed_jobs_{timestamp}.csv'
    jobs_df.to_csv(output_file, index=False)
    print(f"Scraped {len(jobs_df)} jobs and saved to {output_file}")

if __name__ == "__main__":
    main() 