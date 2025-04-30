import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import random
from datetime import datetime

class IndeedScraper:
    def __init__(self):
        self.base_url = "https://www.indeed.com"
        self.setup_driver()
        
    def setup_driver(self):
        try:
            options = uc.ChromeOptions()
            options.add_argument("--start-maximized")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-extensions")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            # Set the Chrome version explicitly
            self.driver = uc.Chrome(
                options=options,
                version_main=136
            )
        except Exception as e:
            print(f"Error setting up Chrome driver: {str(e)}")
            print("Please make sure Chrome is installed and up to date.")
            raise
        
    def get_job_listings(self, query="software engineer", location="United States", num_pages=10):
        all_jobs = []
        
        try:
            # Visit the main page first
            print("Visiting main page...")
            self.driver.get(self.base_url)
            
            # Wait for user to complete Cloudflare check
            print("\nPlease complete the Cloudflare check in the browser window.")
            print("Press Enter in this terminal when you're done...")
            input()
            
            # Add additional delay after user confirmation
            print("Waiting for page to fully load...")
            time.sleep(10)
            
            # Search for jobs
            print("Performing search...")
            search_url = f"{self.base_url}/jobs?q={query}&l={location}"
            self.driver.get(search_url)
            
            # Wait for user to complete any additional checks
            print("\nIf there's another security check, please complete it now.")
            print("Press Enter in this terminal when you're done...")
            input()
            
            # Add additional delay after user confirmation
            print("Waiting for search results to load...")
            time.sleep(10)
            
            current_page = 1
            while current_page <= num_pages:
                print(f"\nScraping page {current_page} of {num_pages}...")
                print("Current URL:", self.driver.current_url)
                
                # Wait for job cards to load
                try:
                    WebDriverWait(self.driver, 20).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "job_seen_beacon"))
                    )
                except Exception as e:
                    print(f"Error waiting for job cards: {str(e)}")
                    print("Available classes on the page:")
                    elements = self.driver.find_elements(By.XPATH, "//*[@class]")
                    for element in elements[:10]:
                        print(f"Class: {element.get_attribute('class')}")
                    break
                
                # Get all job cards
                job_cards = self.driver.find_elements(By.CLASS_NAME, "job_seen_beacon")
                print(f"Found {len(job_cards)} job cards")
                
                for job in job_cards:
                    try:
                        job_data = self._extract_job_data(job)
                        if job_data:
                            all_jobs.append(job_data)
                    except Exception as e:
                        print(f"Error extracting job data: {str(e)}")
                        continue
                
                print(f"Successfully scraped {len(all_jobs)} jobs so far")
                
                # Ask user if they want to continue to next page
                if current_page < num_pages:
                    print("\nPress Enter to continue to next page, or type 'q' to quit...")
                    user_input = input().strip().lower()
                    if user_input == 'q':
                        print("Quitting at user request...")
                        break
                    
                    # Try to find and click next page button
                    try:
                        next_button = self.driver.find_element(By.CSS_SELECTOR, "[aria-label='Next Page']")
                        if next_button.is_enabled():
                            next_button.click()
                            time.sleep(3)  # Short wait after clicking
                        else:
                            print("No more pages available")
                            break
                    except:
                        print("Could not find next page button")
                        break
                
                current_page += 1
                    
        except Exception as e:
            print(f"Error during scraping: {str(e)}")
        finally:
            self.driver.quit()
            
        return pd.DataFrame(all_jobs)
    
    def _extract_job_data(self, job_card):
        try:
            # Try different selectors for each field
            title = self._find_element_text(job_card, [
                "h2.jobTitle",
                ".jobTitle",
                "h2[class*='jobTitle']",
                "a.jcs-JobTitle"
            ])
            
            company = self._find_element_text(job_card, [
                "span[data-testid='company-name']",
                "span[class*='company']"
            ])
            
            location = self._find_element_text(job_card, [
                "div[data-testid='text-location']",
                "div[class*='location']"
            ])
            
            # Get salary - looking at the exact structure from debug
            salary = ""
            try:
                salary_div = job_card.find_element(By.CSS_SELECTOR, "div.css-by2xwt")
                salary_amount = salary_div.find_element(By.CSS_SELECTOR, "h2.css-1rqpxry").text.strip()
                salary_period = salary_div.find_element(By.CSS_SELECTOR, "span.css-18s3co2").text.strip()
                salary = f"{salary_amount} {salary_period}"
            except:
                pass
            
            # Get job type/benefits - looking at the exact structure from debug
            job_type = ""
            try:
                metadata_container = job_card.find_element(By.CSS_SELECTOR, "ul.metadataContainer")
                benefits = metadata_container.find_elements(By.CSS_SELECTOR, "div.css-18z4q2i")
                job_type = ", ".join([benefit.text.strip() for benefit in benefits])
            except:
                pass
            
            # Get job snippet (short description)
            description = self._find_element_text(job_card, [
                "div[class*='job-snippet']",
                "div[class*='snippet']",
                "div[class*='jobDescriptionText']"
            ])
            
            # For date - looking at the exact structure from debug
            date = ""
            try:
                timing_div = job_card.find_element(By.CSS_SELECTOR, "div[data-testid='timing-attribute']")
                # The date is usually in a separate div or span within the timing attribute
                date_elements = timing_div.find_elements(By.CSS_SELECTOR, "div, span")
                for element in date_elements:
                    text = element.text.strip()
                    if text and text != company and text != location:
                        date = text
                        break
            except:
                pass
            
            # Debug: Print the raw HTML of the job card if we're missing data
            if not all([title, company, location]):
                print("\nDebug - Job card HTML:")
                print(job_card.get_attribute('outerHTML'))
                print("\nFound values:")
                print(f"Title: {title}")
                print(f"Company: {company}")
                print(f"Location: {location}")
                print(f"Salary: {salary}")
                print(f"Job Type: {job_type}")
                print(f"Description: {description}")
                print(f"Date: {date}")
            
            return {
                'title': title,
                'company': company,
                'location': location,
                'salary': salary,
                'job_type': job_type,
                'description': description,
                'date': date,
                'scraped_date': datetime.now().strftime('%Y-%m-%d')
            }
        except Exception as e:
            print(f"Error extracting job data: {str(e)}")
            return None
    
    def _find_element_text(self, element, selectors):
        for selector in selectors:
            try:
                found = element.find_element(By.CSS_SELECTOR, selector)
                return found.text.strip()
            except:
                continue
        return ""

def main():
    try:
        scraper = IndeedScraper()
        
        # Scrape software engineering jobs
        print("Starting job scraping...")
        jobs_df = scraper.get_job_listings(
            query="software engineer",
            location="United States",
            num_pages=10  # You can set this to a higher number since you control navigation
        )
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'indeed_jobs_{timestamp}.csv'
        jobs_df.to_csv(output_file, index=False)
        print(f"Scraped {len(jobs_df)} jobs and saved to {output_file}")
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 