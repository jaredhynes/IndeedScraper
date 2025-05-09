import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import random
from datetime import datetime
import re
import signal
import sys
import os

class IndeedScraper:
    def __init__(self):
        self.base_url = "https://www.indeed.com"
        self.setup_driver()
        self.jobs_data = []  # Store jobs in memory
        
        # Create data directory if it doesn't exist
        self.data_dir = "scraped_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created directory: {self.data_dir}")
        
        # Set up file paths for master files
        self.master_file = os.path.join(self.data_dir, "master_jobs.csv")
        self.backup_file = os.path.join(self.data_dir, "master_jobs_backup.csv")
        
        # Generate timestamp for this run's log
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.data_dir, f"scraping_log_{self.timestamp}.txt")
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"Scraping started at: {datetime.now()}\n")
            f.write(f"Master file: {self.master_file}\n")
            f.write(f"Backup file: {self.backup_file}\n")
            f.write("=" * 50 + "\n\n")
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
    def log_message(self, message):
        """Write message to log file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}\n"
        print(message)  # Also print to console
        with open(self.log_file, 'a') as f:
            f.write(log_message)
        
    def handle_shutdown(self, signum, frame):
        """Handle graceful shutdown when interrupted"""
        self.log_message("\nGracefully shutting down...")
        self.save_data()
        self.log_message(f"Data saved to {self.master_file}")
        self.driver.quit()
        sys.exit(0)
        
    def save_data(self):
        """Save current data to CSV, appending to master file"""
        if self.jobs_data:
            # Create new DataFrame with current jobs
            new_df = pd.DataFrame(self.jobs_data)
            
            # Add timestamp for when these jobs were scraped
            new_df['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # If master file exists, append to it
            if os.path.exists(self.master_file):
                try:
                    # Read existing data
                    existing_df = pd.read_csv(self.master_file)
                    
                    # Combine with new data
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    
                    # Remove any duplicates based on job title, company, and location
                    combined_df = combined_df.drop_duplicates(
                        subset=['title', 'company', 'location'],
                        keep='last'
                    )
                    
                    # Save to master file
                    combined_df.to_csv(self.master_file, index=False)
                    # Save backup copy
                    combined_df.to_csv(self.backup_file, index=False)
                    
                    self.log_message(f"Added {len(new_df)} new jobs to master file")
                    self.log_message(f"Total jobs in master file: {len(combined_df)}")
                    self.log_message(f"Backup saved to {self.backup_file}")
                    
                except Exception as e:
                    self.log_message(f"Error appending to master file: {str(e)}")
                    # If there's an error, save new data to a separate file
                    error_file = os.path.join(self.data_dir, f"error_save_{self.timestamp}.csv")
                    new_df.to_csv(error_file, index=False)
                    self.log_message(f"Saved new data to {error_file} due to error")
            else:
                # If master file doesn't exist, create it
                new_df.to_csv(self.master_file, index=False)
                new_df.to_csv(self.backup_file, index=False)
                self.log_message(f"Created new master file with {len(new_df)} jobs")
                self.log_message(f"Backup saved to {self.backup_file}")
            
            # Clear the jobs_data list after saving
            self.jobs_data = []
        
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
        
    def get_job_listings(self, query="software engineer", location="United States", num_pages=150):
        consecutive_empty_pages = 0
        max_empty_pages = 3  # Stop if we hit 3 empty pages in a row
        
        try:
            # Visit the main page first
            self.log_message("Visiting main page...")
            self.driver.get(self.base_url)
            
            # Wait for user to complete Cloudflare check
            self.log_message("\nPlease complete the Cloudflare check in the browser window.")
            self.log_message("Press Enter in this terminal when you're done...")
            input()
            
            # Add additional delay after user confirmation
            self.log_message("Waiting for page to fully load...")
            time.sleep(10)
            
            # Search for jobs
            self.log_message("Performing search...")
            search_url = f"{self.base_url}/jobs?q={query}&l={location}"
            self.driver.get(search_url)
            
            # Wait for user to complete any additional checks
            self.log_message("\nIf there's another security check, please complete it now.")
            self.log_message("Press Enter in this terminal when you're done...")
            input()
            
            # Add additional delay after user confirmation
            self.log_message("Waiting for search results to load...")
            time.sleep(10)
            
            current_page = 1
            while current_page <= num_pages:
                self.log_message(f"\nScraping page {current_page} of {num_pages}...")
                self.log_message(f"Current URL: {self.driver.current_url}")
                
                # Wait for job cards to load
                try:
                    WebDriverWait(self.driver, 20).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "job_seen_beacon"))
                    )
                except Exception as e:
                    self.log_message(f"Error waiting for job cards: {str(e)}")
                    self.log_message("Available classes on the page:")
                    elements = self.driver.find_elements(By.XPATH, "//*[@class]")
                    for element in elements[:10]:
                        self.log_message(f"Class: {element.get_attribute('class')}")
                    break
                
                # Get all job cards
                job_cards = self.driver.find_elements(By.CLASS_NAME, "job_seen_beacon")
                self.log_message(f"Found {len(job_cards)} job cards")
                
                if len(job_cards) == 0:
                    consecutive_empty_pages += 1
                    self.log_message(f"No job cards found on page {current_page}. Consecutive empty pages: {consecutive_empty_pages}")
                    if consecutive_empty_pages >= max_empty_pages:
                        self.log_message(f"Stopping after {consecutive_empty_pages} consecutive empty pages")
                        break
                else:
                    consecutive_empty_pages = 0  # Reset counter if we found jobs
                
                page_jobs = []  # Store jobs from current page
                for job in job_cards:
                    try:
                        job_data = self._extract_job_data(job)
                        if job_data:
                            page_jobs.append(job_data)
                    except Exception as e:
                        self.log_message(f"Error extracting job data: {str(e)}")
                        continue
                
                # Add page jobs to main list
                self.jobs_data.extend(page_jobs)
                self.log_message(f"Successfully scraped {len(page_jobs)} jobs from this page")
                self.log_message(f"Total jobs scraped so far: {len(self.jobs_data)}")
                
                # Save data after each page
                self.save_data()
                
                # Automatically continue to next page if not on last page
                if current_page < num_pages:
                    self.log_message(f"Waiting 5 seconds before moving to page {current_page + 1}...")
                    time.sleep(5)  # Wait 5 seconds between pages
                    
                    # Try to find and click next page button
                    try:
                        next_button = self.driver.find_element(By.CSS_SELECTOR, "[aria-label='Next Page']")
                        if next_button.is_enabled():
                            next_button.click()
                            time.sleep(3)  # Short wait after clicking
                        else:
                            self.log_message("No more pages available")
                            break
                    except Exception as e:
                        self.log_message(f"Error navigating to next page: {str(e)}")
                        self.log_message("Trying alternative navigation method...")
                        try:
                            # Try to find the next page number and click it
                            next_page_num = current_page + 1
                            next_page_button = self.driver.find_element(By.CSS_SELECTOR, f"a[data-testid='pagination-page-{next_page_num}']")
                            if next_page_button:
                                next_page_button.click()
                                time.sleep(3)
                            else:
                                self.log_message("Could not find next page button")
                                break
                        except:
                            self.log_message("Could not navigate to next page")
                            break
                
                current_page += 1
                    
        except Exception as e:
            self.log_message(f"Error during scraping: {str(e)}")
        finally:
            # Save any remaining data before quitting
            self.save_data()
            self.driver.quit()
            
        return pd.DataFrame(self.jobs_data)
    
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
            
            # Click on the job card to view details
            try:
                # Find and click the job title link
                job_link = job_card.find_element(By.CSS_SELECTOR, "a.jcs-JobTitle")
                job_link.click()
                
                # Wait for the description to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, "jobDescriptionText"))
                )
                
                # Get the full description
                description = self.driver.find_element(By.ID, "jobDescriptionText").text.strip()
                
                # Go back to the job listings
                self.driver.back()
                
                # Wait for the job listings to reload
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "job_seen_beacon"))
                )
                
                # Add a small delay to ensure the page is fully loaded
                time.sleep(2)
            except Exception as e:
                self.log_message(f"Error getting job description: {str(e)}")
                description = ""
            
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
                self.log_message("\nDebug - Job card HTML:")
                self.log_message(job_card.get_attribute('outerHTML'))
                self.log_message("\nFound values:")
                self.log_message(f"Title: {title}")
                self.log_message(f"Company: {company}")
                self.log_message(f"Location: {location}")
                self.log_message(f"Salary: {salary}")
                self.log_message(f"Job Type: {job_type}")
                self.log_message(f"Description: {description}")
                self.log_message(f"Date: {date}")
            
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
            self.log_message(f"Error extracting job data: {str(e)}")
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
        scraper.log_message("Starting job scraping...")
        jobs_df = scraper.get_job_listings(
            query="software engineer",
            location="United States",
            num_pages=150
        )
        
        scraper.log_message(f"\nScraping completed. Total jobs scraped in this session: {len(jobs_df)}")
        scraper.log_message(f"Master file: {scraper.master_file}")
        scraper.log_message(f"Backup file: {scraper.backup_file}")
        scraper.log_message(f"Log file: {scraper.log_file}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        # The scraper's handle_shutdown will be called automatically

if __name__ == "__main__":
    main() 