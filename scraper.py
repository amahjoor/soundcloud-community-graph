import os
import time
import json
import random
import re
import numpy as np
import pandas as pd
import scipy.sparse
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.safari.options import Options as SafariOptions

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class SoundCloudScraper:
    def __init__(self, data_dir="./data", max_items=500, chrome_path=None, browser="safari"):
        """
        Initialize the SoundCloud scraper.
        
        Args:
            data_dir: Directory to store data
            max_items: Maximum number of items to collect per interaction type
            chrome_path: Optional path to Chrome binary
            browser: Browser to use ('safari' or 'chrome')
        """
        self.data_dir = data_dir
        self.max_items = max_items
        self.chrome_path = chrome_path
        self.browser = browser.lower()
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize data structures
        self.seeds_file = os.path.join(data_dir, "seeds.csv")
        self.remaining_seeds_file = os.path.join(data_dir, "remaining_seeds.csv")
        self.matrix_file = os.path.join(data_dir, "sparse_matrix.npz")
        self.tracker_file = os.path.join(data_dir, "tracker.json")
        
        self._init_driver()
        self._load_or_init_data()
    
    def _init_driver(self):
        """Initialize the WebDriver with appropriate options."""
        try:
            if self.browser == "safari":
                print("Initializing Safari WebDriver...")
                # Safari is built-in on macOS and doesn't need a separate driver installation
                safari_options = SafariOptions()
                self.driver = webdriver.Safari(options=safari_options)
                print("Safari WebDriver initialized successfully")
            else:
                print("Initializing Chrome WebDriver...")
                chrome_options = Options()
                chrome_options.add_argument("--disable-notifications")
                chrome_options.add_argument("--disable-infobars")
                chrome_options.add_argument("--mute-audio")
                
                # If chrome_path was provided, use it
                if self.chrome_path and os.path.exists(self.chrome_path):
                    print(f"Using provided Chrome binary: {self.chrome_path}")
                    chrome_options.binary_location = self.chrome_path
                else:
                    # Try to find Chrome in common locations for macOS
                    chrome_paths = [
                        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                        "/Applications/Chrome.app/Contents/MacOS/Chrome",
                        # Add other common paths if needed
                    ]
                    
                    for path in chrome_paths:
                        if os.path.exists(path):
                            print(f"Found Chrome at: {path}")
                            chrome_options.binary_location = path
                            break
                
                self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
                print("Chrome WebDriver initialized successfully")
            
            self.driver.implicitly_wait(10)
        except Exception as e:
            print(f"Error initializing WebDriver: {e}")
            print("\nTroubleshooting suggestions:")
            print("1. If using Chrome, ensure Google Chrome is installed on your system")
            print("2. If using Safari, ensure Safari's developer menu is enabled:")
            print("   - Open Safari, go to Preferences > Advanced")
            print("   - Check 'Show Develop menu in menu bar'")
            print("   - In the Develop menu, enable 'Allow Remote Automation'")
            print("\nOr try specifying a different browser:")
            print("   python main.py scrape --browser=safari")
            raise
    
    def _load_or_init_data(self):
        """Load existing data or initialize new data structures."""
        if os.path.exists(self.tracker_file):
            # Load existing data
            with open(self.tracker_file, 'r') as f:
                self.tracker = json.load(f)
            
            self.seeds = pd.read_csv(self.seeds_file)
            self.remaining_seeds = pd.read_csv(self.remaining_seeds_file)
            
            if os.path.exists(self.matrix_file):
                self.sparse_matrix = scipy.sparse.load_npz(self.matrix_file)
                self.sparse_matrix = self.sparse_matrix.tolil()
            else:
                self.sparse_matrix = scipy.sparse.lil_matrix((0, 0))
        else:
            # Initialize new data
            self.tracker = {
                "current_index": 0,
                "last_request": 0,
                "accounts_processed": 0
            }
            
            # Create empty dataframes
            self.seeds = pd.DataFrame(columns=['names', 'links'])
            self.remaining_seeds = pd.DataFrame(columns=['names', 'links', 'followers', 'emerge_date'])
            self.sparse_matrix = scipy.sparse.lil_matrix((0, 0))
            
            # Save initial data
            self._save_data()
    
    def _save_data(self):
        """Save all data to disk."""
        # Save tracker data
        with open(self.tracker_file, 'w') as f:
            json.dump(self.tracker, f)
        
        # Save seeds and remaining_seeds
        self.seeds.to_csv(self.seeds_file, index=False)
        self.remaining_seeds.to_csv(self.remaining_seeds_file, index=False)
        
        # Save sparse matrix
        sparse_csr = self.sparse_matrix.tocsr()
        scipy.sparse.save_npz(self.matrix_file, sparse_csr)
    
    def _respect_rate_limits(self):
        """Ensure we don't exceed rate limits by adding random delays."""
        current_time = time.time()
        time_since_last = current_time - self.tracker["last_request"]
        
        # Ensure at least 1-3 seconds between requests
        if time_since_last < 1:
            delay = 1 + random.random() * 2
            time.sleep(delay)
        
        # Add occasional longer delays
        if random.random() < 0.1:
            print("Taking a short break to avoid rate limiting...")
            time.sleep(5 + random.random() * 5)
        
        self.tracker["last_request"] = time.time()
    
    def get_html_scroll(self, url, max_scrolls=50):
        """
        Scroll a page to load dynamic content.
        
        Args:
            url: URL to scroll
            max_scrolls: Maximum number of scrolls to perform
        
        Returns:
            The page source after scrolling
        """
        self._respect_rate_limits()
        self.driver.get(url)
        
        scroll_pause_time = 0.5
        
        # Get initial scroll height
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        for i in range(max_scrolls):
            # Scroll down
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # Wait to load
            time.sleep(scroll_pause_time)
            
            # Calculate new scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            
            # Break if end of page
            if new_height == last_height:
                # Double check by waiting a bit longer
                time.sleep(2)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
            
            last_height = new_height
        
        return self.driver.page_source
    
    def parse_likes(self, html):
        """Parse likes from HTML content."""
        links = re.findall('<a class="sound__coverArt" href="/\S{,50}/', html)
        usernames = [str[34:len(str)-1].split("/")[0] for str in links]
        
        # Limit to max_items
        if len(usernames) > self.max_items:
            usernames = usernames[:self.max_items]
            
        return np.column_stack((usernames, usernames))
    
    def parse_reposts(self, html):
        """Parse reposts from HTML content."""
        links = re.findall('<a class="sound__coverArt" href="/.{,100}/', html)
        usernames = [str[34:len(str)-1].split("/")[0] for str in links]
        
        # Limit to max_items
        if len(usernames) > self.max_items:
            usernames = usernames[:self.max_items]
            
        return np.column_stack((usernames, usernames))
    
    def parse_comments(self, html):
        """Parse comments from HTML content."""
        links = re.findall('a class="sc-link-light" href="/.{,100}/', html)
        usernames = [str[31:len(str)-1].split("/")[0] for str in links]
        
        # Limit to max_items
        if len(usernames) > self.max_items:
            usernames = usernames[:self.max_items]
            
        return np.column_stack((usernames, usernames))
    
    def parse_follows(self, html):
        """Parse follows from HTML content."""
        links = re.findall('<a href="/.{,100}" class="userBadgeListItem__image">', html)
        usernames = [str[10:len(str)-35] for str in links]
        
        # Limit to max_items
        if len(usernames) > self.max_items:
            usernames = usernames[:self.max_items]
            
        return np.column_stack((usernames, usernames))
    
    def parse_followers(self, html):
        """Parse follower count from HTML content."""
        followers = re.findall('"followers_count":\d{,50},', html)
        if not followers:
            return 0
        
        num = int(followers[0][18:len(followers[0])-1])
        return num
    
    def parse_name(self, html):
        """Parse display name from HTML content."""
        name_match = re.findall('"username":".{,200}"', html)
        if not name_match:
            return "unknown"
            
        name = name_match[0][12:]
        name = name.split('\"')[0]
        return name
    
    def parse_tracks(self, html):
        """
        Parse track upload dates and determine 'emerge date'.
        
        Returns:
            emerge_date: The date representing when the artist "emerged" (15% quantile of upload dates)
        """
        dates = re.findall('datetime=".{,25}">', html)
        
        if not dates:
            return "unknown"
            
        dates = [str[10:len(str)-2] for str in dates]
        dates = pd.to_datetime(dates)
        
        # Convert to unix time
        unix_times = (dates - pd.Timestamp("1970-01-01 00:00:00+00:00")) // pd.Timedelta('1s')
        unix_times = unix_times.to_numpy()
        unix_times = np.unique(unix_times)  # Reduce effects of albums with multiple tracks
        
        if len(unix_times) == 0:
            return "unknown"
            
        emerge_date = np.quantile(unix_times, 0.15)
        emerge_date = datetime.utcfromtimestamp(emerge_date)
        return emerge_date.strftime('%m/%d/%Y')
    
    def add_to_seeds(self, interactions, seeds):
        """
        Add newly encountered accounts to the seeds dataframe.
        
        Args:
            interactions: Array of interaction data
            seeds: DataFrame of seed accounts
            
        Returns:
            Updated seeds DataFrame
        """
        for i in range(interactions.shape[0]):
            if interactions[i, 1] not in seeds['links'].values:
                new_row = pd.DataFrame({
                    'names': [interactions[i, 0]],
                    'links': [interactions[i, 1]]
                })
                seeds = pd.concat([seeds, new_row], ignore_index=True)
        
        return seeds
    
    def collect_interactions(self, username):
        """
        Collect all interactions for a given username.
        
        Args:
            username: SoundCloud username to collect interactions for
            
        Returns:
            Tuple of (success, account_data, interaction_data)
        """
        base_url = f"https://soundcloud.com/{username}"
        
        try:
            # Get account info
            self.driver.get(base_url)
            html_profile = self.driver.page_source
            
            followers = self.parse_followers(html_profile)
            display_name = self.parse_name(html_profile)
            
            # Get tracks info for emerge date
            tracks_url = f"{base_url}/tracks"
            html_tracks = self.get_html_scroll(tracks_url)
            emerge_date = self.parse_tracks(html_tracks)
            
            # Collect interactions
            interaction_data = {}
            
            # Likes
            likes_url = f"{base_url}/likes"
            html_likes = self.get_html_scroll(likes_url)
            interaction_data['likes'] = self.parse_likes(html_likes)
            
            # Comments
            comments_url = f"{base_url}/comments"
            html_comments = self.get_html_scroll(comments_url)
            interaction_data['comments'] = self.parse_comments(html_comments)
            
            # Following
            following_url = f"{base_url}/following"
            html_following = self.get_html_scroll(following_url)
            interaction_data['following'] = self.parse_follows(html_following)
            
            # Reposts
            reposts_url = f"{base_url}/reposts"
            html_reposts = self.get_html_scroll(reposts_url)
            interaction_data['reposts'] = self.parse_reposts(html_reposts)
            
            # Create account data
            account_data = {
                'names': display_name,
                'links': username,
                'followers': followers,
                'emerge_date': emerge_date
            }
            
            return True, account_data, interaction_data
            
        except Exception as e:
            print(f"Error collecting data for {username}: {e}")
            return False, None, None
    
    def update_matrix(self, username, interaction_data):
        """
        Update the sparse matrix with interaction data for a username.
        
        Args:
            username: SoundCloud username
            interaction_data: Dictionary of interaction data by type
        """
        # Resize matrix to add a new row
        current_row_idx = self.sparse_matrix.shape[0]
        self.sparse_matrix.resize((current_row_idx + 1, self.seeds.shape[0]))
        
        # Find row index for this username
        row_idx = self.remaining_seeds[self.remaining_seeds['links'] == username].index[0]
        
        # Process each interaction type
        for interaction_type, data in interaction_data.items():
            # Add new accounts to seeds
            self.seeds = self.add_to_seeds(data, self.seeds)
            
            # Resize matrix to accommodate new columns
            self.sparse_matrix.resize((self.sparse_matrix.shape[0], self.seeds.shape[0]))
            
            # Get weight based on interaction type
            weight = 1 if interaction_type == 'likes' else 3
            
            # Update matrix
            for j in range(data.shape[0]):
                target_username = data[j, 1]
                
                # Find column index
                col_idx = self.seeds[self.seeds['links'] == target_username].index[0]
                
                # Update matrix element
                self.sparse_matrix[row_idx, col_idx] += weight
    
    def find_next_account(self):
        """
        Find the next account to scrape based on interaction frequency.
        
        Returns:
            Username of the next account to scrape
        """
        # Convert to CSC format for efficient column operations
        csc_matrix = self.sparse_matrix.tocsc()
        
        # Sum interactions by column
        sums = np.array(csc_matrix.sum(axis=0)).flatten()
        
        # Create DataFrame with sums
        df = pd.DataFrame({
            'links': self.seeds['links'],
            'sums': sums
        })
        
        # Remove accounts already in remaining_seeds
        df = df[~df['links'].isin(self.remaining_seeds['links'])]
        
        # Return account with highest interaction count
        if df.shape[0] > 0:
            return df.loc[df['sums'].idxmax(), 'links']
        else:
            return None
    
    def add_seed_accounts(self, usernames):
        """
        Add initial seed accounts to remaining_seeds.
        
        Args:
            usernames: List of SoundCloud usernames to add as seeds
        """
        for username in usernames:
            if username not in self.remaining_seeds['links'].values:
                new_row = pd.DataFrame({
                    'names': [username],
                    'links': [username],
                    'followers': [0],
                    'emerge_date': ["unknown"]
                })
                self.remaining_seeds = pd.concat([self.remaining_seeds, new_row], ignore_index=True)
    
    def scrape(self, seed_accounts=None, max_accounts=100):
        """
        Main scraping method that manages the scraping process.
        
        Args:
            seed_accounts: List of SoundCloud usernames to start with
            max_accounts: Maximum number of accounts to scrape
        """
        try:
            # Add seed accounts if provided
            if seed_accounts:
                self.add_seed_accounts(seed_accounts)
            
            # Start from the current index
            current_idx = self.tracker["current_index"]
            
            while self.tracker["accounts_processed"] < max_accounts:
                # Determine next account to scrape
                if current_idx < len(self.remaining_seeds):
                    username = self.remaining_seeds.iloc[current_idx]['links']
                else:
                    # Find account with highest interaction count
                    username = self.find_next_account()
                    
                    if not username:
                        print("No more accounts to scrape.")
                        break
                    
                    # Add to remaining_seeds
                    new_row = pd.DataFrame({
                        'names': [username],
                        'links': [username],
                        'followers': [0],
                        'emerge_date': ["unknown"]
                    })
                    self.remaining_seeds = pd.concat([self.remaining_seeds, new_row], ignore_index=True)
                
                print(f"Scraping account {self.tracker['accounts_processed']+1}/{max_accounts}: {username}")
                
                # Collect data for this account
                success, account_data, interaction_data = self.collect_interactions(username)
                
                if success:
                    # Update account data
                    self.remaining_seeds.loc[current_idx, 'names'] = account_data['names']
                    self.remaining_seeds.loc[current_idx, 'followers'] = account_data['followers']
                    self.remaining_seeds.loc[current_idx, 'emerge_date'] = account_data['emerge_date']
                    
                    # Update matrix with interaction data
                    self.update_matrix(username, interaction_data)
                    
                    # Update progress
                    self.tracker["accounts_processed"] += 1
                
                # Move to next account
                current_idx += 1
                self.tracker["current_index"] = current_idx
                
                # Save progress periodically
                if self.tracker["accounts_processed"] % 5 == 0:
                    print(f"Saving progress after {self.tracker['accounts_processed']} accounts...")
                    self._save_data()
                
        except KeyboardInterrupt:
            print("\nScraping paused. Progress has been saved.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Save final data
            self._save_data()
            self.driver.quit()
    
    def resume(self, max_accounts=100):
        """Resume scraping from the last position."""
        print(f"Resuming scraping from account {self.tracker['current_index']} " +
              f"({self.tracker['accounts_processed']} accounts processed so far)")
        self.scrape(max_accounts=max_accounts)

if __name__ == "__main__":
    # Example usage
    scraper = SoundCloudScraper(data_dir="./data", max_items=500)
    
    # Start with some seed accounts
    seed_accounts = ["diplo", "flume", "kaytranada", "fourtet", "charlesthefirst"]
    
    # Run the scraper
    scraper.scrape(seed_accounts=seed_accounts, max_accounts=100) 