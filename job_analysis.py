import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud
import folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

# Set up NLTK data path
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Download all required NLTK data
try:
    # Download required NLTK data with explicit path
    for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
        nltk.download(resource, download_dir=nltk_data_path, quiet=True)
    
    # Create a simple word tokenizer that doesn't rely on punkt_tab
    word_tokenizer = RegexpTokenizer(r'\w+')
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")
    print("Please try downloading manually using:")
    print("python -m nltk.downloader punkt stopwords averaged_perceptron_tagger")

class JobDataAnalyzer:
    def __init__(self, csv_file):
        print(f"Reading {csv_file}...")
        self.df = pd.read_csv(csv_file)
        print(f"Found {len(self.df)} job listings")
        
        # Initialize NLTK components
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stop_words = set(stopwords.words('english'))
        
        # Define technical skills categories
        self.tech_skills = {
            'programming_languages': [
                'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 
                'kotlin', 'go', 'rust', 'typescript', 'scala', 'perl', 'r', 'matlab'
            ],
            'web_technologies': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 
                'django', 'flask', 'spring', 'asp.net', 'jquery', 'bootstrap'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 
                'dynamodb', 'cassandra', 'elasticsearch', 'neo4j'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 
                'terraform', 'ansible', 'circleci', 'travis'
            ],
            'tools_practices': [
                'git', 'jira', 'agile', 'scrum', 'ci/cd', 'tdd', 'rest', 'graphql',
                'microservices', 'maven', 'gradle'
            ],
            'machine_learning': [
                'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'opencv',
                'pandas', 'numpy', 'machine learning', 'ai', 'deep learning'
            ]
        }
        
        # Create output directory for visualizations
        self.viz_dir = "visualizations"
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
        
    def detect_remote_jobs(self):
        """Detect remote jobs based on job description and title"""
        print("Detecting remote jobs...")
        
        def is_remote_job(text):
            if pd.isna(text):
                return False
                
            remote_indicators = [
                'remote', 'work from home', 'wfh', 'virtual', 'telecommute',
                'work remotely', 'remote work', 'remote position', 'remote role',
                'remote opportunity', 'remote job', 'work anywhere'
            ]
            
            text = text.lower()
            return any(indicator in text for indicator in remote_indicators)
        
        # Check both title and description for remote indicators
        self.df['is_remote'] = self.df.apply(
            lambda row: is_remote_job(row['title']) or is_remote_job(row['description']),
            axis=1
        )
        
        remote_count = self.df['is_remote'].sum()
        total_count = len(self.df)
        print(f"Found {remote_count} remote jobs out of {total_count} total jobs ({remote_count/total_count*100:.1f}%)")

    def analyze(self):
        """Perform comprehensive data analysis with visualizations"""
        # Extract skills
        self.extract_skills()
        
        # Clean salary data
        self.clean_salary()
        
        # Detect remote jobs
        self.detect_remote_jobs()
        
        # Create visualizations
        self.create_skill_visualizations()
        self.create_geographic_heatmap()
        
        # Perform clustering with 8 clusters
        self.perform_clustering(n_clusters=8)
        
        # Print summary statistics
        print("\nAnalysis Summary:")
        print(f"Total jobs analyzed: {len(self.df)}")
        print(f"Number of unique locations: {self.df['location'].nunique()}")
        print(f"Number of remote jobs: {self.df['is_remote'].sum()}")
        
        # Save analyzed data
        self.save_analyzed_data('analyzed_jobs.csv')
        
    def clean_salary(self):
        """Extract and standardize salary information to hourly rates"""
        def extract_salary(text):
            if pd.isna(text) or text == '':
                return None
                
            # Regular expressions for different salary formats
            patterns = {
                # Yearly formats
                'yearly': r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per year|annually|year|yr|a year)',
                'yearly_no_space': r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)(?:per year|annually|year|yr|a year)',
                'range_yearly': r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*-\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per year|annually|year|yr|a year)',
                'range_yearly_no_space': r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*-\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)(?:per year|annually|year|yr|a year)',
                
                # Hourly formats
                'hourly': r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per hour|hourly|hr|an hour)',
                'hourly_no_space': r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)(?:per hour|hourly|hr|an hour)',
                'range_hourly': r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*-\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per hour|hourly|hr|an hour)',
                'range_hourly_no_space': r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*-\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)(?:per hour|hourly|hr|an hour)',
                
                # K notation formats
                'k_yearly': r'\$(\d{1,3}(?:\.\d{1,2})?)\s*k\s*(?:per year|annually|year|yr|a year)',
                'k_yearly_no_space': r'\$(\d{1,3}(?:\.\d{1,2})?)\s*k(?:per year|annually|year|yr|a year)',
                'k_range_yearly': r'\$(\d{1,3}(?:\.\d{1,2})?)\s*k\s*-\s*\$(\d{1,3}(?:\.\d{1,2})?)\s*k\s*(?:per year|annually|year|yr|a year)',
                'k_range_yearly_no_space': r'\$(\d{1,3}(?:\.\d{1,2})?)\s*k\s*-\s*\$(\d{1,3}(?:\.\d{1,2})?)\s*k(?:per year|annually|year|yr|a year)',
                
                # From formats
                'from_hourly': r'From\s+\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:an hour|per hour|hourly|hr)',
                'from_yearly': r'From\s+\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per year|annually|year|yr|a year)',
                'from_k_yearly': r'From\s+\$(\d{1,3}(?:\.\d{1,2})?)\s*k\s*(?:per year|annually|year|yr|a year)',
                
                # Up to formats
                'up_to_hourly': r'Up to\s+\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:an hour|per hour|hourly|hr)',
                'up_to_yearly': r'Up to\s+\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per year|annually|year|yr|a year)',
                'up_to_k_yearly': r'Up to\s+\$(\d{1,3}(?:\.\d{1,2})?)\s*k\s*(?:per year|annually|year|yr|a year)',
                
                # Starting at formats
                'starting_hourly': r'Starting at\s+\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:an hour|per hour|hourly|hr)',
                'starting_yearly': r'Starting at\s+\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per year|annually|year|yr|a year)',
                'starting_k_yearly': r'Starting at\s+\$(\d{1,3}(?:\.\d{1,2})?)\s*k\s*(?:per year|annually|year|yr|a year)'
            }
            
            def convert_to_hourly(salary, is_yearly):
                """Convert salary to hourly rate"""
                if is_yearly:
                    return salary / 2080
                return salary
            
            # First try exact matches
            for pattern_type, pattern in patterns.items():
                match = re.search(pattern, str(text).lower())
                if match:
                    is_yearly = any(x in pattern_type for x in ['yearly', 'k_', 'from_yearly', 'up_to_yearly', 'starting_yearly'])
                    
                    if 'range' in pattern_type or 'k_range' in pattern_type:
                        min_salary = float(match.group(1).replace(',', ''))
                        max_salary = float(match.group(2).replace(',', ''))
                        
                        if 'k_' in pattern_type:
                            min_salary *= 1000
                            max_salary *= 1000
                        
                        min_hourly = convert_to_hourly(min_salary, is_yearly)
                        max_hourly = convert_to_hourly(max_salary, is_yearly)
                        avg_hourly = (min_hourly + max_hourly) / 2
                        
                        return {
                            'min': min_hourly,
                            'max': max_hourly,
                            'avg': avg_hourly,
                            'original': f"${min_salary:,.0f} - ${max_salary:,.0f} {'/year' if is_yearly else '/hr'}"
                        }
                    else:
                        salary = float(match.group(1).replace(',', ''))
                        
                        if 'k_' in pattern_type:
                            salary *= 1000
                        
                        hourly_rate = convert_to_hourly(salary, is_yearly)
                        
                        return {
                            'min': hourly_rate,
                            'max': hourly_rate,
                            'avg': hourly_rate,
                            'original': f"${salary:,.0f} {'/year' if is_yearly else '/hr'}"
                        }
            
            # If no exact match, try to find any number with $ and year/hour indicators
            fallback_patterns = [
                (r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*-\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?).*?(?:year|yr|annually)', True),
                (r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*-\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?).*?(?:hour|hr|hourly)', False),
                (r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?).*?(?:year|yr|annually)', True),
                (r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?).*?(?:hour|hr|hourly)', False)
            ]
            
            for pattern, is_yearly in fallback_patterns:
                match = re.search(pattern, str(text).lower())
                if match:
                    if len(match.groups()) > 1:
                        min_salary = float(match.group(1).replace(',', ''))
                        max_salary = float(match.group(2).replace(',', ''))
                        min_hourly = convert_to_hourly(min_salary, is_yearly)
                        max_hourly = convert_to_hourly(max_salary, is_yearly)
                        avg_hourly = (min_hourly + max_hourly) / 2
                        return {
                            'min': min_hourly,
                            'max': max_hourly,
                            'avg': avg_hourly,
                            'original': f"${min_salary:,.0f} - ${max_salary:,.0f} {'/year' if is_yearly else '/hr'}"
                        }
                    else:
                        salary = float(match.group(1).replace(',', ''))
                        hourly_rate = convert_to_hourly(salary, is_yearly)
                        return {
                            'min': hourly_rate,
                            'max': hourly_rate,
                            'avg': hourly_rate,
                            'original': f"${salary:,.0f} {'/year' if is_yearly else '/hr'}"
                        }
            
            return None

        # First try to extract from salary column
        self.df['salary_info'] = self.df['salary'].apply(extract_salary)
        
        # For rows where salary_info is None, try to extract from description
        mask = self.df['salary_info'].isna()
        self.df.loc[mask, 'salary_info'] = self.df.loc[mask, 'description'].apply(extract_salary)
        
        # Create standardized hourly salary columns
        self.df['min_hourly'] = self.df['salary_info'].apply(lambda x: x['min'] if x else None)
        self.df['max_hourly'] = self.df['salary_info'].apply(lambda x: x['max'] if x else None)
        self.df['avg_hourly'] = self.df['salary_info'].apply(lambda x: x['avg'] if x else None)
        self.df['original_salary'] = self.df['salary_info'].apply(lambda x: x['original'] if x else None)
        
        # Print summary of findings
        print("\nJob Analysis Summary:")
        print("=" * 50)
        for idx, row in self.df.iterrows():
            print(f"\nJob {idx + 1}: {row['title']}")
            print(f"Company: {row['company']}")
            if pd.notna(row['original_salary']):
                print(f"Salary: {row['original_salary']} (${row['avg_hourly']:.2f}/hr)")
            else:
                print("Salary: Not specified")
            if 'skills' in row:
                print(f"Skills: {', '.join(row['skills'])}")
        print("\n" + "=" * 50)
        
        # Print overall statistics
        total_salaries = len(self.df)
        extracted_salaries = self.df['salary_info'].notna().sum()
        print(f"\nOverall Statistics:")
        print(f"Total jobs analyzed: {total_salaries}")
        print(f"Jobs with salary information: {extracted_salaries} ({extracted_salaries/total_salaries*100:.1f}%)")
        
        if extracted_salaries > 0:
            print("\nSalary Statistics (Hourly):")
            print(f"Average: ${self.df['avg_hourly'].mean():.2f}/hr")
            print(f"Median: ${self.df['avg_hourly'].median():.2f}/hr")
            print(f"Min: ${self.df['min_hourly'].min():.2f}/hr")
            print(f"Max: ${self.df['max_hourly'].max():.2f}/hr")
            
    def extract_skills(self):
        """Extract technical skills from job descriptions"""
        def find_skills(text):
            if pd.isna(text):
                return []
                
            # Use simple word tokenization instead of NLTK's word_tokenize
            tokens = self.tokenizer.tokenize(text.lower())
            tokens = [t for t in tokens if t not in self.stop_words]
            
            # Find skills
            found_skills = []
            for category, skills in self.tech_skills.items():
                for skill in skills:
                    # Handle multi-word skills
                    if ' ' in skill:
                        if skill.lower() in text.lower():
                            found_skills.append(skill)
                    else:
                        if skill in tokens:
                            found_skills.append(skill)
            
            return list(set(found_skills))  # Remove duplicates
            
        # Extract skills from description
        self.df['skills'] = self.df['description'].apply(find_skills)
        
        # Create skill columns
        all_skills = set()
        for skills in self.df['skills']:
            all_skills.update(skills)
            
        for skill in all_skills:
            self.df[f'skill_{skill}'] = self.df['skills'].apply(lambda x: 1 if skill in x else 0)
            
    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.df['description'].fillna(''))
        
        # Calculate distortions and silhouette scores for different numbers of clusters
        distortions = []
        silhouette_scores = []
        K = range(2, max_clusters + 1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(tfidf_matrix)
            distortions.append(kmeans.inertia_)
            
            # Calculate silhouette score
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(tfidf_matrix, labels)
            silhouette_scores.append(silhouette_avg)
        
        # Plot elbow curve
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('Elbow Method')
        
        # Plot silhouette scores
        plt.subplot(1, 2, 2)
        plt.plot(K, silhouette_scores, 'rx-')
        plt.xlabel('k')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'cluster_analysis.png'))
        plt.close()
        
        # Return optimal k based on silhouette score
        optimal_k = K[np.argmax(silhouette_scores)]
        return optimal_k

    def create_skill_visualizations(self):
        """Create bar charts and word cloud for skills"""
        # Get all skills
        all_skills = []
        for skills in self.df['skills']:
            all_skills.extend(skills)
        skill_counts = Counter(all_skills)
        
        # Create bar chart for top 20 skills
        plt.figure(figsize=(15, 8))
        top_skills = dict(skill_counts.most_common(20))
        plt.bar(top_skills.keys(), top_skills.values())
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 20 Most Common Skills')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'top_skills_bar.png'))
        plt.close()
        
        # Create word cloud
        wordcloud = WordCloud(width=1200, height=800, background_color='white').generate_from_frequencies(skill_counts)
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Skills Word Cloud')
        plt.savefig(os.path.join(self.viz_dir, 'skills_wordcloud.png'))
        plt.close()

    def create_geographic_heatmap(self):
        """Create geographic heatmap of job locations using pre-defined coordinates"""
        print("Creating geographic visualization...")
        
        # Extract city and state from location
        self.df[['City', 'State']] = self.df['location'].str.extract(r'(.*?),\s*(\w{2})')
        
        # Major US cities coordinates (latitude, longitude)
        city_coords = {
            'New York': (40.7128, -74.0060),
            'Los Angeles': (34.0522, -118.2437),
            'Chicago': (41.8781, -87.6298),
            'Houston': (29.7604, -95.3698),
            'Phoenix': (33.4484, -112.0740),
            'Philadelphia': (39.9526, -75.1652),
            'San Antonio': (29.4241, -98.4936),
            'San Diego': (32.7157, -117.1611),
            'Dallas': (32.7767, -96.7970),
            'San Jose': (37.3382, -121.8863),
            'Austin': (30.2672, -97.7431),
            'Jacksonville': (30.3322, -81.6557),
            'Fort Worth': (32.7555, -97.3308),
            'Columbus': (39.9612, -82.9988),
            'San Francisco': (37.7749, -122.4194),
            'Charlotte': (35.2271, -80.8431),
            'Indianapolis': (39.7684, -86.1581),
            'Seattle': (47.6062, -122.3321),
            'Denver': (39.7392, -104.9903),
            'Boston': (42.3601, -71.0589),
            'Atlanta': (33.7490, -84.3880),
            'Miami': (25.7617, -80.1918),
            'Portland': (45.5155, -122.6789),
            'Las Vegas': (36.1699, -115.1398),
            'Minneapolis': (44.9778, -93.2650),
            'Detroit': (42.3314, -83.0458),
            'Raleigh': (35.7796, -78.6382),
            'Nashville': (36.1627, -86.7816),
            'Memphis': (35.1495, -90.0490),
            'Kansas City': (39.0997, -94.5786)
        }
        
        # Count jobs by location
        location_counts = self.df.groupby(['City', 'State']).size().reset_index(name='count')
        
        # Create a map centered on the US
        m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
        
        # Add markers for cities with known coordinates
        for _, row in location_counts.iterrows():
            city = row['City']
            if city in city_coords:
                lat, lon = city_coords[city]
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=row['count'] * 0.5,  # Scale radius by job count
                    popup=f"{city}: {row['count']} jobs",
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.6
                ).add_to(m)
        
        # Save the map
        m.save(os.path.join(self.viz_dir, 'job_locations.html'))
        print("Location visualization saved successfully")
        
        # Also create a bar chart of top locations
        plt.figure(figsize=(15, 8))
        top_locations = location_counts.nlargest(15, 'count')
        plt.bar(top_locations['City'] + ', ' + top_locations['State'], top_locations['count'])
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 15 Job Locations')
        plt.xlabel('Location')
        plt.ylabel('Number of Jobs')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'top_locations.png'))
        plt.close()

    def create_remote_trend_plot(self, historical_data):
        """Create plot showing change in remote work trends"""
        # Calculate remote percentages
        current_remote = self.df['is_remote'].sum() / len(self.df) * 100
        historical_remote = historical_data['is_remote'].sum() / len(historical_data) * 100
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(['Historical', 'Current'], [historical_remote, current_remote])
        plt.title('Remote Work Trend')
        plt.ylabel('Percentage of Remote Jobs')
        plt.ylim(0, 100)
        
        # Add percentage labels on bars
        for i, v in enumerate([historical_remote, current_remote]):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center')
        
        plt.savefig(os.path.join(self.viz_dir, 'remote_trend.png'))
        plt.close()

    def perform_clustering(self, n_clusters=None):
        """Perform K-means clustering on job descriptions with optimal cluster selection"""
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.df['description'].fillna(''))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['cluster'] = kmeans.fit_predict(tfidf_matrix)
        
        # Get top terms for each cluster
        feature_names = vectorizer.get_feature_names_out()
        cluster_centers = kmeans.cluster_centers_
        
        # Create scatter plot of clusters using PCA for dimensionality reduction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(tfidf_matrix.toarray())
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                            c=self.df['cluster'], cmap='viridis')
        plt.title('Job Clusters (PCA-reduced)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter, label='Cluster')
        plt.savefig(os.path.join(self.viz_dir, 'cluster_scatter.png'))
        plt.close()
        
        # Print cluster information
        print("\nCluster Analysis:")
        for i in range(n_clusters):
            top_terms = [feature_names[j] for j in cluster_centers[i].argsort()[:-10:-1]]
            cluster_size = (self.df['cluster'] == i).sum()
            print(f"\nCluster {i} (Size: {cluster_size} jobs):")
            print(f"Top terms: {', '.join(top_terms)}")

    def save_analyzed_data(self, output_file):
        """Save the analyzed data to a new CSV file"""
        self.df.to_csv(output_file, index=False)
        print(f"\nAnalyzed data saved to {output_file}")

def main():
    # Initialize analyzer with CSV file
    analyzer = JobDataAnalyzer('scraped_data/master_jobs.csv')
    
    # Perform analysis
    analyzer.analyze()

if __name__ == "__main__":
    main() 