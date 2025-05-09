import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import folium

# Set up NLTK
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

try:
    for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
        nltk.download(resource, download_dir=nltk_data_path, quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")

class BaseJobAnalyzer:
    def __init__(self):
        self.word_tokenizer = RegexpTokenizer(r'\w+')
        self.stop_words = set(stopwords.words('english'))
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

    def extract_skills(self, text):
        if pd.isna(text):
            return []
        
        text = text.lower()
        tokens = self.word_tokenizer.tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        
        found_skills = []
        for category, skills in self.tech_skills.items():
            for skill in skills:
                if ' ' in skill:  # Multi-word skills
                    if skill in text:
                        found_skills.append((skill, category))
                else:  # Single-word skills
                    if skill in tokens:
                        found_skills.append((skill, category))
        
        return found_skills

    def is_remote_job(self, text):
        """Determine if a job is remote based on its description"""
        if pd.isna(text):
            return False
            
        remote_indicators = [
            'remote', 'work from home', 'wfh', 'virtual', 'telecommute',
            'work remotely', 'remote work', 'remote position', 'remote role',
            'remote opportunity', 'remote job', 'work anywhere'
        ]
        
        text = text.lower()
        return any(indicator in text for indicator in remote_indicators)

    def perform_clustering(self, n_clusters=8, viz_prefix=None):
        """Perform K-means clustering on job descriptions and save scatter plot"""
        print(f"Performing K-means clustering with {n_clusters} clusters...")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.df['Skills'].apply(lambda x: ' '.join([s[0] for s in x])) if 'Skills' in self.df.columns else self.df['description'].fillna(''))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['cluster'] = kmeans.fit_predict(tfidf_matrix)
        feature_names = vectorizer.get_feature_names_out()
        cluster_centers = kmeans.cluster_centers_
        # PCA for 2D scatter
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(tfidf_matrix.toarray())
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=self.df['cluster'], cmap='viridis')
        plt.title(f'Job Clusters (PCA-reduced) {viz_prefix or ""}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter, label='Cluster')
        fname = f'cluster_scatter_{viz_prefix}.png' if viz_prefix else 'cluster_scatter.png'
        plt.savefig(os.path.join('visualizations', fname))
        plt.close()
        # Save cluster info for comparison
        self.cluster_info = []
        for i in range(n_clusters):
            top_terms = [feature_names[j] for j in cluster_centers[i].argsort()[:-10:-1]]
            cluster_size = (self.df['cluster'] == i).sum()
            self.cluster_info.append({'cluster': i, 'size': cluster_size, 'top_terms': top_terms})
        print(f"Cluster sizes: {[c['size'] for c in self.cluster_info]}")
        return self.cluster_info

class Indeed10kAnalyzer(BaseJobAnalyzer):
    def __init__(self, csv_file):
        super().__init__()
        print(f"Reading {csv_file}...")
        self.df = pd.read_csv(csv_file)
        self.df.columns = [col.strip() for col in self.df.columns]  # Clean column names
        print(f"Found {len(self.df)} job listings")
        print("Columns in dataset:", self.df.columns.tolist())
        
    def clean_data(self):
        """Clean and prepare the dataset for analysis"""
        print("\nCleaning and preparing data...")
        
        # Handle location data
        location_col = 'City' if 'City' in self.df.columns else 'location'
        self.df[['City', 'State']] = self.df[location_col].str.extract(r'(.*?),\s*(\w{2})')
        
        # Clean up date
        if 'Date' in self.df.columns:
            self.df['Days_Ago'] = self.df['Date'].str.extract(r'(\d+)').astype(float)
            self.df.loc[self.df['Date'].str.contains('30\+ days'), 'Days_Ago'] = 30
        
        # Extract skills
        print("Extracting skills...")
        description_col = 'Summary' if 'Summary' in self.df.columns else 'description'
        self.df['Skills'] = self.df[description_col].apply(self.extract_skills)
        
        # Create skill category columns
        for category in self.tech_skills.keys():
            self.df[f'has_{category}'] = self.df['Skills'].apply(
                lambda x: any(skill[1] == category for skill in x)
            )
            
        # Identify remote jobs
        print("Identifying remote jobs...")
        self.df['is_remote'] = self.df[description_col].apply(self.is_remote_job)

    def analyze(self):
        """Analyze the dataset and print results"""
        print("\nAnalyzing dataset...")
        
        # Basic statistics
        total_jobs = len(self.df)
        print(f"\nTotal jobs analyzed: {total_jobs}")
        
        # Location analysis
        print("\nLocation Analysis:")
        print(f"Number of unique locations: {self.df['City'].nunique()}")
        
        # Remote work analysis
        remote_jobs = self.df['is_remote'].sum()
        remote_percentage = (remote_jobs / total_jobs) * 100
        print(f"\nRemote Work Analysis:")
        print(f"Remote jobs: {remote_jobs} ({remote_percentage:.1f}%)")
        print(f"On-site jobs: {total_jobs - remote_jobs} ({100 - remote_percentage:.1f}%)")
        
        # Top locations
        print("\nTop 10 Locations:")
        location_counts = self.df['City'].value_counts().head(10)
        for location, count in location_counts.items():
            print(f"{location}: {count} jobs ({count/total_jobs*100:.1f}%)")
        
        # Skills analysis
        print("\nSkills Analysis:")
        all_skills = [skill[0] for skills in self.df['Skills'] for skill in skills]
        skill_counts = Counter(all_skills)
        
        print("\nTop 20 Most Common Skills:")
        for skill, count in skill_counts.most_common(20):
            print(f"{skill}: {count} jobs ({count/total_jobs*100:.1f}%)")
        
        # Skill category analysis
        print("\nSkill Category Analysis:")
        for category in self.tech_skills.keys():
            category_count = self.df[f'has_{category}'].sum()
            print(f"{category}: {category_count} jobs ({category_count/total_jobs*100:.1f}%)")
        
        # Remote vs On-site Skills Comparison
        print("\nRemote vs On-site Skills Comparison:")
        remote_skills = [skill[0] for skills in self.df[self.df['is_remote']]['Skills'] for skill in skills]
        onsite_skills = [skill[0] for skills in self.df[~self.df['is_remote']]['Skills'] for skill in skills]
        
        remote_skill_counts = Counter(remote_skills)
        onsite_skill_counts = Counter(onsite_skills)
        
        print("\nTop 10 Skills in Remote Jobs:")
        for skill, count in remote_skill_counts.most_common(10):
            print(f"{skill}: {count} jobs ({count/remote_jobs*100:.1f}%)")
        
        print("\nTop 10 Skills in On-site Jobs:")
        for skill, count in onsite_skill_counts.most_common(10):
            print(f"{skill}: {count} jobs ({count/(total_jobs-remote_jobs)*100:.1f}%)")

class IndeedCurrentAnalyzer(BaseJobAnalyzer):
    def __init__(self, csv_file):
        super().__init__()
        print(f"Reading {csv_file}...")
        self.df = pd.read_csv(csv_file)
        print(f"Found {len(self.df)} job listings")
        print("Columns in dataset:", self.df.columns.tolist())
        
    def clean_data(self):
        """Clean and prepare the dataset for analysis"""
        print("\nCleaning and preparing data...")
        
        # Handle location data
        location_col = 'City' if 'City' in self.df.columns else 'location'
        self.df[['City', 'State']] = self.df[location_col].str.extract(r'(.*?),\s*(\w{2})')
        
        # Extract skills
        print("Extracting skills...")
        description_col = 'Summary' if 'Summary' in self.df.columns else 'description'
        self.df['Skills'] = self.df[description_col].apply(self.extract_skills)
        
        # Create skill category columns
        for category in self.tech_skills.keys():
            self.df[f'has_{category}'] = self.df['Skills'].apply(
                lambda x: any(skill[1] == category for skill in x)
            )
            
        # Identify remote jobs
        print("Identifying remote jobs...")
        self.df['is_remote'] = self.df[description_col].apply(self.is_remote_job)

    def analyze(self):
        """Analyze the dataset and print results"""
        print("\nAnalyzing dataset...")
        
        # Basic statistics
        total_jobs = len(self.df)
        print(f"\nTotal jobs analyzed: {total_jobs}")
        
        # Location analysis
        print("\nLocation Analysis:")
        print(f"Number of unique locations: {self.df['City'].nunique()}")
        
        # Remote work analysis
        remote_jobs = self.df['is_remote'].sum()
        remote_percentage = (remote_jobs / total_jobs) * 100
        print(f"\nRemote Work Analysis:")
        print(f"Remote jobs: {remote_jobs} ({remote_percentage:.1f}%)")
        print(f"On-site jobs: {total_jobs - remote_jobs} ({100 - remote_percentage:.1f}%)")
        
        # Top locations
        print("\nTop 10 Locations:")
        location_counts = self.df['City'].value_counts().head(10)
        for location, count in location_counts.items():
            print(f"{location}: {count} jobs ({count/total_jobs*100:.1f}%)")
        
        # Skills analysis
        print("\nSkills Analysis:")
        all_skills = [skill[0] for skills in self.df['Skills'] for skill in skills]
        skill_counts = Counter(all_skills)
        
        print("\nTop 20 Most Common Skills:")
        for skill, count in skill_counts.most_common(20):
            print(f"{skill}: {count} jobs ({count/total_jobs*100:.1f}%)")
        
        # Skill category analysis
        print("\nSkill Category Analysis:")
        for category in self.tech_skills.keys():
            category_count = self.df[f'has_{category}'].sum()
            print(f"{category}: {category_count} jobs ({category_count/total_jobs*100:.1f}%)")
        
        # Remote vs On-site Skills Comparison
        print("\nRemote vs On-site Skills Comparison:")
        remote_skills = [skill[0] for skills in self.df[self.df['is_remote']]['Skills'] for skill in skills]
        onsite_skills = [skill[0] for skills in self.df[~self.df['is_remote']]['Skills'] for skill in skills]
        
        remote_skill_counts = Counter(remote_skills)
        onsite_skill_counts = Counter(onsite_skills)
        
        print("\nTop 10 Skills in Remote Jobs:")
        for skill, count in remote_skill_counts.most_common(10):
            print(f"{skill}: {count} jobs ({count/remote_jobs*100:.1f}%)")
        
        print("\nTop 10 Skills in On-site Jobs:")
        for skill, count in onsite_skill_counts.most_common(10):
            print(f"{skill}: {count} jobs ({count/(total_jobs-remote_jobs)*100:.1f}%)")

def plot_elbow_silhouette(df, filename_prefix, text_col='description'):
    print(f"Plotting elbow and silhouette for {filename_prefix}...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df[text_col].fillna(''))
    distortions = []
    silhouette_scores = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(tfidf_matrix)
        distortions.append(kmeans.inertia_)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(tfidf_matrix, labels)
        silhouette_scores.append(silhouette_avg)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.tight_layout()
    plt.savefig(os.path.join('visualizations', f'{filename_prefix}_elbow_silhouette.png'))
    plt.close()

def plot_skills_comparison_percent(hist_skills, curr_skills, hist_total, curr_total, filename='skills_comparison_percent.png', top_n=20):
    print("Plotting skills percentage comparison bar chart...")
    hist_counts = Counter(hist_skills)
    curr_counts = Counter(curr_skills)
    all_skills = set([s for s, _ in hist_counts.most_common(top_n)] + [s for s, _ in curr_counts.most_common(top_n)])
    hist_perc = [100 * hist_counts.get(skill, 0) / hist_total for skill in all_skills]
    curr_perc = [100 * curr_counts.get(skill, 0) / curr_total for skill in all_skills]
    x = np.arange(len(all_skills))
    plt.figure(figsize=(16, 7))
    plt.bar(x-0.2, hist_perc, width=0.4, label='Historical')
    plt.bar(x+0.2, curr_perc, width=0.4, label='Current')
    plt.xticks(x, all_skills, rotation=45, ha='right')
    plt.xlabel('Skill')
    plt.ylabel('Percent of Jobs (%)')
    plt.title('Top Skills Comparison (Percent of Jobs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('visualizations', filename))
    plt.close()

def plot_remote_comparison(hist_remote_pct, curr_remote_pct, filename='remote_comparison.png'):
    print("Plotting remote work percentage comparison...")
    plt.figure(figsize=(7, 6))
    plt.bar(['Historical', 'Current'], [hist_remote_pct, curr_remote_pct], color=['blue', 'orange'])
    plt.ylabel('Percent Remote Jobs (%)')
    plt.title('Remote Work Percentage Comparison')
    for i, v in enumerate([hist_remote_pct, curr_remote_pct]):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join('visualizations', filename))
    plt.close()

def create_job_location_map(df, filename, city_coords, color='red', label='Jobs'):
    print(f"Creating job location map: {filename}")
    # Extract city/state if not already present
    if 'City' not in df.columns or 'State' not in df.columns:
        location_col = 'City' if 'City' in df.columns else 'location'
        df[['City', 'State']] = df[location_col].str.extract(r'(.*?),\s*(\w{2})')
    location_counts = df.groupby(['City', 'State']).size().reset_index(name='count')
    m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
    for _, row in location_counts.iterrows():
        city = row['City']
        if city in city_coords:
            lat, lon = city_coords[city]
            folium.CircleMarker(
                location=[lat, lon],
                radius=row['count'] * 0.5,
                popup=f"{city}: {row['count']} jobs",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                tooltip=label
            ).add_to(m)
    m.save(os.path.join('visualizations', filename))
    print(f"Saved {filename}")

def create_combined_job_location_map(hist_df, curr_df, city_coords):
    print("Creating combined job location map...")
    # Extract city/state if not already present
    for df in [hist_df, curr_df]:
        if 'City' not in df.columns or 'State' not in df.columns:
            location_col = 'City' if 'City' in df.columns else 'location'
            df[['City', 'State']] = df[location_col].str.extract(r'(.*?),\s*(\w{2})')
    m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
    # Historical layer
    hist_layer = folium.FeatureGroup(name='Historical Jobs', show=True)
    hist_counts = hist_df.groupby(['City', 'State']).size().reset_index(name='count')
    for _, row in hist_counts.iterrows():
        city = row['City']
        if city in city_coords:
            lat, lon = city_coords[city]
            folium.CircleMarker(
                location=[lat, lon],
                radius=row['count'] * 0.5,
                popup=f"{city}: {row['count']} jobs (Historical)",
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.5,
                tooltip='Historical'
            ).add_to(hist_layer)
    hist_layer.add_to(m)
    # Current layer
    curr_layer = folium.FeatureGroup(name='Current Jobs', show=True)
    curr_counts = curr_df.groupby(['City', 'State']).size().reset_index(name='count')
    for _, row in curr_counts.iterrows():
        city = row['City']
        if city in city_coords:
            lat, lon = city_coords[city]
            folium.CircleMarker(
                location=[lat, lon],
                radius=row['count'] * 0.5,
                popup=f"{city}: {row['count']} jobs (Current)",
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.5,
                tooltip='Current'
            ).add_to(curr_layer)
    curr_layer.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(os.path.join('visualizations', 'job_locations_combined.html'))
    print("Saved job_locations_combined.html")

def compare_datasets():
    """Compare historical and current job datasets"""
    # Initialize analyzers
    historical_analyzer = Indeed10kAnalyzer('indeed_10k.csv')  # Historical dataset
    current_analyzer = IndeedCurrentAnalyzer('analyzed_jobs.csv')  # Current dataset
    
    # Clean data
    historical_analyzer.clean_data()
    current_analyzer.clean_data()
    
    # City coordinates (same as in job_analysis.py)
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
    # Create job location maps
    create_job_location_map(historical_analyzer.df, 'job_locations_historical.html', city_coords, color='blue', label='Historical')
    create_job_location_map(current_analyzer.df, 'job_locations_current.html', city_coords, color='red', label='Current')
    create_combined_job_location_map(historical_analyzer.df, current_analyzer.df, city_coords)
    
    # Elbow and silhouette for historical data
    description_col = 'Summary' if 'Summary' in historical_analyzer.df.columns else 'description'
    plot_elbow_silhouette(historical_analyzer.df, 'historical', text_col=description_col)
    
    # Perform clustering
    hist_clusters = historical_analyzer.perform_clustering(n_clusters=8, viz_prefix='historical')
    curr_clusters = current_analyzer.perform_clustering(n_clusters=8, viz_prefix='current')
    
    # Skills comparison bar chart (percentages)
    hist_skills = [skill[0] for skills in historical_analyzer.df['Skills'] for skill in skills]
    curr_skills = [skill[0] for skills in current_analyzer.df['Skills'] for skill in skills]
    plot_skills_comparison_percent(hist_skills, curr_skills, len(historical_analyzer.df), len(current_analyzer.df))
    
    # Remote work percentage comparison
    hist_remote = historical_analyzer.df['is_remote'].sum()
    hist_total = len(historical_analyzer.df)
    curr_remote = current_analyzer.df['is_remote'].sum()
    curr_total = len(current_analyzer.df)
    hist_remote_pct = 100 * hist_remote / hist_total if hist_total else 0
    curr_remote_pct = 100 * curr_remote / curr_total if curr_total else 0
    plot_remote_comparison(hist_remote_pct, curr_remote_pct)
    
    # Print top terms for each cluster
    print('\n=== Cluster Top Terms Comparison ===')
    for i in range(8):
        print(f'Cluster {i}:')
        print(f'  Historical: {", ".join(hist_clusters[i]["top_terms"])})')
        print(f'  Current:    {", ".join(curr_clusters[i]["top_terms"])})')
    
    # Analyze datasets
    print("\n=== Historical Dataset Analysis ===")
    historical_analyzer.analyze()
    
    print("\n=== Current Dataset Analysis ===")
    current_analyzer.analyze()
    
    # Compare remote work trends
    print("\n=== Remote Work Trend Analysis ===")
    historical_remote = historical_analyzer.df['is_remote'].sum()
    historical_total = len(historical_analyzer.df)
    historical_remote_pct = (historical_remote / historical_total) * 100
    
    current_remote = current_analyzer.df['is_remote'].sum()
    current_total = len(current_analyzer.df)
    current_remote_pct = (current_remote / current_total) * 100
    
    print(f"Historical remote jobs: {historical_remote} ({historical_remote_pct:.1f}%)")
    print(f"Current remote jobs: {current_remote} ({current_remote_pct:.1f}%)")
    print(f"Change in remote work percentage: {current_remote_pct - historical_remote_pct:.1f}%")
    
    # Compare skill trends
    print("\n=== Skill Trend Analysis ===")
    
    # Get skill frequencies for both datasets
    historical_skills = [skill[0] for skills in historical_analyzer.df['Skills'] for skill in skills]
    current_skills = [skill[0] for skills in current_analyzer.df['Skills'] for skill in skills]
    
    historical_skill_counts = Counter(historical_skills)
    current_skill_counts = Counter(current_skills)
    
    # Calculate percentage changes for top skills
    all_skills = set(historical_skill_counts.keys()) | set(current_skill_counts.keys())
    skill_changes = {}
    
    for skill in all_skills:
        historical_pct = (historical_skill_counts[skill] / historical_total) * 100
        current_pct = (current_skill_counts[skill] / current_total) * 100
        change = current_pct - historical_pct
        skill_changes[skill] = change
    
    # Print top growing and declining skills
    print("\nTop 10 Growing Skills:")
    for skill, change in sorted(skill_changes.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{skill}: {change:+.1f}%")
    
    print("\nTop 10 Declining Skills:")
    for skill, change in sorted(skill_changes.items(), key=lambda x: x[1])[:10]:
        print(f"{skill}: {change:+.1f}%")

if __name__ == "__main__":
    compare_datasets() 