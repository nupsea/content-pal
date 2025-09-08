"""
No-API Web Enrichment System

Uses public data sources without requiring API keys:
- Wikipedia for comprehensive movie information
- Web scraping for publicly available movie data
- Open datasets and knowledge bases
- Local processing of scraped content

NO API KEYS REQUIRED!
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import wikipediaapi
import time
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from urllib.parse import quote
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoAPIEnrichment:
    """
    Movie enrichment using freely available web sources
    """
    
    def __init__(self):
        # Wikipedia API (no key required!)
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='MovieEnrichmentBot/1.0'
        )
        
        # Cache directory for scraped data
        self.cache_dir = Path("data/scraped_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Headers for web scraping (appear more like real browser)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Rate limiting
        self.last_request = 0
        self.request_delay = 2  # 2 seconds between requests to be respectful
        
        # Movie genre keywords for classification
        self.genre_keywords = {
            'action': ['fight', 'battle', 'war', 'combat', 'explosion', 'chase', 'weapon', 'military', 'soldier'],
            'comedy': ['funny', 'humor', 'laugh', 'comic', 'hilarious', 'amusing', 'witty', 'satirical'],
            'drama': ['emotional', 'family', 'relationship', 'serious', 'tragedy', 'character', 'life'],
            'horror': ['scary', 'frightening', 'ghost', 'monster', 'terror', 'nightmare', 'haunted', 'evil'],
            'romance': ['love', 'romantic', 'relationship', 'wedding', 'couple', 'heart', 'passion'],
            'thriller': ['suspense', 'mystery', 'dangerous', 'tension', 'psychological', 'crime'],
            'sci-fi': ['future', 'space', 'alien', 'technology', 'robot', 'scientific', 'planet'],
            'fantasy': ['magic', 'wizard', 'fairy', 'mythical', 'supernatural', 'dragon', 'quest'],
            'crime': ['detective', 'murder', 'police', 'criminal', 'investigation', 'heist', 'gang'],
            'documentary': ['real', 'true', 'factual', 'documentary', 'biography', 'history']
        }
        
        # Quality indicators from text
        self.quality_indicators = {
            'highly-rated': ['acclaimed', 'masterpiece', 'excellent', 'outstanding', 'brilliant', 'exceptional'],
            'award-winner': ['oscar', 'academy award', 'golden globe', 'emmy', 'bafta', 'cannes', 'winner'],
            'cult-classic': ['cult', 'iconic', 'legendary', 'classic', 'influential', 'groundbreaking'],
            'blockbuster': ['blockbuster', 'box office', 'highest-grossing', 'commercial success', 'hit'],
            'indie': ['independent', 'indie', 'art house', 'festival', 'low budget', 'artistic']
        }
        
        logger.info("🆓 No-API Enrichment System initialized")
        logger.info("📖 Using Wikipedia and public web sources")
    
    def _rate_limit(self):
        """Respectful rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request = time.time()
    
    def _get_cache_path(self, source: str, identifier: str) -> Path:
        """Generate cache file path"""
        safe_id = re.sub(r'[^\w\s-]', '', identifier.lower()).strip()[:50]
        return self.cache_dir / f"{source}_{safe_id}.json"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Dict]:
        """Load cached data"""
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.debug(f"Cache read error: {e}")
        return None
    
    def _save_to_cache(self, cache_path: Path, data: Dict):
        """Save data to cache"""
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"Cache write error: {e}")
    
    def get_wikipedia_info(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """Get movie information from Wikipedia"""
        cache_path = self._get_cache_path("wikipedia", f"{title}_{year}")
        cached_data = self._load_from_cache(cache_path)
        
        if cached_data:
            return cached_data
        
        self._rate_limit()
        
        try:
            # Try different Wikipedia page title formats
            search_titles = [
                f"{title} (film)",
                f"{title} ({year} film)" if year else None,
                f"{title} (movie)",
                title
            ]
            
            wiki_data = {}
            
            for search_title in search_titles:
                if not search_title:
                    continue
                
                try:
                    page = self.wiki.page(search_title)
                    
                    if page.exists():
                        wiki_data = {
                            'title': page.title,
                            'summary': page.summary[:1000],  # First 1000 chars
                            'text': page.text[:3000],        # First 3000 chars for analysis
                            'url': page.fullurl,
                            'found': True
                        }
                        break
                        
                except Exception as e:
                    logger.debug(f"Wikipedia search error for '{search_title}': {e}")
                    continue
            
            if not wiki_data:
                wiki_data = {'found': False}
            
            # Cache the result
            self._save_to_cache(cache_path, wiki_data)
            return wiki_data
            
        except Exception as e:
            logger.debug(f"Wikipedia error for '{title}': {e}")
            return {'found': False}
    
    def scrape_imdb_basic_info(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """Scrape basic IMDb information (be respectful!)"""
        cache_path = self._get_cache_path("imdb_basic", f"{title}_{year}")
        cached_data = self._load_from_cache(cache_path)
        
        if cached_data:
            return cached_data
        
        self._rate_limit()
        
        try:
            # Search IMDb for the movie
            search_query = f"{title} {year}" if year else title
            search_url = f"https://www.imdb.com/find?q={quote(search_query)}&s=tt"
            
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for first movie result
            result_items = soup.find_all('td', class_='result_text')
            
            imdb_data = {'found': False}
            
            if result_items:
                first_result = result_items[0]
                link = first_result.find('a')
                
                if link and 'href' in link.attrs:
                    movie_url = f"https://www.imdb.com{link['href']}"
                    
                    # Get basic movie page info
                    self._rate_limit()
                    movie_response = requests.get(movie_url, headers=self.headers, timeout=10)
                    movie_response.raise_for_status()
                    
                    movie_soup = BeautifulSoup(movie_response.content, 'html.parser')
                    
                    # Extract basic information
                    imdb_data = {
                        'found': True,
                        'url': movie_url,
                        'title_element': movie_soup.find('h1')
                    }
                    
                    # Try to extract genre information
                    genre_elements = movie_soup.find_all('span', class_='ipc-chip__text')
                    genres = [elem.get_text() for elem in genre_elements[:5]]  # First 5 genres
                    imdb_data['genres'] = genres
                    
                    # Try to extract plot
                    plot_elements = movie_soup.find_all('span', {'data-testid': 'plot-xs_to_m'})
                    if plot_elements:
                        imdb_data['plot'] = plot_elements[0].get_text()
                    
                    # Small delay to be respectful
                    time.sleep(1)
            
            # Cache the result
            self._save_to_cache(cache_path, imdb_data)
            return imdb_data
            
        except Exception as e:
            logger.debug(f"IMDb scraping error for '{title}': {e}")
            return {'found': False}
    
    def extract_tags_from_text(self, text: str, title: str = "") -> List[str]:
        """Extract semantic tags from text content"""
        if not text:
            return []
        
        text_lower = text.lower()
        title_lower = title.lower()
        extracted_tags = []
        
        # Genre classification based on keywords
        for genre, keywords in self.genre_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                extracted_tags.append(genre)
        
        # Quality indicators
        for quality, indicators in self.quality_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                extracted_tags.append(quality)
        
        # Director extraction (common patterns)
        director_patterns = [
            r'directed by ([A-Z][a-z]+ [A-Z][a-z]+)',
            r'director ([A-Z][a-z]+ [A-Z][a-z]+)',
            r'([A-Z][a-z]+ [A-Z][a-z]+).*direct'
        ]
        
        for pattern in director_patterns:
            matches = re.findall(pattern, text)
            for match in matches[:2]:  # Top 2 directors
                director_tag = match.lower().replace(' ', '-')
                extracted_tags.append(f"director-{director_tag}")
        
        # Actor extraction (leading actors)
        actor_patterns = [
            r'starring ([A-Z][a-z]+ [A-Z][a-z]+)',
            r'stars ([A-Z][a-z]+ [A-Z][a-z]+)',
            r'featuring ([A-Z][a-z]+ [A-Z][a-z]+)'
        ]
        
        for pattern in actor_patterns:
            matches = re.findall(pattern, text)
            for match in matches[:3]:  # Top 3 actors
                actor_tag = match.lower().replace(' ', '-')
                extracted_tags.append(f"actor-{actor_tag}")
        
        # Theme extraction (common movie themes)
        theme_keywords = {
            'coming-of-age': ['growing up', 'teenager', 'adolescent', 'youth', 'school'],
            'family': ['family', 'father', 'mother', 'parent', 'child', 'son', 'daughter'],
            'friendship': ['friend', 'friendship', 'buddy', 'companion'],
            'revenge': ['revenge', 'vengeance', 'payback', 'retribution'],
            'love-story': ['love story', 'romance', 'romantic', 'relationship'],
            'survival': ['survival', 'survive', 'stranded', 'wilderness'],
            'war': ['war', 'battle', 'military', 'soldier', 'combat'],
            'time-travel': ['time travel', 'time machine', 'past', 'future']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                extracted_tags.append(theme)
        
        # Extract year-based tags
        decades = re.findall(r'\b(19\d0s|20\d0s)\b', text)
        for decade in decades[:2]:
            extracted_tags.append(decade)
        
        return list(set(extracted_tags))  # Remove duplicates
    
    def enrich_single_movie(self, title: str, year: Optional[int], 
                          existing_description: str = "") -> List[str]:
        """Enrich a single movie using free web sources"""
        all_tags = []
        
        try:
            # Get Wikipedia information
            wiki_info = self.get_wikipedia_info(title, year)
            if wiki_info and wiki_info.get('found'):
                wiki_text = wiki_info.get('text', '') + ' ' + wiki_info.get('summary', '')
                wiki_tags = self.extract_tags_from_text(wiki_text, title)
                all_tags.extend(wiki_tags)
            
            # Get basic IMDb information (with rate limiting)
            if len(all_tags) < 5:  # Only if we need more information
                imdb_info = self.scrape_imdb_basic_info(title, year)
                if imdb_info and imdb_info.get('found'):
                    plot = imdb_info.get('plot', '')
                    if plot:
                        imdb_tags = self.extract_tags_from_text(plot, title)
                        all_tags.extend(imdb_tags)
                    
                    # Add IMDb genres
                    genres = imdb_info.get('genres', [])
                    for genre in genres:
                        all_tags.append(genre.lower().replace(' ', '-'))
            
            # Also analyze existing description
            if existing_description:
                desc_tags = self.extract_tags_from_text(existing_description, title)
                all_tags.extend(desc_tags)
            
            # Clean and deduplicate
            clean_tags = []
            seen = set()
            
            for tag in all_tags:
                if tag and isinstance(tag, str) and len(tag) > 2:
                    clean_tag = tag.strip().lower().replace(' ', '-')
                    if clean_tag not in seen and len(clean_tag) <= 25:
                        clean_tags.append(clean_tag)
                        seen.add(clean_tag)
            
            return clean_tags[:20]  # Top 20 tags
            
        except Exception as e:
            logger.debug(f"Error enriching '{title}': {e}")
            return []
    
    def enrich_dataset_no_api(self, input_csv: str, output_csv: str, 
                            max_movies: Optional[int] = None) -> pd.DataFrame:
        """Enrich dataset using free web sources"""
        logger.info("🆓 Starting NO-API enrichment process...")
        logger.info("📖 Using Wikipedia + respectful web scraping")
        
        # Load data
        df = pd.read_csv(input_csv, encoding='latin-1').fillna('')
        
        if max_movies:
            df = df.head(max_movies)
            logger.info(f"Processing sample of {max_movies} movies")
        
        logger.info(f"Processing {len(df)} movies...")
        
        no_api_tags = []
        successful_enrichments = 0
        
        for idx, row in df.iterrows():
            if idx % 25 == 0:
                logger.info(f"Progress: {idx}/{len(df)} movies processed...")
            
            title = row['title']
            year = None
            try:
                year = int(row['release_year']) if pd.notna(row['release_year']) else None
            except:
                pass
            
            try:
                # Extract tags using free sources
                tags = self.enrich_single_movie(title, year, row.get('description', ''))
                
                if tags:
                    successful_enrichments += 1
                    no_api_tags.append(' | '.join(tags))
                else:
                    no_api_tags.append('')
                
                # Random delay to appear more human-like
                if idx % 5 == 0:
                    time.sleep(random.uniform(1, 3))
                    
            except Exception as e:
                logger.debug(f"Error processing '{title}': {e}")
                no_api_tags.append('')
        
        # Add tags to dataframe
        df['no_api_enhanced_tags'] = no_api_tags
        
        # Save enriched dataset
        df.to_csv(output_csv, index=False, encoding='utf-8')
        
        # Statistics
        enrichment_rate = (successful_enrichments / len(df)) * 100
        avg_tags = 0
        if successful_enrichments > 0:
            non_empty = [tags for tags in no_api_tags if tags]
            avg_tags = sum(len(tags.split('|')) for tags in non_empty) / len(non_empty)
        
        logger.info("✅ NO-API ENRICHMENT COMPLETE!")
        logger.info(f"📁 Saved: {output_csv}")
        logger.info(f"🎯 Enrichment success rate: {enrichment_rate:.1f}%")
        logger.info(f"📈 Average tags per enriched movie: {avg_tags:.1f}")
        logger.info(f"💾 All data cached for offline use")
        
        return df


def main():
    """Main function for no-API enrichment"""
    print("🆓 NO-API WEB ENRICHMENT")
    print("=" * 40)
    print("Uses FREE sources:")
    print("• Wikipedia API (no key needed)")
    print("• Respectful web scraping")
    print("• Public movie databases")
    print("• Smart text analysis")
    print()
    
    # Install required package
    try:
        import wikipediaapi
    except ImportError:
        print("📦 Installing required package...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'wikipedia-api'])
        import wikipediaapi
    
    # Create enricher
    enricher = NoAPIEnrichment()
    
    # Test with sample
    print("🧪 Testing with 10-movie sample...")
    
    sample_df = enricher.enrich_dataset_no_api(
        input_csv="data/netflix_titles_cleaned.csv",
        output_csv="data/netflix_no_api_enriched_sample.csv",
        max_movies=10
    )
    
    # Show results
    print("\n🎯 Sample enriched movies:")
    for i in range(min(3, len(sample_df))):
        row = sample_df.iloc[i]
        tags = row['no_api_enhanced_tags']
        print(f"\n{i+1}. {row['title']} ({row['release_year']})")
        if tags:
            tag_list = tags.split('|')[:6]
            print(f"   Tags: {' | '.join(tag_list)}...")
        else:
            print("   Tags: (none found)")
    
    print(f"\n✅ Sample complete! No API keys needed!")
    print("To process full dataset, modify max_movies parameter")


if __name__ == "__main__":
    main()