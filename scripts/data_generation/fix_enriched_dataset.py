#!/usr/bin/env python3
"""
Fix Enriched Dataset Issue

Quick fix: Use the existing full model-enriched dataset but improve the tags
to be more semantic and search-friendly.
"""

import pandas as pd
import re
from typing import List

def improve_tags(tag_string: str) -> str:
    """Improve existing tags to be more search-friendly"""
    if pd.isna(tag_string) or not tag_string:
        return ""
    
    tags = [tag.strip() for tag in str(tag_string).split('|') if tag.strip()]
    improved_tags = []
    
    for tag in tags:
        # Clean up tag format
        clean_tag = tag.lower().strip()
        
        # Skip very short or generic tags
        if len(clean_tag) <= 2:
            continue
            
        # Convert names to searchable format
        if any(char.isupper() for char in tag):
            # This looks like a name - convert to searchable format
            name_parts = clean_tag.replace('-', ' ').split()
            if len(name_parts) >= 2:
                # Add full name and last name for director searches
                full_name = '-'.join(name_parts)
                last_name = name_parts[-1]
                improved_tags.extend([full_name, last_name])
            else:
                improved_tags.append(clean_tag)
        else:
            improved_tags.append(clean_tag)
    
    # Remove duplicates while preserving order
    unique_tags = []
    seen = set()
    for tag in improved_tags:
        if tag not in seen and len(tag) > 2:
            unique_tags.append(tag)
            seen.add(tag)
    
    return ' | '.join(unique_tags[:25])  # Limit to 25 tags

def main():
    """Fix the existing enriched dataset"""
    print("🔧 FIXING ENRICHED DATASET TAGS")
    print("=" * 40)
    
    # Load existing enriched dataset
    input_file = "data/netflix_titles_enriched.csv"
    
    try:
        print(f"📖 Loading: {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8')
        
        if 'semantic_tags' not in df.columns:
            print("❌ No semantic_tags column found!")
            return
            
        print(f"📊 Dataset size: {len(df)} movies")
        
        # Count existing tags
        non_empty_before = df[df['semantic_tags'] != '']['semantic_tags']
        if len(non_empty_before) > 0:
            avg_before = non_empty_before.str.split('|').str.len().mean()
            print(f"📈 Average tags before: {avg_before:.1f}")
        
        print("🔄 Improving tag quality...")
        
        # Improve tags
        df['semantic_tags'] = df['semantic_tags'].apply(improve_tags)
        
        # Add additional semantic tags from existing data
        for idx, row in df.iterrows():
            existing_tags = row['semantic_tags']
            additional_tags = []
            
            # Add decade tags
            try:
                year = int(row['release_year'])
                decade = (year // 10) * 10
                additional_tags.append(f"{decade}s")
                
                if year >= 2020:
                    additional_tags.append("recent")
                elif year >= 2010:
                    additional_tags.append("modern")
                else:
                    additional_tags.append("classic")
            except:
                pass
            
            # Add content type
            content_type = str(row.get('type', '')).lower()
            if content_type in ['movie', 'tv show']:
                additional_tags.append(content_type.replace(' ', '-'))
            
            # Add genre tags (cleaned up)
            genres = str(row.get('listed_in', ''))
            if genres:
                for genre in genres.split(',')[:3]:  # Top 3 genres
                    clean_genre = genre.strip().lower().replace(' ', '-')
                    if len(clean_genre) > 3:
                        additional_tags.append(clean_genre)
            
            # Combine with existing tags
            if existing_tags:
                combined = existing_tags + ' | ' + ' | '.join(additional_tags)
            else:
                combined = ' | '.join(additional_tags)
            
            df.at[idx, 'semantic_tags'] = combined
        
        # Final cleanup pass
        df['semantic_tags'] = df['semantic_tags'].apply(improve_tags)
        
        # Statistics after improvement
        non_empty_after = df[df['semantic_tags'] != '']['semantic_tags']
        success_rate = len(non_empty_after) / len(df) * 100
        
        if len(non_empty_after) > 0:
            avg_after = non_empty_after.str.split('|').str.len().mean()
            print(f"📈 Average tags after: {avg_after:.1f}")
        
        print(f"✅ Enrichment success rate: {success_rate:.1f}%")
        
        # Save improved dataset
        output_file = "data/netflix_titles_enriched_fixed.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"📁 Saved improved dataset: {output_file}")
        
        # Show sample
        print("\n🎯 Sample improved movies:")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            tags = row['semantic_tags']
            print(f"\n{i+1}. {row['title']} ({row['release_year']})")
            if tags:
                tag_list = tags.split('|')[:8]
                print(f"   Tags: {' | '.join([t.strip() for t in tag_list])}...")
        
        print(f"\n🚀 To use this dataset:")
        print(f"   cp {output_file} data/netflix_titles_enriched.csv")
        print(f"   pipenv run python src/run_comprehensive_evaluation.py")
        
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        print("Generate it first with:")
        print("   pipenv run python src/preprocessing/model_based_tagger.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()