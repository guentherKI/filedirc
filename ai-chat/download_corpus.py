"""
Download high-quality text corpus for AI training
"""
import requests
import os
import time

def download_gutenberg_book(book_id, filename):
    """Download a book from Project Gutenberg"""
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    print(f"📥 Downloading: {filename}...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save to extra_corpus
        filepath = os.path.join('extra_corpus', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"✅ Saved {filename} ({len(response.text)} chars)")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def download_wikipedia_article(title, filename):
    """Download Wikipedia article as plain text"""
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'titles': title,
        'prop': 'extracts',
        'explaintext': True,
        'exsectionformat': 'plain'
    }
    
    print(f"📥 Downloading Wikipedia: {title}...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        pages = data['query']['pages']
        page = list(pages.values())[0]
        
        if 'extract' in page:
            text = page['extract']
            filepath = os.path.join('extra_corpus', filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"✅ Saved {filename} ({len(text)} chars)")
            return True
        else:
            print(f"❌ No content found for {title}")
            return False
    except Exception as e:
        print(f"❌ Failed to download {title}: {e}")
        return False

# Create extra_corpus directory if it doesn't exist
os.makedirs('extra_corpus', exist_ok=True)

print("=" * 70)
print("📚 CORPUS DOWNLOADER")
print("=" * 70)

# Download Classic Literature (Public Domain)
print("\n📖 Classic Literature:")
books = [
    (1342, "pride_and_prejudice.txt"),  # Pride and Prejudice
    (84, "frankenstein.txt"),            # Frankenstein
    (1661, "sherlock_holmes.txt"),       # Sherlock Holmes Adventures
    (2701, "moby_dick.txt"),             # Moby Dick (partial - long)
]

for book_id, filename in books:
    download_gutenberg_book(book_id, filename)
    time.sleep(1)  # Be polite to servers

# Download Wikipedia Articles (Diverse Topics)
print("\n🌍 Wikipedia Knowledge:")
wiki_articles = [
    ("Artificial intelligence", "wiki_ai.txt"),
    ("Machine learning", "wiki_machine_learning.txt"),
    ("Natural language processing", "wiki_nlp.txt"),
    ("Computer programming", "wiki_programming.txt"),
    ("Python (programming language)", "wiki_python.txt"),
    ("History of the Internet", "wiki_internet.txt"),
    ("Philosophy", "wiki_philosophy.txt"),
    ("Science", "wiki_science.txt"),
    ("Mathematics", "wiki_mathematics.txt"),
    ("Literature", "wiki_literature.txt"),
]

for title, filename in wiki_articles:
    download_wikipedia_article(title, filename)
    time.sleep(1)

print("\n" + "=" * 70)
print("✅ Download Complete!")
print("=" * 70)

# Calculate total size
total_chars = 0
for filename in os.listdir('extra_corpus'):
    if filename.endswith('.txt'):
        filepath = os.path.join('extra_corpus', filename)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            total_chars += len(f.read())

print(f"\n📊 Total Corpus Size: {total_chars:,} characters ({total_chars / 1024 / 1024:.2f} MB)")
print(f"🚀 Ready for training!")
