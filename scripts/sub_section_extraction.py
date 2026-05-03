"""
Find ALL unique links under /health-information/ that are NOT under /feline-health-topics/
These might be category pages or other article sections.
Also look at the RAW HTML structure to see if articles are listed as plain text (not links) on the index page.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = 'https://www.vet.cornell.edu'
INDEX_URL = f'{BASE_URL}/departments-centers-and-institutes/cornell-feline-health-center/health-information/feline-health-topics'

session = requests.Session()
session.headers.update({'User-Agent': 'FeliNet/0.1'})

resp = session.get(INDEX_URL, timeout=30)
soup = BeautifulSoup(resp.text, 'lxml')

other_pages = set()
for a in soup.find_all('a', href=True):
    full = urljoin(BASE_URL, a['href'])
    if 'cornell-feline-health-center/health-information' in full and '/feline-health-topics/' not in full:
        text = a.get_text(strip=True)[:50]
        other_pages.add((text, full))
    

print('=== Non-topic health-information pages ===')
for text, url in sorted(other_pages, key=lambda x: x[1]):
    print(f'  {text:<50} | {url}')

print()

# Find all text that looks like article titles but isn't a link
print('=== Section headings on the page ===')
for tag in soup.find_all(['h2', 'h3', 'h4']):
    text = tag.get_text(strip=True)
    has_link = tag.find('a') is not None
    print(f'  {"[LINKED]" if has_link else "[TEXT]  "} {tag.name}: {text[:60]}')