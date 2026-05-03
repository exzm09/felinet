"""
Get EVERY links on the page without filtering.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = 'https://www.vet.cornell.edu'
INDEX_URL = f'{BASE_URL}/departments-centers-and-institutes/cornell-feline-health-center/health-information/feline-health-topics'

session = requests.Session()
session.headers.update({'User-Agent': 'FelinNet/0.1'})

resp = session.get(INDEX_URL, timeout=30)
soup = BeautifulSoup(resp.text, 'lxml')

# See ALL links on the page and where they point.
all_links = set()
for a in soup.find_all('a', href=True):
    full = urljoin(BASE_URL, a['href'])
    text = a.get_text(strip=True)[:50]
    all_links.add((text, full))

print(f'Total unique links on page: {len(all_links)}')
print()

# Show only vet.cornell.edu links, grouped by pattern
feline_links = [(t, u) for t, u in all_links if 'vet.cornell.edu' in u]
print(f"vet.cornell.edu links: {len(feline_links)}")
print()

for text, url in sorted(all_links, key=lambda x: x[1]):
    caught = '/feline-health-topics/' in url and url != INDEX_URL
    marker = 'CAUGHT' if caught else 'MISSED'
    print(f'{marker} | {text[:40]:<40} | {url}')
