import cloudscraper
import re

scraper = cloudscraper.create_scraper()   # this will solve the JS challenge for you
robots_txt = scraper.get("https://www.vestiairecollective.com/robots.txt").text
sitemaps = re.findall(r"^Sitemap:\s*(.+)$", robots_txt, flags=re.M)
print(sitemaps)
