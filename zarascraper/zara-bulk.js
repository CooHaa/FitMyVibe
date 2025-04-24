// zara-bulk.js — Zara category scraper (fixed gallery filtering)
// ------------------------------------------------------------
//  • No pageText, single-size variant
//  • Collects every JPG/JPEG in gallery (relaxed regex)
//  • Picks 3rd‑to‑last image when available, else first
// ------------------------------------------------------------

const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');

const config = {
  outputDir: './zara-data',
  maxProductsPerCategory: 120,
  delayBetweenPages: 3000,
  screenshot: false,
  categories: [
    {
      name: 'mens-all',
      url: 'https://www.zara.com/us/en/man-all-products-l7465.html?v1=2458839&regionGroupId=41'
    }
  ]
};

// ------------------------------------------------------------
//  utilities
// ------------------------------------------------------------
const delay = (ms) => new Promise((r) => setTimeout(r, ms));
const rand = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
const ensureDir = async (d) => { try { await fs.mkdir(d, { recursive: true }); } catch (e) { if (e.code !== 'EEXIST') throw e; } };
const saveJSON = async (f, data) => { const fp = path.join(config.outputDir, f); await fs.writeFile(fp, JSON.stringify(data, null, 2)); console.log(`⇢ ${fp}`); };

// ------------------------------------------------------------
//  scrolling helpers
// ------------------------------------------------------------
async function autoScroll(page, step = 1000) {
  await page.evaluate(async (s) => {
    await new Promise((res) => {
      let total = 0;
      const timer = setInterval(() => {
        window.scrollBy(0, s);
        total += s;
        if (total >= document.documentElement.scrollHeight - window.innerHeight) {
          clearInterval(timer);
          res();
        }
      }, 180);
    });
  }, step);
}

function collectLinks() {
  const sels = ['.product-link', 'a[data-qa-action="product-link"]', '.item a', 'article a', 'div[class*="product"] a'];
  for (const sel of sels) {
    const els = document.querySelectorAll(sel);
    if (els.length) return [...new Set([...els].map((e) => e.href))];
  }
  return [...new Set([...document.querySelectorAll('a')].map((e) => e.href).filter((h) => /\/p\d+.*\.html/i.test(h)))];
}

async function extractProductLinks(page, cat) {
  const set = new Set();
  let last = 0;
  while (set.size < config.maxProductsPerCategory) {
    (await page.evaluate(collectLinks)).forEach((h) => set.add(h));
    if (set.size >= config.maxProductsPerCategory || set.size === last) break;
    last = set.size;
    await autoScroll(page);
    await delay(2000);
  }
  return [...set].slice(0, config.maxProductsPerCategory);
}

// ------------------------------------------------------------
//  product scraping
// ------------------------------------------------------------
async function scrapeProduct(page, url, idx, cat) {
  console.log(`   #${idx + 1}/${cat}  ${url}`);
  try {
    await page.goto(url, { waitUntil: 'networkidle2', timeout: 60_000 });
    await delay(3500);

    const product = await page.evaluate(() => {
      const strip = (u) => (u ? u.split('?')[0] : u);
      const isJpg = (u) => /\.jpe?g($|\?)/i.test(u);
      const bad = /transparent-background|logos|powered_by_logo|track\.php/i;

      // structured data (first variant)
      let sd = null;
      const tag = document.querySelector('script[type="application/ld+json"]');
      if (tag) { try { sd = JSON.parse(tag.textContent); } catch {} }
      if (Array.isArray(sd)) sd = sd[0];

      // collect image URLs from <img>, <source>, and og:image
      const imgs = [...document.querySelectorAll('img')];
      const sources = [...document.querySelectorAll('source')];
      const metas = [...document.querySelectorAll('meta[property="og:image"]')];

      let urls = imgs.map((i) => i.src || i.dataset.src).concat(
        sources.flatMap((s) => s.srcset.split(',').map((c) => c.trim().split(' ')[0])),
        metas.map((m) => m.content || '')
      );
      if (sd?.image) urls = urls.concat(Array.isArray(sd.image) ? sd.image : [sd.image]);

      const pics = [...new Set(urls.map(strip).filter(Boolean))].filter((u) => isJpg(u) && !bad.test(u));
      const pick = pics.length >= 3 ? pics[pics.length - 3] : pics[0] || null;

      return {
        structuredData: sd,
        extractedData: {
          productName: document.querySelector('.product-detail-info__header-name')?.innerText.trim() || document.querySelector('h1')?.innerText.trim() || null,
          price: document.querySelector('.money-amount__main')?.innerText.trim() || document.querySelector('[class*="price"]')?.innerText.trim() || null,
          color: document.querySelector('.product-color-extended-name')?.innerText.trim() || document.querySelector('[class*="color"]')?.innerText.trim() || null,
          description: document.querySelector('.expandable-text__inner-content p')?.innerText.trim() || document.querySelector('[class*="description"]')?.innerText.trim() || null,
          productCode: document.querySelector('.product-color-extended-name__copy-action')?.innerText.trim() || null,
          size: sd?.size || null,
          selectedImageUrl: pick,
          allImages: pics
        }
      };
    });

    return { ...product, url, category: cat, scrapedAt: new Date().toISOString() };
  } catch (e) {
    console.error(`   ✗ ${e.message}`);
    return { url, category: cat, scrapedAt: new Date().toISOString(), error: e.message };
  }
}

// ------------------------------------------------------------
//  category handler
// ------------------------------------------------------------
async function scrapeCategory(browser, cat) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });
  await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36');
  await page.goto(cat.url, { waitUntil: 'networkidle2', timeout: 60_000 });
  await delay(3000);

  const links = await extractProductLinks(page, cat.name);
  await saveJSON(`${cat.name}-product-links.json`, links);

  const products = [];
  for (let i = 0; i < links.length; i++) {
    const data = await scrapeProduct(page, links[i], i, cat.name);
    products.push(data);
    await saveJSON(`${cat.name}-products.json`, products);
    if (i < links.length - 1) await delay(rand(1000, 4000));
  }
  await page.close();
  return products;
}

// ------------------------------------------------------------
//  main
// ------------------------------------------------------------
(async () => {
  console.log('▶ Starting Zara scraper');
  await ensureDir(config.outputDir);
  const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  try {
    const all = {};
    for (const cat of config.categories) {
      all[cat.name] = await scrapeCategory(browser, cat);
      if (cat !== config.categories[config.categories.length - 1]) await delay(config.delayBetweenPages);
    }
    await saveJSON('all-products.json', all);
    console.log('✔ Done');
  } catch (err) {
    console.error('Fatal:', err);
  } finally {
    await browser.close();
    console.log('Browser closed');
  }
})();
