// zara-category-scraper.js
const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');

const config = {
  outputDir: './zara-data',
  maxProductsPerCategory: 50,
  delayBetweenPages: 3000, 
  screenshot: true, // potential product listing images
  categories: [
    {
      name: 'women',
      url: 'https://www.zara.com/us/en/woman-special-prices-l1314.html?v1=2419737&regionGroupId=41'
    },
    {
      name: 'men',
      url: 'https://www.zara.com/us/en/man-special-prices-l806.html?v1=2436823&regionGroupId=41'
    }
  ]
};

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const randomDelay = (min = 1000, max = 5000) => {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

async function ensureDirectoryExists(directory) {
  try {
    await fs.mkdir(directory, { recursive: true });
  } catch (err) {
    if (err.code !== 'EEXIST') throw err;
  }
}

async function saveJSON(filename, data) {
  const filePath = path.join(config.outputDir, filename);
  await fs.writeFile(filePath, JSON.stringify(data, null, 2));
  console.log(`Data saved to ${filePath}`);
}

// Extract products from a category page
async function extractProductLinks(page, categoryName) {
  console.log(`Extracting product links from ${categoryName} category...`);
  
  if (config.screenshot) {
    await page.screenshot({ path: `${config.outputDir}/${categoryName}-category.png` });
  }

  const productLinks = await page.evaluate(() => {
    const selectors = [
      '.product-link', // General product link class
      'a[data-qa-action="product-link"]', // Product links with data attribute
      '.item a', // Another common pattern
      'article a', // Try article links
      'div[class*="product"] a', // Links inside product divs
      // Add more selectors as needed based on Zara's current structure
    ];
    
    let links = [];
    
    for (const selector of selectors) {
      const elements = document.querySelectorAll(selector);
      if (elements.length > 0) {
        links = Array.from(elements).map(el => el.href);
        break;
      }
    }
    
    // ff standard selectors fail, try to find all links that match product pattern
    if (links.length === 0) {
      links = Array.from(document.querySelectorAll('a'))
        .map(el => el.href)
        .filter(href => href && href.includes('/p') && /p\d+\.html/.test(href));
    }
    
    return [...new Set(links)]; // Remove duplicates
  });
  
  console.log(`Found ${productLinks.length} products in ${categoryName} category`);
  return productLinks.slice(0, config.maxProductsPerCategory);
}

async function scrapeProductPage(page, url, index, category) {
  console.log(`[${category}] Scraping product ${index + 1}: ${url}`);
  
  try {
    await page.goto(url, {
      waitUntil: 'networkidle2',
      timeout: 60000
    });
    
    await page.waitForSelector('body', { timeout: 30000 });
    await delay(5000);
    
    if (config.screenshot) {
      await page.screenshot({ 
        path: `${config.outputDir}/${category}-product-${index + 1}.png` 
      });
    }
    
    const productData = await page.evaluate(() => {
      // extract JSON-LD structured data
      const jsonLdScript = document.querySelector('script[type="application/ld+json"]');
      let structuredData = null;
      
      if (jsonLdScript) {
        try {
          structuredData = JSON.parse(jsonLdScript.textContent);
        } catch (e) {
          console.error('Error parsing JSON-LD');
        }
      }
      
      return {
        structuredData,
        extractedData: {
          productName: document.querySelector('.product-detail-info__header-name')?.innerText.trim() || 
                      document.querySelector('h1')?.innerText.trim() || null,
          price: document.querySelector('.money-amount__main')?.innerText.trim() || 
                document.querySelector('[class*="price"]')?.innerText.trim() || null,
          color: document.querySelector('.product-color-extended-name')?.innerText.trim() || 
                document.querySelector('[class*="color"]')?.innerText.trim() || null,
          description: document.querySelector('.expandable-text__inner-content p')?.innerText.trim() || 
                      document.querySelector('[class*="description"]')?.innerText.trim() || null,
          productCode: document.querySelector('.product-color-extended-name__copy-action')?.innerText.trim() || null,
          // Extract all text that might be useful
          pageText: Array.from(document.querySelectorAll('p, h1, h2, h3, h4, h5, span'))
            .map(el => el.innerText.trim())
            .filter(Boolean)
            .join(' | ')
        }
      };
    });
    
    productData.url = url;
    productData.category = category;
    productData.scrapedAt = new Date().toISOString();
    
    return productData;
  } catch (error) {
    console.error(`Error scraping product ${url}:`, error.message);
    return {
      url,
      category,
      scrapedAt: new Date().toISOString(),
      error: error.message
    };
  }
}

async function scrapeCategory(browser, category) {
  console.log(`\n=== Starting to scrape ${category.name} category ===\n`);
  
  const page = await browser.newPage();
  
  await page.setViewport({ width: 1280, height: 800 });
  await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36');
  
  try {
    console.log(`Navigating to ${category.name} category page...`);
    await page.goto(category.url, { waitUntil: 'networkidle2', timeout: 60000 });
    
    await page.waitForSelector('body', { timeout: 30000 });
    await delay(5000);
    
    const productLinks = await extractProductLinks(page, category.name);
    
    await saveJSON(`${category.name}-product-links.json`, productLinks);
    
    const products = [];
    
    for (let i = 0; i < productLinks.length; i++) {
      try {
        const productData = await scrapeProductPage(page, productLinks[i], i, category.name);
        products.push(productData);
        
        await saveJSON(`${category.name}-products.json`, products);
        
        if (i < productLinks.length - 1) {
          const delayTime = randomDelay();
          console.log(`Waiting ${delayTime}ms before next product...`);
          await delay(delayTime);
        }
      } catch (error) {
        console.error(`Error processing product ${i + 1}:`, error.message);
      }
    }
    
    console.log(`\n=== Finished scraping ${products.length} products from ${category.name} category ===\n`);
    return products;
  } catch (error) {
    console.error(`Error scraping ${category.name} category:`, error);
    return [];
  } finally {
    await page.close();
  }
}

// main
async function main() {
  console.log('Starting Zara scraper...');
  
  await ensureDirectoryExists(config.outputDir);
  
  const browser = await puppeteer.launch({
    headless: "new",
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  console.log('Browser launched');
  
  try {
    const results = {};
    
    for (const category of config.categories) {
      results[category.name] = await scrapeCategory(browser, category);
      
      if (category !== config.categories[config.categories.length - 1]) {
        console.log(`Waiting ${config.delayBetweenPages}ms before next category...`);
        await delay(config.delayBetweenPages);
      }
    }
    
    await saveJSON('all-products.json', results);
    
    console.log('\n=== Scraping completed successfully ===\n');
  } catch (error) {
    console.error('Error in main process:', error);
  } finally {
    await browser.close();
    console.log('Browser closed');
  }
}

main().catch(console.error);