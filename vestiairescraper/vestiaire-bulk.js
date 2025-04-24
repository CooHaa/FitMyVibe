const puppeteer = require('puppeteer');
const fs        = require('fs');
const path      = require('path');

const delay = ms => new Promise(r => setTimeout(r, ms));
const randomDelay = async (min = 1e3, max = 3e3) => {
  const t = Math.floor(Math.random() * (max - min + 1)) + min;
  console.log(`Waiting ${t} ms …`);
  await delay(t);
};


const looksLikeListing = url => /\.shtml(\?|$)/.test(url);

// ---------- scrolling -------------------------------------------------------
async function scrollUntilProductsFound(page, want, maxScrolls = 30) {
  console.log(`Scroll until ${want} products (≤ ${maxScrolls} scrolls)`);
  let prevCount = 0, stagnation = 0;

  for (let i = 1; i <= maxScrolls; i++) {
    /* 1️⃣ body scroll */
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));

    /* 2️⃣ in-grid scroll (some view-ports have a div-scroller) */
    await page.evaluate(() => {
      const grid = document.querySelector('[data-testid="plp-grid"]');
      if (grid) grid.scrollTop = grid.scrollHeight;
    });

    await delay(2e3);

    const cur = await page.evaluate(() =>
      document.querySelectorAll('a[href$=".shtml"]').length);

    console.log(`scroll #${i}: ${cur} links`);

    if (cur >= want) return true;

    if (cur === prevCount) {
      if (++stagnation >= 3) {
        console.log('No new items – probably end of list.');
        return false;
      }
    } else {
      stagnation = 0;
      prevCount  = cur;
    }
    await randomDelay(1e3, 2e3);
  }
  console.log(`Hit maxScrolls (${maxScrolls})`);
  return false;
}

async function scrapeProductPage(page, url) {
  try {
    await page.goto(url, {waitUntil: 'networkidle2', timeout: 60e3});
    await randomDelay(1e3, 2e3);

    // JSON-LD first …
    const jsonLd = await page.$eval(
      'script[type="application/ld+json"]',
      el => el.textContent,
    ).catch(() => null);

    const product = { url, scrapedAt: new Date().toISOString() };

    if (jsonLd) {
      try {
        product.structuredData = JSON.parse(jsonLd);
        const sd = product.structuredData;
        product.formattedData = {
          productName: sd.name,
          brand:       sd.brand?.name,
          price:       sd.offers?.price,
          currency:    sd.offers?.priceCurrency,
          description: sd.description,
          image:       sd.image,
          availability:sd.offers?.availability,
          seller:      sd.seller?.name,
          condition:   sd.itemCondition,
        };
      } catch (e) {
        console.warn('bad JSON-LD:', e.message);
      }
    }

    if (!product.formattedData) {
      product.formattedData = await page.evaluate(() => {
        // … same code you already had …
      });
    }

    return product;
  } catch (e) {
    console.error(`product-scrape fail: ${url} – ${e.message}`);
    return { url, error: e.message, formattedData: {} };
  }
}

async function scrapeListingsPage(page, want = 0) {
  await delay(5e3);                  // wait for JS
  if (want) await scrollUntilProductsFound(page, want);

  const listings = await page.evaluate(() => {
    const extractPriceInfo = text => {
      if (!text) return {price:null,currency:null};
      const m = text.match(/([$€£¥])\s*([\d,.]+)/);
      if (!m) return {price:null,currency:null};
      const cur = { '$':'USD', '€':'EUR', '£':'GBP', '¥':'JPY' }[m[1]] || m[1];
      return {price:m[2].replace(/,/g,''), currency:cur};
    };

    const cards = Array.from(document.querySelectorAll('a[href$=".shtml"]'));
    return cards.map(card => {
      const url = card.href;
      /* Basic title/seller & price sniffing (shortened) … */
      const container = card.closest('article') || card;

      const title = (container.querySelector('h3,h2,h1')||{}).textContent?.trim()
                 || card.title || null;

      const img   = (card.querySelector('img')||{}).src||null;

      const {price,currency} = extractPriceInfo(container.textContent);

      return { url, title, price, currency, image: img };
    });
  });

  console.log(`listings extracted: ${listings.length}`);
  return listings;
}


const scrapeMultiplePages = async (page, startUrl, maxPages = 3, targetProductCount = 0) => {
  let allListings = [];
  let currentUrl = startUrl;
  let pageCounter = 1;
  
  while (pageCounter <= maxPages) {
    console.log(`Scraping page ${pageCounter}: ${currentUrl}`);
    
    try {
      await page.goto(currentUrl, {
        waitUntil: 'networkidle0',
        timeout: 90000 
      });
    } catch (error) {
      console.log(`Navigation timeout or error: ${error.message}`);
      console.log('Continuing anyway as the page may have partially loaded...');
    }
    
    
    console.log('Waiting for content to settle...');
    await delay(8000); // Longer initial wait
    
    
    const pageListings = await scrapeListingsPage(page, targetProductCount);
    console.log(`Found ${pageListings.length} listings on page ${pageCounter}`);

    allListings = [...allListings, ...pageListings];
 
    if (pageListings.length === 0 && pageCounter === 1) {
      console.log('No listings found with standard method, trying alternative approach...');
      
      const productLinks = await page.evaluate(() => {
        const links = Array.from(document.querySelectorAll('a[href*="/products/"]'));
        return links.map(link => {
          const url = link.href;
          const urlParts = url.split('/');
          const productsIndex = urlParts.indexOf('products');
          let seller = null;
          if (productsIndex > 0 && productsIndex < urlParts.length - 1) {
            const productSlug = urlParts[productsIndex + 1];
            const parts = productSlug.split('-');
            if (parts.length > 0) {
              seller = parts[0]; // First part of the slug is usually the seller username
            }
          }
          
          let title = null;
          if (productsIndex > 0 && productsIndex < urlParts.length - 1) {
            const productSlug = urlParts[productsIndex + 1];
            const titleParts = productSlug.split('-');
            if (titleParts.length > 1) {
              // Join all parts after the seller name with spaces
              title = titleParts.slice(1).join(' ').replace(/-/g, ' ');
            }
          }
          
          return {
            url,
            title,
            seller
          };
        }).filter(item => item.url);
      });
      
      console.log(`Found ${productLinks.length} product links with alternative method`);
      allListings = [...allListings, ...productLinks];
    }
    
    // Look for next page link
    const hasNextPage = await page.evaluate(() => {
      // Try different selectors for pagination
      const nextSelectors = [
        '[data-testid="pagination-next"]', 
        '.pagination-next', 
        'a[rel="next"]',
        '.next a',
        'a[aria-label="Next page"]',
        'button[aria-label="Next page"]'
      ];
      
      for (const selector of nextSelectors) {
        try {
          const nextLink = document.querySelector(selector);
          if (nextLink && !nextLink.disabled && !nextLink.classList.contains('disabled')) {
            return nextLink.href || null;
          }
        } catch (e) {
          console.log(`Error with selector ${selector}: ${e.message}`);
        }
      }
      
      // Try to find pagination by context - look for a group of numbered links
      const paginationLinks = Array.from(document.querySelectorAll('nav a, [role="navigation"] a, [class*="pagination"] a'))
        .filter(a => a.innerText.trim().match(/^\d+$/));
      
      if (paginationLinks.length > 0) {
        // Find current page number
        const currentPage = Array.from(document.querySelectorAll('a.active, [aria-current="page"], a[class*="current"]'))
          .find(a => a.innerText.trim().match(/^\d+$/));
        
        if (currentPage) {
          const currentNum = parseInt(currentPage.innerText.trim());
          // Find link to next page
          const nextPageLink = paginationLinks.find(a => parseInt(a.innerText.trim()) === currentNum + 1);
          return nextPageLink ? nextPageLink.href : null;
        }
      }
      
      return null;
    });
    
    if (!hasNextPage || pageCounter >= maxPages) {
      console.log('No more pages to scrape or reached max pages limit');
      break;
    }
    
    currentUrl = hasNextPage;
    pageCounter++;
    
    await randomDelay(3000, 6000);
  }
  
  return allListings;
};

const getDetailedProductInfo = async (browser, listings) => {
  console.log(`\nGetting detailed information for all ${listings.length} products...`);
  
  const detailedProducts = [];
  const chunkSize = 5; 
  
  for (let i = 0; i < listings.length; i += chunkSize) {
    const chunk = listings.slice(i, i + chunkSize);
    console.log(`Processing chunk ${Math.floor(i/chunkSize) + 1}/${Math.ceil(listings.length/chunkSize)} (${chunk.length} products)`);
    
    const promises = chunk.map(async (listing, index) => {
      try {
        const page = await browser.newPage();
        
        await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15');
        await page.setExtraHTTPHeaders({
          'Accept-Language': 'en-US,en;q=0.9',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8'
        });
        
        console.log(`Processing product ${i + index + 1}/${listings.length}: ${listing.url}`);
        
        // Get detailed product info
        const productData = await scrapeProductPage(page, listing.url);
        
        // Create a complete product record
        const detailedProduct = {
          ...listing,
          detailed: productData.formattedData,
          structuredData: productData.structuredData
        };
        
        if (productData.formattedData) {
          if (productData.formattedData.productName && !listing.title) {
            detailedProduct.title = productData.formattedData.productName;
          }
          
          if (productData.formattedData.price && !listing.price) {
            detailedProduct.price = productData.formattedData.price;
          }
          
          if (productData.formattedData.currency && !listing.currency) {
            detailedProduct.currency = productData.formattedData.currency;
          }
          
          if (productData.formattedData.seller && !listing.seller) {
            detailedProduct.seller = productData.formattedData.seller;
          }
        }
        
        await page.close();
        
        return detailedProduct;
      } catch (error) {
        console.error(`Error processing ${listing.url}:`, error.message);
        return {
          ...listing,
          error: error.message
        };
      }
    });
    
    const chunkResults = await Promise.all(promises);
    detailedProducts.push(...chunkResults);
    
    if (i + chunkSize < listings.length) {
      console.log('Pausing between chunks to avoid rate limiting...');
      await randomDelay(5000, 10000);
    }
  }
  
  return detailedProducts;
};


(async () => {
  const outputDir = path.join(__dirname, 'output');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir);
  }
  
  console.log('Starting Vestiaire scraper...');
  console.log('Output will be saved to', outputDir);
  
  const browser = await puppeteer.launch({
    headless: true, 
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--window-size=1920,1080',
      '--disable-web-security'
    ],
    defaultViewport: null
  });
  
  console.log('Browser launched');
  
  try {
    const page = await browser.newPage();
    
    page.on('console', msg => console.log('BROWSER CONSOLE:', msg.text()));
    
    await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15');
    
    await page.setExtraHTTPHeaders({
      'Accept-Language': 'en-US,en;q=0.9',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8'
    });
    
    await randomDelay(2000, 5000);
    
    const targetProductCount = 30;
    
    const listingsPageUrl = 'https://us.vestiairecollective.com/men-bags/#categoryParent=Bags%23141_gender=Men%232';
    
    console.log(`Starting with URL: ${listingsPageUrl}`);
    console.log(`Target product count: ${targetProductCount}`);
    
    const maxPagesToScrape = 1; 
    const listings = await scrapeMultiplePages(page, listingsPageUrl, maxPagesToScrape, targetProductCount);
    
    console.log(`Total products found: ${listings.length}`);
    
    if (listings.length > 0) {
      const detailedProducts = await getDetailedProductInfo(browser, listings);
      
      const categoryMatch = listingsPageUrl.match(/category\/([^\/]+)\/([^\/]+)/);
      const categoryString = categoryMatch ? `${categoryMatch[1]}-${categoryMatch[2]}` : 'category';
      
      fs.writeFileSync(path.join(outputDir, `Vestiaire-mens-accessories.json`), JSON.stringify(detailedProducts, null, 2));
      console.log(`All ${detailedProducts.length} products with detailed info saved to output/Vestiaire-${categoryString}.json`);
    } else {
      console.log('No products found. Try a different category or URL.');
    }
    
  } catch (error) {
    console.error('Error during scraping:', error);
  } finally {
    await browser.close();
    console.log('Browser closed');
  }
})();