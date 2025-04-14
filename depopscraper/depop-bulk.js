const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const randomDelay = async (min = 1000, max = 3000) => {
  const delayTime = Math.floor(Math.random() * (max - min)) + min;
  console.log(`Waiting ${delayTime}ms...`);
  await delay(delayTime);
};

async function scrollUntilProductsFound(page, targetCount, maxScrolls = 30) {
  console.log(`Scrolling until ${targetCount} products are found (max ${maxScrolls} scrolls)...`);
  
  let previousProductCount = 0;
  let scrollsWithoutNewProducts = 0;
  let scrollCount = 0;
  
  while (scrollCount < maxScrolls) {
    scrollCount++;
    
    // Scroll down
    await page.evaluate(() => {
      window.scrollTo(0, document.body.scrollHeight);
    });
    
    await delay(2000); 
    
    const currentProductCount = await page.evaluate(() => {
      return document.querySelectorAll('a[href*="/products/"]').length;
    });
    
    console.log(`Scroll #${scrollCount}: Found ${currentProductCount} products`);
    
    if (currentProductCount >= targetCount) {
      console.log(`Target of ${targetCount} products reached!`);
      return true;
    }
    
    if (currentProductCount === previousProductCount) {
      scrollsWithoutNewProducts++;
      
      if (scrollsWithoutNewProducts >= 3) {
        console.log('No new products loaded after multiple scrolls. Probably reached the end.');
        return false;
      }
    } else {
      scrollsWithoutNewProducts = 0;
      previousProductCount = currentProductCount;
    }
    
    await randomDelay(1000, 2000);
  }
  
  console.log(`Reached maximum number of scrolls (${maxScrolls}) without finding ${targetCount} products.`);
  return false;
}

const scrapeProductPage = async (page, url) => {
  try {
    await page.goto(url, {
      waitUntil: 'networkidle2', 
      timeout: 60000 
    });
    
    console.log(`Scraping product at: ${url}`);
    
    // Add a small random delay
    await randomDelay(1000, 2000);
    
    const jsonLd = await page.evaluate(() => {
      const script = document.querySelector('script[type="application/ld+json"]');
      return script ? script.textContent : null;
    });
    
    let productData = {
      url: url,
      scrapedAt: new Date().toISOString()
    };
    
    if (jsonLd) {
      console.log('JSON-LD data found');
      try {
        productData.structuredData = JSON.parse(jsonLd);
        
        productData.formattedData = {
          productName: productData.structuredData.name,
          brand: productData.structuredData.brand?.name,
          price: productData.structuredData.offers?.price,
          currency: productData.structuredData.offers?.priceCurrency,
          description: productData.structuredData.description,
          image: productData.structuredData.image,
          availability: productData.structuredData.offers?.availability,
          seller: productData.structuredData.seller?.name || extractSellerFromUrl(url),
          condition: productData.structuredData.itemCondition,
          size: extractSizeFromDescription(productData.structuredData.description)
        };
      } catch (e) {
        console.error('Error parsing JSON-LD:', e.message);
      }
    } else {
      console.log('No JSON-LD data found, trying HTML extraction');
      
      // Fallback to HTML scraping
      productData.formattedData = await page.evaluate(() => {
        // Extract product information from HTML elements
        const getTextContent = (selector) => {
          const element = document.querySelector(selector);
          return element ? element.textContent.trim() : null;
        };
        
        const productName = 
          getTextContent('h1') || 
          getTextContent('.product-name') || 
          getTextContent('[data-testid="product-name"]');
        
        const brandElement = 
          document.querySelector('.designer-name') || 
          document.querySelector('[data-testid="product-brand"]');
        const brand = brandElement ? brandElement.textContent.trim() : null;
        
        const priceElement = 
          document.querySelector('.price') || 
          document.querySelector('[data-testid="product-price"]') ||
          document.querySelector('.product-price');
        const priceText = priceElement ? priceElement.textContent.trim() : null;
        
        let price = null;
        let currency = null;
        if (priceText) {
          const priceMatch = priceText.match(/[\d,.]+/);
          price = priceMatch ? priceMatch[0].replace(/,/g, '') : null;
          
          const currencyMatch = priceText.match(/[$€£¥]/);
          currency = currencyMatch ? currencyMatch[0] : null;
          if (currency === '$') currency = 'USD';
          if (currency === '€') currency = 'EUR';
          if (currency === '£') currency = 'GBP';
          if (currency === '¥') currency = 'JPY';
        }
        
        const description = 
          getTextContent('.product-description') || 
          getTextContent('[data-testid="product-description"]');
        
        const imageElement = 
          document.querySelector('.product-image') || 
          document.querySelector('[data-testid="product-image"]') ||
          document.querySelector('img[srcset*="depop"]');
        
        const image = imageElement ? 
          (imageElement.src || 
           (imageElement.srcset && imageElement.srcset.split(',').pop().trim().split(' ')[0])) : 
          null;
        
        const sellerElement = document.querySelector('[data-testid="username"], .username, .seller-name');
        const seller = sellerElement ? sellerElement.textContent.trim() : null;
        
        const sizeElement = document.querySelector('[data-testid="size"], .size, [class*="size"]');
        const size = sizeElement ? sizeElement.textContent.trim() : null;
        
        const conditionElement = document.querySelector('[data-testid="condition"], .condition, [class*="condition"]');
        const condition = conditionElement ? conditionElement.textContent.trim() : null;
        
        return {
          productName,
          brand,
          price,
          currency,
          description,
          image,
          seller,
          size,
          condition
        };
      });
    }
    
    if (!productData.formattedData.seller) {
      productData.formattedData.seller = extractSellerFromUrl(url);
    }
    
    return productData;
  } catch (error) {
    console.error(`Error scraping ${url}:`, error.message);
    return {
      url: url,
      error: error.message,
      formattedData: {}
    };
  }
};

function extractSellerFromUrl(url) {
  try {
    const urlParts = url.split('/');
    const productsIndex = urlParts.indexOf('products');
    if (productsIndex > 0 && productsIndex < urlParts.length - 1) {
      const productSlug = urlParts[productsIndex + 1];
      const parts = productSlug.split('-');
      if (parts.length > 0) {
        return parts[0]; 
      }
    }
  } catch (e) {
    console.log('Error extracting seller from URL:', e.message);
  }
  return null;
}

function extractSizeFromDescription(description) {
  if (!description) return null;
  
  // Common size patterns
  const sizePatterns = [
    /size\s*:\s*([a-zA-Z0-9\s\/\-\.]+)/i,  // "Size: M" or "Size: 10"
    /size\s*([a-zA-Z0-9\s\/\-\.]+)/i,      // "Size M" or "Size 10"
    /\b(xs|s|m|l|xl|xxl|xxxl|small|medium|large|x-large)\b/i, // Common size abbreviations
    /\bsize\s*\(([^)]+)\)/i,               // "Size (Medium)" or "Size (10)"
    /\b(one size)\b/i                      // "One size"
  ];
  
  for (const pattern of sizePatterns) {
    const match = description.match(pattern);
    if (match && match[1]) {
      return match[1].trim();
    }
  }
  
  return null;
}

const scrapeListingsPage = async (page, targetProductCount = 0) => {
  console.log('Extracting product listings from page...');
    
  console.log('Waiting for page content to fully load...');
  await delay(5000);
  
  if (targetProductCount > 0) {
    await scrollUntilProductsFound(page, targetProductCount);
  } else {
    console.log('Scrolling to load more content...');
    await page.evaluate(async () => {
      for (let i = 0; i < 5; i++) {
        window.scrollBy(0, window.innerHeight);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      window.scrollTo(0, 0);
    });
    
    // Wait after scrolling
    await delay(2000);
  }
  
  const pageContent = await page.content();
  fs.writeFileSync('debug-page-html.txt', pageContent);
  console.log('Page HTML saved to debug-page-html.txt');
  
  const hasProductLinks = await page.evaluate(() => {
    const productLinks = document.querySelectorAll('a[href*="/products/"]');
    console.log(`Found ${productLinks.length} product links`);
    return productLinks.length;
  });
  console.log(`Found ${hasProductLinks} raw product links in the page`);
  
  const possibleSelectors = [
    'main a[href*="/products/"]',
    'article a[href*="/products/"]',
    'a[href*="/products/"]',
    'div[class*="product"] a'
  ];
  
  for (const selector of possibleSelectors) {
    const count = await page.evaluate((sel) => {
      return document.querySelectorAll(sel).length;
    }, selector);
    console.log(`Selector "${selector}" found ${count} elements`);
  }
  
  let workingSelector = null;
  for (const selector of possibleSelectors) {
    const count = await page.evaluate((sel) => {
      return document.querySelectorAll(sel).length;
    }, selector);
    if (count > 0) {
      workingSelector = selector;
      console.log(`Using selector: ${workingSelector}`);
      break;
    }
  }
  
  if (!workingSelector) {
    console.log('WARNING: No working selector found. Page structure may have changed.');
    console.log('Falling back to generic anchor elements...');
    workingSelector = 'a';
  }
  
  const listings = await page.evaluate((selector) => {
    // Helper function to extract price and currency from a text
    const extractPriceInfo = (text) => {
      if (!text) return { price: null, currency: null };
      
      const priceRegex = /[$€£¥]\s*[\d,.]+|\d+\s*[$€£¥]/;
      const priceMatch = text.match(priceRegex);
      
      if (!priceMatch) return { price: null, currency: null };
      
      const numericMatch = priceMatch[0].match(/[\d,.]+/);
      const price = numericMatch ? numericMatch[0].replace(/,/g, '') : null;
      
      const currencyMatch = priceMatch[0].match(/[$€£¥]/);
      let currency = currencyMatch ? currencyMatch[0] : null;
      
      if (currency === '$') currency = 'USD';
      if (currency === '€') currency = 'EUR';
      if (currency === '£') currency = 'GBP';
      if (currency === '¥') currency = 'JPY';
      
      return { price, currency };
    };
    
    const productCards = Array.from(document.querySelectorAll(selector))
      .filter(el => el.href && el.href.includes('/products/'));
    
    console.log(`Processing ${productCards.length} cards with selector ${selector}`);
    
    return productCards.map(card => {
      const url = card.href;
      
      const urlParts = url.split('/');
      const productsIndex = urlParts.indexOf('products');
      let seller = null;
      if (productsIndex > 0 && productsIndex < urlParts.length - 1) {
        const productSlug = urlParts[productsIndex + 1];
        const parts = productSlug.split('-');
        if (parts.length > 0) {
          seller = parts[0];
        }
      }
      
      const container = card.closest('article') || 
                        card.closest('div[role="listitem"]') || 
                        card.closest('div');
      
      let title = null;
      
      if (card.textContent && card.textContent.trim().length > 3 && card.textContent.trim().length < 100) {
        title = card.textContent.trim();
      }
      
      if (!title && container) {
        const titleElement = container.querySelector('h1, h2, h3, h4, h5, h6, [class*="title"], [class*="name"], strong, p');
        if (titleElement && titleElement.textContent.trim()) {
          title = titleElement.textContent.trim();
        }
      }
      
      if (!title) {
        try {
          const urlParts = url.split('/');
          const productsIndex = urlParts.indexOf('products');
          if (productsIndex > 0 && productsIndex < urlParts.length - 1) {
            const productSlug = urlParts[productsIndex + 1];
            const titleParts = productSlug.split('-');
            if (titleParts.length > 1) {
              title = titleParts.slice(1).join(' ').replace(/-/g, ' ');
            }
          }
        } catch (e) {
          console.log('Error extracting title from URL');
        }
      }
      
      let image = null;
      const imgInLink = card.querySelector('img');
      if (imgInLink) {
        image = imgInLink.src || imgInLink.dataset.src;
        if (!image && imgInLink.srcset) {
          const srcset = imgInLink.srcset.split(',');
          if (srcset.length > 0) {
            const lastSrc = srcset[srcset.length - 1].trim().split(' ')[0];
            image = lastSrc;
          }
        }
      }
      
      if (!image && container) {
        const imgInContainer = container.querySelector('img');
        if (imgInContainer) {
          image = imgInContainer.src || imgInContainer.dataset.src;
          if (!image && imgInContainer.srcset) {
            const srcset = imgInContainer.srcset.split(',');
            if (srcset.length > 0) {
              const lastSrc = srcset[srcset.length - 1].trim().split(' ')[0];
              image = lastSrc;
            }
          }
        }
      }
      
      let price = null;
      let currency = null;
      
      if (container) {
        const priceElements = container.querySelectorAll('[class*="price"]');
        if (priceElements.length > 0) {
          for (const el of priceElements) {
            if (el.textContent) {
              const priceInfo = extractPriceInfo(el.textContent);
              if (priceInfo.price) {
                price = priceInfo.price;
                currency = priceInfo.currency;
                break;
              }
            }
          }
        }
        
        if (!price) {
          const allSpans = container.querySelectorAll('span');
          for (const span of allSpans) {
            if (span.textContent && /[$€£¥]/.test(span.textContent)) {
              const priceInfo = extractPriceInfo(span.textContent);
              if (priceInfo.price) {
                price = priceInfo.price;
                currency = priceInfo.currency;
                break;
              }
            }
          }
        }
        
        if (!price && container.textContent) {
          const priceInfo = extractPriceInfo(container.textContent);
          price = priceInfo.price;
          currency = priceInfo.currency;
        }
      }
      
      return {
        url,
        title,
        price,
        currency,
        image,
        seller
      };
    }).filter(item => item.url); // Only keep items that have a URL
  }, workingSelector);
  
  console.log(`Found ${listings.length} product listings after filtering`);
  return listings;
};

const scrapeMultiplePages = async (page, startUrl, maxPages = 3, targetProductCount = 0) => {
  let allListings = [];
  let currentUrl = startUrl;
  let pageCounter = 1;
  
  while (pageCounter <= maxPages) {
    console.log(`Scraping page ${pageCounter}: ${currentUrl}`);
    
    try {
      await page.goto(currentUrl, {
        waitUntil: 'networkidle0', // Wait until there are no network connections for at least 500ms
        timeout: 90000 // Longer timeout (90 seconds)
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
  const chunkSize = 5; // Process in chunks to avoid memory issues
  
  // Split the listings into chunks
  for (let i = 0; i < listings.length; i += chunkSize) {
    const chunk = listings.slice(i, i + chunkSize);
    console.log(`Processing chunk ${Math.floor(i/chunkSize) + 1}/${Math.ceil(listings.length/chunkSize)} (${chunk.length} products)`);
    
    // Process each chunk in parallel
    const promises = chunk.map(async (listing, index) => {
      try {
        // Create a new page for each product to avoid issues with previous page state
        const page = await browser.newPage();
        
        // Set user agent and headers
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
        
        // Update basic info with more accurate data from the product page
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
  
  console.log('Starting Depop scraper...');
  console.log('Output will be saved to', outputDir);
  
  const browser = await puppeteer.launch({
    headless: true, // Keep false to see the browser and manually solve CAPTCHAs
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
    
    const targetProductCount = 75;
    
    const listingsPageUrl = 'https://www.depop.com/explore/under-usd100/?moduleOrigin=shop_by_price';
    
    console.log(`Starting with URL: ${listingsPageUrl}`);
    console.log(`Target product count: ${targetProductCount}`);
    
    // Scrape listings from one or more pages
    const maxPagesToScrape = 1; // Usually just need 1 page with continuous scrolling
    const listings = await scrapeMultiplePages(page, listingsPageUrl, maxPagesToScrape, targetProductCount);
    
    console.log(`Total products found: ${listings.length}`);
    
    if (listings.length > 0) {
      // Get detailed information for ALL products
      const detailedProducts = await getDetailedProductInfo(browser, listings);
      
      // Extract category from URL for the filename
      const categoryMatch = listingsPageUrl.match(/category\/([^\/]+)\/([^\/]+)/);
      const categoryString = categoryMatch ? `${categoryMatch[1]}-${categoryMatch[2]}` : 'category';
      
      // Save the detailed products data
      fs.writeFileSync(path.join(outputDir, `depop-${categoryString}.json`), JSON.stringify(detailedProducts, null, 2));
      console.log(`All ${detailedProducts.length} products with detailed info saved to output/depop-${categoryString}.json`);
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