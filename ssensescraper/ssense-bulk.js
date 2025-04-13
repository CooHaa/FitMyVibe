const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const CONFIG = {
  minDelay: 6000,
  maxDelay: 10000,
  breakInterval: 10,
  breakDuration: 30000,
  captchaTimeout: 60000,
  outputFile: 'ssense-products.json',
  progressFile: 'scraping-progress.json',
  userAgents: [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.58'
  ]
};

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
const randomBetween = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
const getRandomUserAgent = () => CONFIG.userAgents[Math.floor(Math.random() * CONFIG.userAgents.length)];

const loadLinksFile = (filePath) => {
  try {
    const data = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Error reading links file:', error);
    return null;
  }
};

const loadProgress = () => {
  try {
    if (fs.existsSync(CONFIG.progressFile)) {
      const data = fs.readFileSync(CONFIG.progressFile, 'utf8');
      return JSON.parse(data);
    }
  } catch (error) {
    console.error('Error reading progress:', error);
  }
  return { processedUrls: [], currentIndex: 0 };
};

const saveProgress = (progress) => {
  try {
    fs.writeFileSync(CONFIG.progressFile, JSON.stringify(progress, null, 2));
  } catch (error) {
    console.error('Error saving progress:', error);
  }
};

const loadExistingProducts = () => {
  try {
    if (fs.existsSync(CONFIG.outputFile)) {
      const data = fs.readFileSync(CONFIG.outputFile, 'utf8');
      return JSON.parse(data);
    }
  } catch (error) {
    console.error('Error reading products:', error);
  }
  return [];
};

const saveProducts = (products) => {
  try {
    fs.writeFileSync(CONFIG.outputFile, JSON.stringify(products, null, 2));
    console.log(`Saved ${products.length} products`);
  } catch (error) {
    console.error('Error saving products:', error);
  }
};

async function scrapeProduct(browser, url, retryCount = 0) {
  if (retryCount > 3) {
    console.error(`Failed after multiple retries: ${url}`);
    return null;
  }
  
  const page = await browser.newPage();
  const userAgent = getRandomUserAgent();
  await page.setUserAgent(userAgent);
  
  await page.setExtraHTTPHeaders({
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'max-age=0'
  });
  
  await page.setViewport({ 
    width: randomBetween(1250, 1300), 
    height: randomBetween(780, 820) 
  });
  
  await page.evaluateOnNewDocument(() => {
    Object.defineProperty(navigator, 'webdriver', { get: () => false });
    
    Object.defineProperty(navigator, 'plugins', {
      get: () => {
        return [1, 2, 3, 4, 5].map(i => ({
          0: {
            type: "application/x-google-chrome-pdf",
            suffixes: "pdf",
            description: "Portable Document Format"
          },
          name: `Chrome PDF Plugin ${i}`,
          description: `Portable Document Format ${i}`,
          filename: `internal-pdf-viewer-${i}`,
          length: 1
        }));
      }
    });
    
    Object.defineProperty(navigator, 'languages', {
      get: () => ['en-US', 'en', 'es']
    });
  });
  
  try {
    console.log(`Navigating: ${url}`);
    
    await page.goto(url, {
      waitUntil: 'networkidle2',
      timeout: 60000
    });
    
    const screenshotPath = `screenshot-${Date.now()}.png`;
    await page.screenshot({ path: screenshotPath });
    console.log(`Screenshot: ${screenshotPath}`);
    
    const isCaptcha = await page.evaluate(() => {
      return document.body.innerText.includes('Press & Hold') || 
             document.body.innerText.includes('confirm you are a human');
    });
    
    if (isCaptcha) {
      console.log('CAPTCHA detected! Manual solving needed.');
      await delay(CONFIG.captchaTimeout);
      await page.screenshot({ path: `after-captcha-${Date.now()}.png` });
      
      const stillCaptcha = await page.evaluate(() => {
        return document.body.innerText.includes('Press & Hold') || 
               document.body.innerText.includes('confirm you are a human');
      });
      
      if (stillCaptcha) {
        console.log('CAPTCHA still present. Retrying...');
        await page.close();
        await delay(randomBetween(15000, 25000));
        return scrapeProduct(browser, url, retryCount + 1);
      }
    }
    
    const isAccessDenied = await page.evaluate(() => {
      return document.body.innerText.includes('Access Denied') || 
             document.body.innerText.includes('You don\'t have permission');
    });
    
    if (isAccessDenied) {
      console.log('Access Denied. Taking break...');
      await page.close();
      await delay(randomBetween(30000, 60000));
      return scrapeProduct(browser, url, retryCount + 1);
    }
    
    const jsonLd = await page.evaluate(() => {
      const script = document.querySelector('script[type="application/ld+json"]');
      return script ? script.textContent : null;
    });
    
    let productData = {
      url,
      scrapedAt: new Date().toISOString(),
      formattedData: {}
    };
    
    if (jsonLd) {
      console.log('JSON-LD found');
      try {
        productData.structuredData = JSON.parse(jsonLd);
        
        productData.formattedData = {
          productName: productData.structuredData.name,
          brand: productData.structuredData.brand?.name,
          price: productData.structuredData.offers?.price,
          currency: productData.structuredData.offers?.priceCurrency,
          description: productData.structuredData.description,
          image: productData.structuredData.image,
          availability: productData.structuredData.offers?.availability
        };
      } catch (e) {
        console.error('Error parsing JSON-LD:', e.message);
      }
    } else {
      console.log('Falling back to HTML extraction');
      
      productData.formattedData = await page.evaluate(() => {
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
          document.querySelector('img[srcset*="ssense"]');
        
        const image = imageElement ? 
          (imageElement.src || 
           (imageElement.srcset && imageElement.srcset.split(',').pop().trim().split(' ')[0])) : 
          null;
        
        return {
          productName,
          brand,
          price,
          currency,
          description,
          image
        };
      });
    }
    
    if (!productData.formattedData.productName) {
      console.log('No product data found. Retrying...');
      await page.close();
      return scrapeProduct(browser, url, retryCount + 1);
    }
    
    console.log('Data extracted:');
    console.log(`Name: ${productData.formattedData.productName}`);
    console.log(`Brand: ${productData.formattedData.brand}`);
    console.log(`Price: ${productData.formattedData.price} ${productData.formattedData.currency}`);
    
    await page.close();
    return productData;
    
  } catch (error) {
    console.error(`Error: ${url}:`, error);
    await page.close();
    
    const backoffTime = Math.pow(2, retryCount) * 10000;
    console.log(`Retry in ${backoffTime/1000}s...`);
    await delay(backoffTime);
    
    return scrapeProduct(browser, url, retryCount + 1);
  }
}

async function processUrls(linksFilePath) {
  const categoriesWithLinks = loadLinksFile(linksFilePath);
  if (!categoriesWithLinks) {
    console.error('Failed to load links. Exiting...');
    return;
  }
  
  let allLinks = [];
  categoriesWithLinks.forEach(category => {
    if (category.links && Array.isArray(category.links)) {
      allLinks = [...allLinks, ...category.links];
    }
  });
  
  console.log(`Found ${allLinks.length} links`);
  
  const progress = loadProgress();
  let { processedUrls, currentIndex } = progress;
  
  console.log(`Resuming from ${currentIndex}, processed ${processedUrls.length}`);
  
  let products = loadExistingProducts();
  console.log(`Loaded ${products.length} products`);
  
  const browser = await puppeteer.launch({
    headless: false,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-features=IsolateOrigins,site-per-process',
      '--disable-blink-features=AutomationControlled',
      '--window-size=1280,800'
    ]
  });
  
  console.log('Browser launched');
  
  try {
    for (let i = currentIndex; i < allLinks.length; i++) {
      const url = allLinks[i];
      
      if (processedUrls.includes(url)) {
        console.log(`Skipping: ${url}`);
        continue;
      }
      
      console.log(`Processing ${i+1}/${allLinks.length}: ${url}`);
      
      const productData = await scrapeProduct(browser, url);
      
      if (productData) {
        products.push(productData);
        processedUrls.push(url);
        currentIndex = i + 1;
        saveProgress({ processedUrls, currentIndex });
        saveProducts(products);
      }
      
      if ((i + 1) % CONFIG.breakInterval === 0 && i < allLinks.length - 1) {
        const breakTime = randomBetween(CONFIG.breakDuration * 0.8, CONFIG.breakDuration * 1.2);
        console.log(`Break for ${Math.round(breakTime/1000)}s...`);
        await delay(breakTime);
      } else {
        const waitTime = randomBetween(CONFIG.minDelay, CONFIG.maxDelay);
        console.log(`Wait ${Math.round(waitTime/1000)}s...`);
        await delay(waitTime);
      }
    }
    
    console.log('All links processed!');
    console.log(`Total products: ${products.length}`);
    
  } catch (error) {
    console.error('Processing error:', error);
  } finally {
    await browser.close();
    console.log('Browser closed');
  }
}

const linksFilePath = process.argv[2] || 'ssense-product-links.json';
processUrls(linksFilePath).catch(console.error);