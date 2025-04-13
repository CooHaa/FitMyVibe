const puppeteer = require('puppeteer');
const fs = require('fs');

// Helper function for delays (compatible with all Puppeteer versions)
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  // Launch browser with non-headless mode to see and solve CAPTCHAs manually
  const browser = await puppeteer.launch({
    headless: false, // Change to false to see the browser and manually solve CAPTCHAs
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-features=IsolateOrigins,site-per-process'
    ]
  });
  
  console.log('Browser launched');
  
  try {
    const page = await browser.newPage();
    
    await page.setViewport({ width: 1280, height: 800 });
    
    // Use a more realistic user agent
    await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36');
    
    // Add extra headers to appear more like a real browser
    await page.setExtraHTTPHeaders({
      'Accept-Language': 'en-US,en;q=0.9',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
      'Accept-Encoding': 'gzip, deflate, br',
      'Connection': 'keep-alive',
      'Upgrade-Insecure-Requests': '1'
    });
    
    // Human-like behavior: Add random initial delay
    const initialDelay = Math.floor(Math.random() * 3000) + 2000;
    console.log(`Waiting ${initialDelay}ms before navigating...`);
    await delay(initialDelay);
    
    console.log('Navigating to SSENSE product page...');
    
    await page.goto('https://www.ssense.com/en-us/women/product/jimmy-choo/black-azia-95-heeled-sandals/16909281', {
      waitUntil: 'networkidle2', 
      timeout: 60000 
    });
    
    console.log('Page loaded');
    
    // Save screenshot for debugging
    await page.screenshot({ path: 'ssense-page.png' });
    
    // Check if we hit a CAPTCHA
    const isCaptcha = await page.evaluate(() => {
      return document.body.innerText.includes('Press & Hold') || 
             document.body.innerText.includes('confirm you are a human');
    });
    
    if (isCaptcha) {
      console.log('⚠️ CAPTCHA detected! Please solve it manually in the browser window.');
      console.log('Waiting 30 seconds for manual CAPTCHA solving...');
      
      // Wait longer to allow manual solving - adjust as needed
      await delay(30000);
      
      // Take another screenshot after CAPTCHA is solved
      await page.screenshot({ path: 'after-captcha.png' });
    }
    
    // Add another small random delay
    await delay(Math.floor(Math.random() * 2000) + 1000);
    
    const jsonLd = await page.evaluate(() => {
      const script = document.querySelector('script[type="application/ld+json"]');
      return script ? script.textContent : null;
    });
    
    let productData = {
      url: 'https://www.ssense.com/en-us/women/product/jimmy-choo/black-azia-95-heeled-sandals/16909281',
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
          availability: productData.structuredData.offers?.availability
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
        
        // Try various possible selectors for product info
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
        
        // Extract price and currency
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
        
        // Get product image
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
    
    fs.writeFileSync('ssense-product.json', JSON.stringify(productData, null, 2));
    console.log('Product data saved to ssense-product.json');
    
    console.log('\nProduct Information:');
    console.log('Name:', productData.formattedData?.productName || 'Not found');
    console.log('Brand:', productData.formattedData?.brand || 'Not found');
    console.log('Price:', productData.formattedData?.price || 'Not found', productData.formattedData?.currency || '');
    console.log('Description:', productData.formattedData?.description || 'Not found');
    
  } catch (error) {
    console.error('Error during scraping:', error);
  } finally {
    await browser.close();
    console.log('Browser closed');
  }
})();