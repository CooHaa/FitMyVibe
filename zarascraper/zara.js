// extracts one product listing from single website. 
const puppeteer = require('puppeteer');
const fs = require('fs');

(async () => {
  const browser = await puppeteer.launch({
    headless: "new",
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  console.log('Browser launched');
  
  try {
    const page = await browser.newPage();
    
    await page.setViewport({ width: 1280, height: 800 });
    
    await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36');
    
    console.log('Navigating to Zara product page...');
    
    await page.goto('https://www.zara.com/us/en/floral-print-maxi-skirt-p05039130.html?v1=437430532&v2=2420454', {
      waitUntil: 'networkidle2', 
      timeout: 60000 
    });
    
    console.log('Page loaded');
    
    await page.waitForSelector('.product-detail-info', { timeout: 30000 }).catch(() => console.log('Selector timeout - continuing anyway'));
    
    await page.screenshot({ path: 'zara-page.png' });
    console.log('Screenshot saved');
    
    const jsonLd = await page.evaluate(() => {
      const script = document.querySelector('script[type="application/ld+json"]');
      return script ? script.textContent : null;
    });
    
    let productData = {};
    
    if (jsonLd) {
      console.log('JSON-LD data found');
      try {
        productData.structuredData = JSON.parse(jsonLd);
      } catch (e) {
        console.error('Error parsing JSON-LD:', e.message);
      }
    } else {
      console.log('No JSON-LD data found');
    }
    
    // Extract product details from the page
    productData.extractedData = await page.evaluate(() => {
      return {
        productName: document.querySelector('.product-detail-info__header-name')?.innerText.trim() || 'Not found',
        price: document.querySelector('.money-amount__main')?.innerText.trim() || 'Not found',
        color: document.querySelector('.product-color-extended-name')?.innerText.split('|')[0].trim() || 'Not found',
        description: document.querySelector('.expandable-text__inner-content p')?.innerText.trim() || 'Not found',
        productCode: document.querySelector('.product-color-extended-name__copy-action')?.innerText.trim() || 'Not found',
        sizes: Array.from(document.querySelectorAll('.product-detail-size-info__name')).map(el => el.innerText.trim()),
        images: Array.from(document.querySelectorAll('picture.media-image source')).map(el => {
          const srcSet = el.getAttribute('srcset');
          if (srcSet) {
            const sources = srcSet.split(', ');
            const lastSource = sources[sources.length - 1];
            return lastSource.split(' ')[0]; // Remove width descriptor
          }
          return null;
        }).filter(Boolean)
      };
    });
    
    const html = await page.content();
    fs.writeFileSync('zara-full.html', html);
    console.log('Full HTML saved for debugging');
    
    productData.url = 'https://www.zara.com/us/en/floral-print-maxi-skirt-p05039130.html?v1=437430532&v2=2420454';
    productData.scrapedAt = new Date().toISOString();
    
    fs.writeFileSync('zara-product.json', JSON.stringify(productData, null, 2));
    console.log('Product data saved to zara-product.json');
    
    console.log('\nProduct Information:');
    console.log('Name:', productData.extractedData.productName);
    console.log('Price:', productData.extractedData.price);
    console.log('Color:', productData.extractedData.color);
    console.log('Description:', productData.extractedData.description);
    console.log('Product Code:', productData.extractedData.productCode);
    
  } catch (error) {
    console.error('Error during scraping:', error);
  } finally {
    // Close the browser
    await browser.close();
    console.log('Browser closed');
  }
})();