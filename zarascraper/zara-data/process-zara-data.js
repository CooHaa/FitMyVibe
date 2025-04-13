//parses raw json file from all product listings, and extracts unique products. 

const fs = require('fs');

function processZaraData(inputFile, outputFile) {
  try {
    const rawData = fs.readFileSync(inputFile, 'utf8');
    const data = JSON.parse(rawData);
    
    let productEntries = [];
    
    if (Array.isArray(data)) {
      data.forEach(item => {
        if (item.structuredData && Array.isArray(item.structuredData)) {
          productEntries.push(...item.structuredData);
        }
      });
    } else if (data.men || data.women) {
      Object.keys(data).forEach(category => {
        if (Array.isArray(data[category])) {
          data[category].forEach(item => {
            if (item.structuredData && Array.isArray(item.structuredData)) {
              productEntries.push(...item.structuredData);
            }
          });
        }
      });
    }
    
    console.log(`Found ${productEntries.length} total product entries`);
    
    const productMap = new Map();
    
    productEntries.forEach(product => {
      if (!product.name || !product.color) return;
      
      const key = `${product.name}__${product.color}`;
      
      if (!productMap.has(key)) {
        productMap.set(key, product);
      }
    });
    
    console.log(`Consolidated to ${productMap.size} unique products`);
    
    const uniqueProducts = Array.from(productMap.values());
    
    fs.writeFileSync(outputFile, JSON.stringify(uniqueProducts, null, 2));
    console.log(`Processed data saved to ${outputFile}`);
    
    if (uniqueProducts.length > 0) {
      console.log('\nSample product:');
      console.log(JSON.stringify(uniqueProducts[0], null, 2));
    }
    
    return {
      totalEntries: productEntries.length,
      uniqueProducts: uniqueProducts.length
    };
  } catch (error) {
    console.error('Error processing data:', error.message);
    throw error;
  }
}

function processZaraDataWithSizes(inputFile, outputFile) {
  try {
    const rawData = fs.readFileSync(inputFile, 'utf8');
    const data = JSON.parse(rawData);
    
    let productEntries = [];
    
    if (Array.isArray(data)) {
      data.forEach(item => {
        if (item.structuredData && Array.isArray(item.structuredData)) {
          productEntries.push(...item.structuredData);
        }
      });
    } else if (data.men || data.women) {
      Object.keys(data).forEach(category => {
        if (Array.isArray(data[category])) {
          data[category].forEach(item => {
            if (item.structuredData && Array.isArray(item.structuredData)) {
              productEntries.push(...item.structuredData);
            }
          });
        }
      });
    }
    
    console.log(`Found ${productEntries.length} total product entries`);
    
    const productGroups = {};
    
    productEntries.forEach(product => {
      if (!product.name || !product.color) return;
      
      const key = `${product.name}__${product.color}`;
      
      if (!productGroups[key]) {
        productGroups[key] = {
          ...product,
          allSizes: [],
          availableSizes: []
        };
        
        delete productGroups[key].size;
      }
      
      if (product.size) {
        productGroups[key].allSizes.push(product.size);
        
        if (product.offers && product.offers.availability === "https://schema.org/InStock") {
          productGroups[key].availableSizes.push(product.size);
        }
      }
    });
    
    const uniqueProducts = Object.values(productGroups);
    
    console.log(`Consolidated to ${uniqueProducts.length} unique products`);
    
    fs.writeFileSync(outputFile, JSON.stringify(uniqueProducts, null, 2));
    console.log(`Processed data saved to ${outputFile}`);
    
    if (uniqueProducts.length > 0) {
      console.log('\nSample product:');
      console.log(JSON.stringify(uniqueProducts[0], null, 2));
    }
    
    return {
      totalEntries: productEntries.length,
      uniqueProducts: uniqueProducts.length
    };
  } catch (error) {
    console.error('Error processing data:', error.message);
    throw error;
  }
}

function main() {
  const args = process.argv.slice(2);
  
  let inputFile = 'all-products.json';
  let outputFile = 'unique-products.json';
  let includeSizes = false;
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--input' && i + 1 < args.length) {
      inputFile = args[i + 1];
      i++;
    } else if (args[i] === '--output' && i + 1 < args.length) {
      outputFile = args[i + 1];
      i++;
    } else if (args[i] === '--include-sizes') {
      includeSizes = true;
    }
  }
  
  console.log(`Processing Zara data...`);
  console.log(`Input: ${inputFile}`);
  console.log(`Output: ${outputFile}`);
  console.log(`Include sizes: ${includeSizes ? 'Yes' : 'No'}`);
  
  if (includeSizes) {
    processZaraDataWithSizes(inputFile, outputFile);
  } else {
    processZaraData(inputFile, outputFile);
  }
}

main();