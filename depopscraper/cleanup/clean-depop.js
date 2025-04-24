const fs = require('fs');
const path = require('path');

// Load the original file
const inputPath = path.join(__dirname, 'output', 'depop-mens-bottoms.json');
const outputPath = path.join(__dirname, 'output', 'depop-mens-bottoms-cleaned.json');

// Read and parse the JSON
const rawData = fs.readFileSync(inputPath, 'utf-8');
const products = JSON.parse(rawData);

// Remove structuredData from each product
const cleanedProducts = products.map(({ structuredData, ...rest }) => rest);

// Save to new file
fs.writeFileSync(outputPath, JSON.stringify(cleanedProducts, null, 2));

console.log(`Cleaned file saved to ${outputPath}`);
