const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

const folder = path.join(__dirname, 'assets');
fs.readdirSync(folder).forEach(file => {
  if (file.endsWith('.png')) {
    sharp(path.join(folder, file))
      .png({ compressionLevel: 9, adaptiveFiltering: true })
      .toFile(path.join(folder, file), (err) => {
        if (err) console.error(err);
        else console.log(`Processed ${file}`);
      });
  }
});
