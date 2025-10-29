const pptxgen = require('pptxgenjs');
const html2pptx = require('C:\\Users\\Owner\\.claude\\skills\\pptx\\scripts\\html2pptx.js');
const path = require('path');

async function createPresentation() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';  // Matches 720pt × 405pt HTML dimensions
  pptx.author = 'Team Audio Source Separation';
  pptx.title = 'Week 5 Progress Report: 5 Mini-Labs';
  pptx.subject = 'Audio Separation Pipeline';

  const slidesDir = 'C:\\users\\owner\\working-session\\output\\slides';

  // Process all slides
  const slides = [
    'slide01-title.html',
    'slide02-overview.html',
    'slide03-pipeline.html',
    'slide04-lab1.html',
    'slide05-lab2.html',
    'slide06-lab3.html',
    'slide07-lab4.html',
    'slide08-lab5.html',
    'slide09-team.html',
    'slide10-heodon.html',
    'slide11-nextsteps.html',
    'slide12-summary.html',
    'slide13-questions.html'
  ];

  console.log('Creating presentation with', slides.length, 'slides...\n');

  for (let i = 0; i < slides.length; i++) {
    const slideFile = path.join(slidesDir, slides[i]);
    console.log(`Processing slide ${i + 1}/${slides.length}: ${slides[i]}`);

    try {
      const { slide, placeholders } = await html2pptx(slideFile, pptx);
      console.log(`  ✓ Slide ${i + 1} created successfully`);

      if (placeholders.length > 0) {
        console.log(`  ℹ  Found ${placeholders.length} placeholder(s)`);
      }
    } catch (error) {
      console.error(`  ✗ Error on slide ${i + 1}:`, error.message);
      throw error;
    }
  }

  // Save presentation
  const outputFile = 'C:\\users\\owner\\working-session\\output\\Week5-Progress-Report-Updated.pptx';
  await pptx.writeFile({ fileName: outputFile });

  console.log('\n✓ Presentation created successfully!');
  console.log('  File:', outputFile);
  console.log('  Slides:', slides.length);
}

createPresentation().catch(error => {
  console.error('\n✗ Error creating presentation:', error.message);
  console.error(error.stack);
  process.exit(1);
});
