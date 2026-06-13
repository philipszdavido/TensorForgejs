const fs = require('fs');
const path = require('path');
const {marked} = require('marked');

marked.setOptions({
    gfm: true,
    breaks: true
});

const SOURCE_DIR = path.join(__dirname, '');
const OUTPUT_DIR = path.join(__dirname, 'dist-docs');
const STYLE_PATH = path.join(__dirname, 'style.css');

if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, {recursive: true});
}

if (fs.existsSync(STYLE_PATH)) {
    fs.copyFileSync(STYLE_PATH, path.join(OUTPUT_DIR, 'style.css'));
}

const templateHtml = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>TensorForge Documentation</title>
    <link rel="stylesheet" href="style.css" />
  </head>
  <body>
    <div class="app-container">
      <nav class="sidebar">
        <div class="brand">TensorForge<span>JS</span></div>
        <div class="nav-group">
          <div class="nav-title">Getting Started</div>
          <ul class="nav-links">
            <li><a href="index.html">Overview</a></li>
            <li><a href="guide.html">Full Guide</a></li>
            <li><a href="examples.html">Examples</a></li>
          </ul>
        </div>
        <div class="nav-group">
          <div class="nav-title">Architecture</div>
          <ul class="nav-links">
            <li><a href="architecture.html">Internal Design</a></li>
            <li><a href="core.html">Core Structures</a></li>
            <li><a href="math.html">Math Operations</a></li>
            <li><a href="models.html">Supported Models</a></li>
          </ul>
        </div>
        <div class="nav-group">
          <div class="nav-title">Reference</div>
          <ul class="nav-links">
            <li><a href="api.html">API Reference</a></li>
            <li><a href="usage.html">Minimal Usage</a></li>
          </ul>
        </div>
      </nav>
      <main class="main-content">
        </main>
    </div>
    <script>
      const currentFile = window.location.pathname.split('/').pop() || 'index.html';
      document.querySelectorAll('.nav-links a').forEach(link => {
        if (link.getAttribute('href') === currentFile) {
          link.classList.add('active');
        }
      });
    </script>
  </body>
</html>`;

fs.readdir(SOURCE_DIR, (err, files) => {
    if (err) {
        process.exit(1);
    }

    files.forEach(file => {
        if (path.extname(file) === '.md') {
            const filePath = path.join(SOURCE_DIR, file);
            const mdContent = fs.readFileSync(filePath, 'utf-8');
            const htmlBody = marked.parse(mdContent);
            const completePage = htmlBody//templateHtml.replace('', htmlBody);
            const outputFilename = file.replace('.md', '.html');
            fs.writeFileSync(path.join(OUTPUT_DIR, outputFilename), completePage, 'utf-8');
        }
    });
});
