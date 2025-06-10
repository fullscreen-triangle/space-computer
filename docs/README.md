# Space Computer Documentation Site

This directory contains the complete documentation website for Space Computer, built with Jekyll and designed for GitHub Pages.

## 🚀 **Quick Start**

### **View Online**
Visit the live documentation at: [https://yourusername.github.io/space-computer](https://yourusername.github.io/space-computer)

### **Run Locally**
```bash
# Navigate to docs directory
cd docs

# Install Ruby dependencies
bundle install

# Start Jekyll development server
bundle exec jekyll serve

# Open browser to http://localhost:4000
```

## 📋 **Site Structure**

### **Pages**
- **`index.md`** → Homepage with overview and navigation
- **`getting-started.md`** → Quick start guide for users
- **`platform.md`** → System architecture and design
- **`models.md`** → AI models and intelligence layer
- **`modules.md`** → Technical implementation details
- **`orchestration.md`** → Meta-orchestration system
- **`api-reference.md`** → Complete API documentation
- **`use-cases.md`** → Real-world applications and case studies
- **`deployment.md`** → Production deployment guide
- **`contributing.md`** → Developer contribution guide

### **Jekyll Configuration**
- **`_config.yml`** → Site configuration and navigation
- **`Gemfile`** → Ruby dependencies
- **`_layouts/default.html`** → Main page layout
- **`_includes/`** → Reusable HTML components
  - `head.html` → HTML head with SEO and meta tags
  - `header.html` → Site navigation header
  - `footer.html` → Site footer with links

## 🎨 **Features**

### **Enhanced UI/UX**
- **Responsive Design** → Mobile-friendly layouts
- **Custom Styling** → Modern, professional appearance
- **Navigation** → Intuitive menu with breadcrumbs
- **Table of Contents** → Auto-generated for long pages
- **Search-friendly** → SEO optimized with structured data

### **Technical Features**
- **Syntax Highlighting** → Code blocks with Prism.js
- **Math Rendering** → LaTeX equations with MathJax
- **Social Sharing** → Open Graph and Twitter Cards
- **Analytics Ready** → Google Analytics integration
- **Performance Optimized** → Fast loading with CDN assets

### **Documentation Features**
- **Comprehensive Coverage** → Every system component documented
- **Code Examples** → Interactive code samples
- **API Documentation** → Complete endpoint reference
- **Use Case Studies** → Real-world application examples
- **Contribution Guidelines** → Clear developer onboarding

## ⚙️ **Development**

### **Local Development Setup**
```bash
# Install Ruby (if not installed)
# On macOS:
brew install ruby

# On Ubuntu/Debian:
sudo apt-get install ruby-full

# Install Bundler
gem install bundler

# Install dependencies
bundle install

# Start development server with live reload
bundle exec jekyll serve --livereload

# Build static site
bundle exec jekyll build
```

### **Customization**

#### **Site Configuration**
Edit `_config.yml` to customize:
```yaml
title: "Your Site Title"
description: "Your site description"
url: "https://yourusername.github.io"
github_username: yourusername
twitter_username: yourtwitter
```

#### **Navigation Menu**
Update the `header_pages` list in `_config.yml`:
```yaml
header_pages:
  - index.md
  - getting-started.md
  - your-new-page.md
```

#### **Styling**
- **CSS** → Add custom styles to `_layouts/default.html`
- **Colors** → Update CSS variables for theme colors
- **Layout** → Modify `_layouts/` and `_includes/` files

### **Adding New Pages**
1. **Create Markdown File** → Add `your-page.md` to docs/
2. **Add Front Matter** → Include Jekyll configuration
3. **Update Navigation** → Add to `_config.yml` header_pages
4. **Link from Other Pages** → Add internal links

Example front matter:
```yaml
---
layout: default
title: "Your Page Title"
description: "Page description for SEO"
show_toc: true
show_navigation: true
---

# Your Page Content
```

## 🚀 **Deployment**

### **GitHub Pages (Automatic)**
1. **Push to Repository** → Changes automatically deploy
2. **Enable GitHub Pages** → Settings → Pages → Source: Deploy from branch
3. **Configure Domain** → Optional: Set custom domain
4. **SSL Certificate** → Automatically provided by GitHub

### **Manual Deployment**
```bash
# Build static site
bundle exec jekyll build

# Deploy _site/ directory to your web server
rsync -av _site/ user@server:/var/www/html/
```

### **Custom Domain Setup**
1. **Add CNAME File** → Create `docs/CNAME` with your domain
2. **Configure DNS** → Point domain to GitHub Pages IP
3. **Enable HTTPS** → GitHub automatically provides SSL

## 📊 **Analytics & SEO**

### **Google Analytics**
Add your tracking ID to `_config.yml`:
```yaml
google_analytics: UA-XXXXXXXX-X
```

### **SEO Optimization**
The site includes:
- **Meta Tags** → Title, description, keywords
- **Open Graph** → Social media sharing
- **Structured Data** → JSON-LD for search engines
- **Sitemap** → Auto-generated sitemap.xml
- **Robots.txt** → Search engine instructions

### **Performance Monitoring**
Monitor site performance with:
- **Google PageSpeed Insights**
- **GTmetrix**
- **WebPageTest**
- **Lighthouse** (built into Chrome DevTools)

## 🔧 **Troubleshooting**

### **Common Issues**

#### **Bundle Install Fails**
```bash
# Update RubyGems
gem update --system

# Install missing dependencies
bundle install --retry=3

# Clear cache if needed
bundle clean --force
```

#### **Jekyll Build Errors**
```bash
# Check for syntax errors
bundle exec jekyll doctor

# Build with verbose output
bundle exec jekyll build --verbose

# Clear cache and rebuild
bundle exec jekyll clean
bundle exec jekyll build
```

#### **GitHub Pages Build Fails**
- Check GitHub Pages build logs in repository Settings
- Ensure all files use supported Jekyll plugins
- Verify `_config.yml` syntax
- Test locally before pushing

### **Development Tips**
- **Live Reload** → Use `--livereload` flag for auto-refresh
- **Drafts** → Store drafts in `_drafts/` folder
- **Future Posts** → Use `--future` flag to show future-dated posts
- **Incremental Builds** → Use `--incremental` for faster builds

## 📚 **Resources**

### **Jekyll Documentation**
- **[Jekyll Official Docs](https://jekyllrb.com/docs/)**
- **[GitHub Pages Docs](https://docs.github.com/en/pages)**
- **[Liquid Template Language](https://shopify.github.io/liquid/)**

### **Themes & Plugins**
- **[Jekyll Themes](https://jekyllthemes.io/)**
- **[GitHub Pages Plugins](https://pages.github.com/versions/)**
- **[Jekyll Plugin Directory](https://planet.jekyllrb.com/)**

### **Tools & Resources**
- **[Markdown Guide](https://www.markdownguide.org/)**
- **[YAML Validator](https://codebeautify.org/yaml-validator)**
- **[Jekyll Now](https://github.com/barryclark/jekyll-now)** → Quick start template

## 🤝 **Contributing**

Help improve the documentation:

1. **Fix Typos** → Create PR with corrections
2. **Add Examples** → Enhance with code samples
3. **Update Content** → Keep information current
4. **Improve Design** → Suggest UI/UX improvements
5. **Add Features** → Implement new Jekyll functionality

See **[Contributing Guide](contributing.md)** for detailed instructions.

## 📄 **License**

This documentation is open source under the [MIT License](../LICENSE). Feel free to use, modify, and distribute as needed.

---

**Questions?** Open an issue or reach out on [Discord](https://discord.gg/spacecomputer)! 