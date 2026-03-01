"""
J.A.S.O.N. Dropshipping Automation Engine
Automated store generation with SEO optimization, landing page builder,
and supplier integration using live crawled data.
"""

import os
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


class DropshippingEngine:
    """
    Autonomous Dropshipping Engine for J.A.R.V.I.S.
    Generates optimized store pages, product listings, and SEO content
    from live supplier data.
    """

    # High-conversion page templates
    LANDING_PAGE_SECTIONS = [
        'hero',
        'social_proof',
        'product_showcase',
        'features_benefits',
        'testimonials',
        'urgency_timer',
        'faq',
        'cta_footer',
    ]

    # SEO keyword templates
    SEO_TEMPLATES = {
        'title': '{product} - {benefit} | {brand} Official Store',
        'description': 'Shop {product} with {discount}% OFF! {benefit}. Free shipping on orders over $49. ⭐ {rating}/5 stars from {reviews}+ reviews.',
        'h1': 'The #{rank} Best-Selling {product} of {year}',
    }

    def __init__(self, config: dict = None, spider_crawler=None):
        self.config = config or {}
        self.spider = spider_crawler  # SpiderCrawler instance for live data

        # Store configuration
        self.store_config = {
            'brand_name': 'VelocityStore',
            'brand_tagline': 'Premium Products, Unbeatable Prices',
            'color_primary': '#6366f1',
            'color_secondary': '#8b5cf6',
            'color_accent': '#f59e0b',
            'free_shipping_threshold': 49,
            'currency': 'USD',
            'currency_symbol': '$',
        }

        # Output directory
        self.output_dir = Path(os.path.dirname(__file__)) / '..' / '..' / 'output' / 'dropship_store'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Product catalog
        self.product_catalog = []
        self.landing_pages = []

        # Stats
        self.stats = {
            'products_sourced': 0,
            'pages_generated': 0,
            'seo_score_avg': 0,
        }

        logger.info("Dropshipping Engine initialized")

    def source_products(self, niche: str, max_products: int = 20) -> List[Dict]:
        """
        Source products from suppliers using the Spider Crawler.
        Applies margin calculation and competition analysis.
        """
        if not self.spider:
            logger.warning("No Spider Crawler available, using demo product data")
            return self._generate_demo_products(niche, max_products)

        logger.info(f"Sourcing products for niche: {niche}")

        # Search suppliers
        supplier_data = self.spider.search_suppliers(niche)

        sourced_products = []

        for supplier_name, supplier_result in supplier_data.get('suppliers', {}).items():
            for product in supplier_result.get('products', [])[:max_products]:
                # Calculate dropshipping margins
                cost = product.get('price', 0)
                if cost <= 0:
                    continue

                optimized_product = self._optimize_product_listing(product, supplier_name, cost)
                sourced_products.append(optimized_product)

        # Sort by estimated margin
        sourced_products.sort(key=lambda p: p.get('estimated_margin_pct', 0), reverse=True)

        self.product_catalog = sourced_products[:max_products]
        self.stats['products_sourced'] = len(self.product_catalog)

        logger.info(f"Sourced {len(self.product_catalog)} products for niche: {niche}")
        return self.product_catalog

    def _optimize_product_listing(self, product: Dict, supplier: str, cost: float) -> Dict:
        """Optimize a single product listing with pricing and SEO"""
        # Pricing strategy: 2.5x-3.5x markup for typical dropshipping
        markup = 2.8
        if cost < 5:
            markup = 3.5  # Higher markup for cheap items
        elif cost > 50:
            markup = 2.2  # Lower markup for expensive items

        selling_price = round(cost * markup, 2)
        # Psychological pricing
        selling_price = math.floor(selling_price) + 0.99 if selling_price > 10 else round(selling_price, 2)

        margin = selling_price - cost
        margin_pct = (margin / selling_price) * 100

        # Generate SEO-optimized title
        original_name = product.get('name', 'Premium Product')
        seo_title = self._generate_seo_title(original_name)

        return {
            'original_name': original_name,
            'seo_title': seo_title,
            'supplier': supplier,
            'cost': cost,
            'selling_price': selling_price,
            'compare_at_price': round(selling_price * 1.4, 2),  # "Was" price
            'estimated_margin': round(margin, 2),
            'estimated_margin_pct': round(margin_pct, 1),
            'image_url': product.get('image_url', ''),
            'product_url': product.get('product_url', ''),
            'rating': product.get('rating', 4.5),
            'orders': product.get('orders_text', ''),
            'description': product.get('description', ''),
            'seo_meta_description': self._generate_meta_description(original_name, selling_price),
        }

    def _generate_seo_title(self, product_name: str) -> str:
        """Generate SEO-optimized product title"""
        # Remove common low-value words
        noise_words = ['free shipping', 'hot sale', 'new arrival', 'wholesale', 'dropshipping']
        title = product_name
        for word in noise_words:
            title = re.sub(word, '', title, flags=re.IGNORECASE)

        # Capitalize properly
        title = ' '.join(word.capitalize() for word in title.split())
        title = title.strip(' -,.')

        # Add power words
        if len(title) < 50:
            title += ' - Premium Quality'

        return title[:65]  # Google title limit

    def _generate_meta_description(self, name: str, price: float) -> str:
        """Generate SEO meta description"""
        return (
            f"Shop {name} for just ${price}! ✅ Premium quality, "
            f"free shipping on orders $49+. ⭐ Rated 4.8/5 by 1000+ happy customers. "
            f"30-day money-back guarantee. Order now!"
        )[:160]

    def _generate_demo_products(self, niche: str, count: int) -> List[Dict]:
        """Generate demonstration product data"""
        import random

        templates = {
            'tech': [
                ('Wireless Bluetooth Earbuds', 8.50),
                ('LED Ring Light with Tripod', 12.00),
                ('Portable Phone Charger 20000mAh', 15.00),
                ('Smart Watch Fitness Tracker', 18.00),
                ('Wireless Car Phone Mount', 7.50),
            ],
            'home': [
                ('Self-Watering Plant Pot', 5.00),
                ('LED Strip Lights 10m', 6.50),
                ('Bamboo Kitchen Utensil Set', 8.00),
                ('Smart Wi-Fi Plug', 7.00),
                ('Essential Oil Diffuser', 11.00),
            ],
            'fashion': [
                ('Oversized Sunglasses UV400', 3.50),
                ('Minimalist Watch Stainless Steel', 12.00),
                ('Canvas Backpack Vintage', 15.00),
                ('Silk Scarf Premium', 8.00),
                ('Titanium Ring Set', 5.00),
            ],
        }

        # Pick closest niche
        niche_key = 'tech'
        for key in templates:
            if key in niche.lower():
                niche_key = key
                break

        products = []
        for name, cost in templates.get(niche_key, templates['tech'])[:count]:
            products.append(self._optimize_product_listing(
                {'name': name, 'rating': round(random.uniform(4.2, 4.9), 1)},
                'demo_supplier',
                cost,
            ))

        self.product_catalog = products
        return products

    # ─── LANDING PAGE GENERATOR ──────────────────────────────────

    def generate_landing_page(
        self,
        product: Dict = None,
        page_type: str = 'product',
        niche: str = 'premium products',
    ) -> Dict[str, Any]:
        """
        Generate a high-conversion landing page as HTML.
        Based on conversion optimization best practices.
        """
        if product is None and self.product_catalog:
            product = self.product_catalog[0]
        elif product is None:
            product = {
                'seo_title': 'Premium Product',
                'selling_price': 29.99,
                'compare_at_price': 49.99,
                'rating': 4.8,
                'description': 'An amazing product.',
            }

        if page_type == 'product':
            html = self._build_product_page(product)
        elif page_type == 'collection':
            html = self._build_collection_page(niche)
        elif page_type == 'homepage':
            html = self._build_homepage(niche)
        else:
            html = self._build_product_page(product)

        # Save to file
        filename = f"landing_{page_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write(html)

        self.stats['pages_generated'] += 1

        result = {
            'filepath': str(filepath),
            'html_length': len(html),
            'page_type': page_type,
            'product': product.get('seo_title', 'N/A'),
            'seo_score': self._calculate_page_seo_score(html),
        }

        self.landing_pages.append(result)
        logger.info(f"Landing page generated: {filepath}")

        return result

    def _build_product_page(self, product: Dict) -> str:
        """Build a high-conversion product landing page"""
        brand = self.store_config['brand_name']
        primary = self.store_config['color_primary']
        secondary = self.store_config['color_secondary']
        accent = self.store_config['color_accent']

        title = product.get('seo_title', 'Premium Product')
        price = product.get('selling_price', 29.99)
        compare_price = product.get('compare_at_price', price * 1.4)
        rating = product.get('rating', 4.8)
        description = product.get('description', 'An exceptional product designed for modern living.')
        meta_desc = product.get('seo_meta_description', self._generate_meta_description(title, price))
        discount_pct = int(((compare_price - price) / compare_price) * 100)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="{meta_desc}">
    <title>{title} | {brand}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: #0a0a0a;
            color: #e5e5e5;
            overflow-x: hidden;
        }}
        
        /* Announcement Bar */
        .announcement {{
            background: linear-gradient(90deg, {primary}, {secondary});
            text-align: center;
            padding: 10px;
            font-size: 14px;
            font-weight: 600;
            color: white;
            animation: pulse 2s ease-in-out infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.85; }}
        }}
        
        /* Navigation */
        nav {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 40px;
            background: rgba(10, 10, 10, 0.95);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(255,255,255,0.06);
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .logo {{
            font-size: 24px;
            font-weight: 800;
            background: linear-gradient(135deg, {primary}, {accent});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .nav-links {{ display: flex; gap: 28px; list-style: none; }}
        .nav-links a {{
            color: #a3a3a3;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            transition: color 0.2s;
        }}
        .nav-links a:hover {{ color: white; }}
        
        /* Hero Section */
        .hero {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 60px;
            max-width: 1200px;
            margin: 0 auto;
            padding: 80px 40px;
            align-items: center;
        }}
        .hero-image {{
            background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1));
            border-radius: 24px;
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(99,102,241,0.2);
            font-size: 120px;
            position: relative;
            overflow: hidden;
        }}
        .hero-image::before {{
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 30% 30%, rgba(99,102,241,0.2), transparent 70%);
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(245,158,11,0.15);
            color: {accent};
            padding: 6px 14px;
            border-radius: 50px;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 16px;
            border: 1px solid rgba(245,158,11,0.3);
        }}
        .hero h1 {{
            font-size: 42px;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 16px;
            background: linear-gradient(135deg, #fff, #a3a3a3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .hero-desc {{
            font-size: 16px;
            color: #737373;
            line-height: 1.6;
            margin-bottom: 24px;
        }}
        .price-block {{
            display: flex;
            align-items: baseline;
            gap: 16px;
            margin-bottom: 24px;
        }}
        .price-current {{
            font-size: 48px;
            font-weight: 800;
            color: white;
        }}
        .price-compare {{
            font-size: 22px;
            color: #525252;
            text-decoration: line-through;
        }}
        .price-save {{
            background: rgba(34,197,94,0.15);
            color: #22c55e;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 700;
        }}
        .rating {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 24px;
            color: {accent};
            font-size: 14px;
        }}
        .cta-btn {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background: linear-gradient(135deg, {primary}, {secondary});
            color: white;
            padding: 18px 42px;
            border-radius: 14px;
            font-size: 18px;
            font-weight: 700;
            border: none;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 8px 32px rgba(99,102,241,0.35);
        }}
        .cta-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(99,102,241,0.5);
        }}
        .trust-badges {{
            display: flex;
            gap: 24px;
            margin-top: 24px;
        }}
        .trust-badge {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            color: #737373;
        }}
        
        /* Social Proof Bar */
        .social-proof {{
            background: rgba(255,255,255,0.03);
            border-top: 1px solid rgba(255,255,255,0.06);
            border-bottom: 1px solid rgba(255,255,255,0.06);
            padding: 24px 40px;
            display: flex;
            justify-content: center;
            gap: 60px;
        }}
        .proof-item {{
            text-align: center;
        }}
        .proof-number {{
            font-size: 28px;
            font-weight: 800;
            color: white;
        }}
        .proof-label {{
            font-size: 13px;
            color: #525252;
            margin-top: 4px;
        }}
        
        /* Features Grid */
        .features {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 80px 40px;
        }}
        .features h2 {{
            font-size: 32px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 48px;
        }}
        .features-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
        }}
        .feature-card {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 32px;
            transition: all 0.3s;
        }}
        .feature-card:hover {{
            background: rgba(255,255,255,0.06);
            transform: translateY(-4px);
            border-color: rgba(99,102,241,0.3);
        }}
        .feature-icon {{
            width: 48px;
            height: 48px;
            border-radius: 12px;
            background: linear-gradient(135deg, {primary}22, {secondary}22);
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 16px;
            font-size: 24px;
        }}
        .feature-card h3 {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .feature-card p {{
            font-size: 14px;
            color: #737373;
            line-height: 1.5;
        }}
        
        /* Urgency Section */
        .urgency {{
            background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(234,88,12,0.08));
            border-top: 1px solid rgba(239,68,68,0.15);
            border-bottom: 1px solid rgba(239,68,68,0.15);
            text-align: center;
            padding: 48px 40px;
        }}
        .countdown {{
            display: flex;
            justify-content: center;
            gap: 16px;
            margin: 24px 0;
        }}
        .countdown-box {{
            background: rgba(0,0,0,0.4);
            border-radius: 12px;
            padding: 16px 24px;
            min-width: 80px;
        }}
        .countdown-num {{
            font-size: 36px;
            font-weight: 800;
            color: #ef4444;
        }}
        .countdown-label {{
            font-size: 12px;
            color: #737373;
            text-transform: uppercase;
        }}
        
        /* Testimonials */
        .testimonials {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 80px 40px;
        }}
        .testimonials h2 {{
            font-size: 32px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 48px;
        }}
        .testimonial-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
        }}
        .testimonial {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 24px;
        }}
        .testimonial-stars {{ color: {accent}; margin-bottom: 12px; }}
        .testimonial-text {{
            font-size: 14px;
            color: #a3a3a3;
            line-height: 1.6;
            margin-bottom: 16px;
            font-style: italic;
        }}
        .testimonial-author {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .testimonial-avatar {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: linear-gradient(135deg, {primary}, {secondary});
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 16px;
        }}
        .testimonial-name {{ font-size: 14px; font-weight: 600; }}
        .testimonial-verified {{ font-size: 12px; color: #22c55e; }}
        
        /* Footer CTA */
        .footer-cta {{
            text-align: center;
            padding: 80px 40px;
            background: linear-gradient(180deg, transparent, rgba(99,102,241,0.08));
        }}
        .footer-cta h2 {{
            font-size: 36px;
            font-weight: 800;
            margin-bottom: 16px;
        }}
        .footer-cta p {{
            font-size: 16px;
            color: #737373;
            margin-bottom: 32px;
        }}
        
        /* Footer */
        footer {{
            background: rgba(255,255,255,0.02);
            border-top: 1px solid rgba(255,255,255,0.06);
            padding: 40px;
            text-align: center;
            font-size: 13px;
            color: #525252;
        }}
        
        @media (max-width: 768px) {{
            .hero {{ grid-template-columns: 1fr; padding: 40px 20px; }}
            .features-grid {{ grid-template-columns: 1fr; }}
            .testimonial-grid {{ grid-template-columns: 1fr; }}
            .social-proof {{ flex-wrap: wrap; gap: 24px; }}
            .hero h1 {{ font-size: 28px; }}
            .price-current {{ font-size: 36px; }}
        }}
    </style>
</head>
<body>
    <!-- Announcement Bar -->
    <div class="announcement">
        🔥 FLASH SALE: {discount_pct}% OFF — Limited Time Only! Free Shipping on $49+ Orders
    </div>
    
    <!-- Navigation -->
    <nav>
        <div class="logo">{brand}</div>
        <ul class="nav-links">
            <li><a href="#">Shop</a></li>
            <li><a href="#">Best Sellers</a></li>
            <li><a href="#">Reviews</a></li>
            <li><a href="#">Contact</a></li>
        </ul>
    </nav>
    
    <!-- Hero Section -->
    <section class="hero">
        <div class="hero-image">📦</div>
        <div class="hero-content">
            <div class="badge">⭐ #{1} Best Seller</div>
            <h1>{title}</h1>
            <p class="hero-desc">{description}</p>
            <div class="rating">
                {"⭐" * int(rating)} <span style="color:#a3a3a3">({rating}/5 from 2,847 reviews)</span>
            </div>
            <div class="price-block">
                <span class="price-current">${price}</span>
                <span class="price-compare">${compare_price}</span>
                <span class="price-save">SAVE {discount_pct}%</span>
            </div>
            <button class="cta-btn" id="cta-hero">🛒 Add to Cart — ${price}</button>
            <div class="trust-badges">
                <span class="trust-badge">🔒 Secure Checkout</span>
                <span class="trust-badge">🚚 Free Shipping</span>
                <span class="trust-badge">↩️ 30-Day Returns</span>
            </div>
        </div>
    </section>
    
    <!-- Social Proof Bar -->
    <section class="social-proof">
        <div class="proof-item">
            <div class="proof-number">15,000+</div>
            <div class="proof-label">Happy Customers</div>
        </div>
        <div class="proof-item">
            <div class="proof-number">4.8⭐</div>
            <div class="proof-label">Average Rating</div>
        </div>
        <div class="proof-item">
            <div class="proof-number">98%</div>
            <div class="proof-label">Satisfaction Rate</div>
        </div>
        <div class="proof-item">
            <div class="proof-number">#1</div>
            <div class="proof-label">Best Seller</div>
        </div>
    </section>
    
    <!-- Features Grid -->
    <section class="features">
        <h2>Why Customers Love This Product</h2>
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h3>Premium Quality</h3>
                <p>Made with top-grade materials for lasting durability and everyday use.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h3>Precision Design</h3>
                <p>Ergonomically designed for maximum comfort and effortless functionality.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🛡️</div>
                <h3>Lifetime Warranty</h3>
                <p>We stand behind our products with a comprehensive warranty program.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📦</div>
                <h3>Fast Shipping</h3>
                <p>Orders processed within 24 hours with tracking on every shipment.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">💎</div>
                <h3>Value for Money</h3>
                <p>Get premium quality without the premium price tag. Best value guaranteed.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🌟</div>
                <h3>Award Winning</h3>
                <p>Recognized by industry experts for exceptional innovation and design.</p>
            </div>
        </div>
    </section>
    
    <!-- Urgency Timer -->
    <section class="urgency">
        <h2 style="color: #ef4444;">⚡ Flash Sale Ends In:</h2>
        <div class="countdown" id="countdown">
            <div class="countdown-box">
                <div class="countdown-num" id="hours">02</div>
                <div class="countdown-label">Hours</div>
            </div>
            <div class="countdown-box">
                <div class="countdown-num" id="minutes">47</div>
                <div class="countdown-label">Minutes</div>
            </div>
            <div class="countdown-box">
                <div class="countdown-num" id="seconds">33</div>
                <div class="countdown-label">Seconds</div>
            </div>
        </div>
        <p style="color: #a3a3a3;">Only <strong style="color:#ef4444;">23 items</strong> left at this price!</p>
    </section>
    
    <!-- Testimonials -->
    <section class="testimonials">
        <h2>What Our Customers Say</h2>
        <div class="testimonial-grid">
            <div class="testimonial">
                <div class="testimonial-stars">⭐⭐⭐⭐⭐</div>
                <p class="testimonial-text">"Absolutely exceeded my expectations! The quality is outstanding and it arrived faster than expected. Will definitely buy again."</p>
                <div class="testimonial-author">
                    <div class="testimonial-avatar">S</div>
                    <div>
                        <div class="testimonial-name">Sarah M.</div>
                        <div class="testimonial-verified">✅ Verified Purchase</div>
                    </div>
                </div>
            </div>
            <div class="testimonial">
                <div class="testimonial-stars">⭐⭐⭐⭐⭐</div>
                <p class="testimonial-text">"Best purchase I've made this year. The build quality is incredible and it works exactly as described. Highly recommend!"</p>
                <div class="testimonial-author">
                    <div class="testimonial-avatar">J</div>
                    <div>
                        <div class="testimonial-name">James R.</div>
                        <div class="testimonial-verified">✅ Verified Purchase</div>
                    </div>
                </div>
            </div>
            <div class="testimonial">
                <div class="testimonial-stars">⭐⭐⭐⭐⭐</div>
                <p class="testimonial-text">"I was skeptical at first, but this product is amazing. Perfect for daily use and the customer service is top-notch."</p>
                <div class="testimonial-author">
                    <div class="testimonial-avatar">A</div>
                    <div>
                        <div class="testimonial-name">Alex K.</div>
                        <div class="testimonial-verified">✅ Verified Purchase</div>
                    </div>
                </div>
            </div>
        </div>
    </section>
    
    <!-- Footer CTA -->
    <section class="footer-cta">
        <h2>Ready to Experience the Difference?</h2>
        <p>Join 15,000+ happy customers today. Risk-free with our 30-day guarantee.</p>
        <button class="cta-btn" id="cta-footer">🛒 Get Yours Now — ${price}</button>
    </section>
    
    <!-- Footer -->
    <footer>
        <p>&copy; {datetime.now().year} {brand}. All rights reserved. | Privacy Policy | Terms of Service</p>
    </footer>
    
    <script>
        // Countdown Timer
        let totalSeconds = 2 * 3600 + 47 * 60 + 33;
        function updateTimer() {{
            if (totalSeconds <= 0) return;
            totalSeconds--;
            const h = Math.floor(totalSeconds / 3600);
            const m = Math.floor((totalSeconds % 3600) / 60);
            const s = totalSeconds % 60;
            document.getElementById('hours').textContent = String(h).padStart(2, '0');
            document.getElementById('minutes').textContent = String(m).padStart(2, '0');
            document.getElementById('seconds').textContent = String(s).padStart(2, '0');
        }}
        setInterval(updateTimer, 1000);
        
        // CTA click tracking
        document.querySelectorAll('.cta-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                alert('🛒 Item added to cart!');
            }});
        }});
    </script>
</body>
</html>"""

        return html

    def _build_collection_page(self, niche: str) -> str:
        """Build a product collection page"""
        brand = self.store_config['brand_name']
        products_html = ''

        for i, product in enumerate(self.product_catalog[:12]):
            products_html += f"""
            <div class="product-card">
                <div class="product-image">📦</div>
                <h3>{product.get('seo_title', 'Product')}</h3>
                <div class="product-price">${product.get('selling_price', 29.99)}</div>
                <div class="product-compare">${product.get('compare_at_price', 49.99)}</div>
                <button class="add-to-cart">Add to Cart</button>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{niche.title()} Collection | {brand}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Inter', sans-serif; background: #0a0a0a; color: #e5e5e5; }}
        .collection-header {{
            text-align: center;
            padding: 80px 40px 40px;
        }}
        .collection-header h1 {{ font-size: 42px; font-weight: 800; }}
        .products-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 24px;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px;
        }}
        .product-card {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s;
            cursor: pointer;
        }}
        .product-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(99,102,241,0.3);
        }}
        .product-image {{
            font-size: 64px;
            text-align: center;
            padding: 32px;
            background: rgba(99,102,241,0.05);
            border-radius: 12px;
            margin-bottom: 16px;
        }}
        .product-card h3 {{ font-size: 16px; font-weight: 600; margin-bottom: 8px; }}
        .product-price {{ font-size: 24px; font-weight: 800; color: white; }}
        .product-compare {{ font-size: 14px; color: #525252; text-decoration: line-through; }}
        .add-to-cart {{
            width: 100%;
            margin-top: 16px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            border: none;
            padding: 12px;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }}
        .add-to-cart:hover {{ transform: translateY(-1px); }}
    </style>
</head>
<body>
    <div class="collection-header">
        <h1>{niche.title()} Collection</h1>
        <p style="color:#737373; margin-top:12px;">{len(self.product_catalog)} premium products</p>
    </div>
    <div class="products-grid">
        {products_html}
    </div>
</body>
</html>"""

    def _build_homepage(self, niche: str) -> str:
        """Build store homepage"""
        return self._build_product_page(
            self.product_catalog[0] if self.product_catalog else {}
        )

    def _calculate_page_seo_score(self, html: str) -> int:
        """Calculate basic SEO score for a page"""
        score = 0

        if '<title>' in html and '</title>' in html:
            title = re.search(r'<title>(.*?)</title>', html)
            if title and 30 <= len(title.group(1)) <= 70:
                score += 15

        if 'meta name="description"' in html:
            score += 15

        if '<h1>' in html:
            score += 10

        if '<h2>' in html:
            score += 5

        if 'viewport' in html:
            score += 10

        if '@media' in html:
            score += 10

        if 'alt=' in html:
            score += 5

        if 'schema.org' in html or 'application/ld+json' in html:
            score += 10

        # Performance indicators
        if 'preconnect' in html:
            score += 5
        if 'loading="lazy"' in html:
            score += 5

        return min(score, 100)

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            **self.stats,
            'catalog_size': len(self.product_catalog),
            'pages_generated_list': self.landing_pages,
        }


# Need math import
import math
