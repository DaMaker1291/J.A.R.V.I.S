"""
J.A.S.O.N. Autonomous Execution Engine — Unified API Server
Integrates Spider Crawler, Visual Replication, and Dropshipping Engine
with the React Command Center dashboard.
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

# Flask for API
try:
    from flask import Flask, jsonify, request, send_file, send_from_directory
    from flask_cors import CORS
    flask_available = True
except ImportError:
    flask_available = False

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jason.core.spider_crawler import SpiderCrawler
from jason.core.visual_replication import VisualReplicationEngine
from jason.core.dropshipping_engine import DropshippingEngine

logger = logging.getLogger(__name__)

# Load config
CONFIG_PATH = PROJECT_ROOT / 'config.yaml'
config = {}
try:
    import yaml
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
except Exception:
    logger.warning("Could not load config.yaml, using defaults")


def create_app() -> 'Flask':
    """Create and configure the Flask application"""
    if not flask_available:
        raise RuntimeError("Flask not installed. Run: pip install flask flask-cors")

    app = Flask(__name__, static_folder=str(PROJECT_ROOT / 'output'))
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize engines
    spider = SpiderCrawler(config)
    visual_engine = VisualReplicationEngine(config)
    dropship_engine = DropshippingEngine(config, spider_crawler=spider)

    # ─── HEALTH & STATUS ──────────────────────────────────────────

    @app.route('/api/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'operational',
            'engine': 'J.A.S.O.N. Autonomous Execution Engine',
            'version': '6.1.0',
            'timestamp': datetime.now().isoformat(),
            'modules': {
                'spider_crawler': True,
                'visual_replication': True,
                'dropshipping_engine': True,
                'cipher_protocol': True,
                'forge_protocol': True,
            }
        })

    @app.route('/api/stats', methods=['GET'])
    def stats():
        return jsonify({
            'spider': spider.get_crawl_stats(),
            'visual': visual_engine.get_stats(),
            'dropshipping': dropship_engine.get_stats(),
            'timestamp': datetime.now().isoformat(),
        })

    # ─── SPIDER CRAWLER ENDPOINTS ─────────────────────────────────

    @app.route('/api/spider/crawl', methods=['POST'])
    def spider_crawl():
        data = request.json or {}
        url = data.get('url')
        extract_type = data.get('extract_type', 'auto')

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        result = spider.crawl_url(url, extract_type)
        return jsonify(result)

    @app.route('/api/spider/search-suppliers', methods=['POST'])
    def spider_search_suppliers():
        data = request.json or {}
        query = data.get('query')
        suppliers = data.get('suppliers')

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        result = spider.search_suppliers(query, suppliers)
        return jsonify(result)

    @app.route('/api/spider/trends', methods=['POST'])
    def spider_trends():
        data = request.json or {}
        query = data.get('query')

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        result = spider.fetch_market_trends(query)
        return jsonify(result)

    @app.route('/api/spider/deep-crawl', methods=['POST'])
    def spider_deep_crawl():
        data = request.json or {}
        url = data.get('url')
        max_depth = data.get('max_depth', 2)
        max_pages = data.get('max_pages', 20)

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        result = spider.deep_crawl(url, max_depth, max_pages)
        return jsonify(result)

    @app.route('/api/spider/seo-analysis', methods=['POST'])
    def spider_seo():
        data = request.json or {}
        url = data.get('url')

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        result = spider.generate_seo_analysis(url)
        return jsonify(result)

    @app.route('/api/spider/stats', methods=['GET'])
    def spider_stats():
        return jsonify(spider.get_crawl_stats())

    # ─── VISUAL REPLICATION ENDPOINTS ─────────────────────────────

    @app.route('/api/visual/analyze-style', methods=['POST'])
    def visual_analyze_style():
        data = request.json or {}
        image_path = data.get('image_path')

        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Valid image_path is required'}), 400

        style = visual_engine.analyze_handwriting_style(image_path)
        return jsonify(style.to_dict())

    @app.route('/api/visual/analyze-worksheet', methods=['POST'])
    def visual_analyze_worksheet():
        data = request.json or {}
        image_path = data.get('image_path')

        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Valid image_path is required'}), 400

        fill_data = visual_engine.analyze_worksheet(image_path)
        return jsonify({'fill_regions': fill_data, 'count': len(fill_data)})

    @app.route('/api/visual/fill-worksheet', methods=['POST'])
    def visual_fill_worksheet():
        data = request.json or {}
        blank_path = data.get('blank_worksheet_path')
        reference_path = data.get('reference_style_path')
        custom_answers = data.get('custom_answers')
        output_path = data.get('output_path')

        if not blank_path or not os.path.exists(blank_path):
            return jsonify({'error': 'Valid blank_worksheet_path is required'}), 400

        result = visual_engine.fill_worksheet_pipeline(
            blank_path, reference_path, custom_answers, output_path
        )
        return jsonify(result)

    @app.route('/api/visual/stats', methods=['GET'])
    def visual_stats():
        return jsonify(visual_engine.get_stats())

    # ─── DROPSHIPPING ENGINE ENDPOINTS ─────────────────────────────

    @app.route('/api/dropship/source-products', methods=['POST'])
    def dropship_source():
        data = request.json or {}
        niche = data.get('niche', 'tech gadgets')
        max_products = data.get('max_products', 20)

        products = dropship_engine.source_products(niche, max_products)
        return jsonify({
            'niche': niche,
            'products': products,
            'count': len(products),
        })

    @app.route('/api/dropship/generate-page', methods=['POST'])
    def dropship_generate_page():
        data = request.json or {}
        page_type = data.get('page_type', 'product')
        niche = data.get('niche', 'premium products')
        product_index = data.get('product_index', 0)

        product = None
        if dropship_engine.product_catalog and product_index < len(dropship_engine.product_catalog):
            product = dropship_engine.product_catalog[product_index]

        result = dropship_engine.generate_landing_page(product, page_type, niche)
        return jsonify(result)

    @app.route('/api/dropship/catalog', methods=['GET'])
    def dropship_catalog():
        return jsonify({
            'products': dropship_engine.product_catalog,
            'count': len(dropship_engine.product_catalog),
        })

    @app.route('/api/dropship/stats', methods=['GET'])
    def dropship_stats():
        return jsonify(dropship_engine.get_stats())

    # ─── FILE SERVING ──────────────────────────────────────────────

    @app.route('/api/file/<path:filepath>')
    def serve_file(filepath):
        full_path = PROJECT_ROOT / filepath
        if full_path.exists():
            return send_file(str(full_path))
        return jsonify({'error': 'File not found'}), 404

    @app.route('/api/output/<path:filepath>')
    def serve_output(filepath):
        output_dir = PROJECT_ROOT / 'output'
        full_path = output_dir / filepath
        if full_path.exists():
            return send_file(str(full_path))
        return jsonify({'error': 'File not found'}), 404

    # ─── ERROR HANDLERS ───────────────────────────────────────────

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
        }), 500

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            'error': 'Endpoint not found',
        }), 404

    return app


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  J.A.S.O.N. AUTONOMOUS EXECUTION ENGINE — API Server           ║
║  Version: 6.1.0                                                 ║
║  Modules: Spider Crawler | Visual Engine | Dropship Engine      ║
║  Server:  http://localhost:5050                                 ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    app.run(host='0.0.0.0', port=5050, debug=True)
