import { useState } from "react";
import { motion } from "motion/react";
import { Search, ShoppingCart, TrendingUp } from "lucide-react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Badge } from "./ui/badge";

export function DropshippingPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [products, setProducts] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const apiBase = "http://localhost:8000";

  const searchProducts = async () => {
    if (!searchQuery.trim()) return;
    setLoading(true);
    try {
      const response = await fetch(`${apiBase}/search-products?q=${encodeURIComponent(searchQuery)}&limit=20`);
      const data = await response.json();
      if (data.products) {
        setProducts(data.products);
      } else {
        setProducts([]);
      }
    } catch (error) {
      console.error("Search failed:", error);
    }
    setLoading(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      searchProducts();
    }
  };

  return (
    <div className="p-8 lg:pt-8 pt-20 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-white mb-1">Dropshipping Store Builder</h1>
          <p className="text-sm text-zinc-400">Find trending products from AliExpress suppliers</p>
        </div>
      </div>

      {/* Search */}
      <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm">
        <div className="flex gap-4">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Search for products (e.g., wireless headphones)"
            className="flex-1 px-4 py-2 bg-black/20 border border-white/10 rounded-lg text-white placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <Button onClick={searchProducts} disabled={loading} className="bg-blue-600 hover:bg-blue-700 text-white">
            <Search className="w-4 h-4 mr-2" />
            {loading ? "Searching..." : "Search"}
          </Button>
        </div>
      </Card>

      {/* Products */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {products.map((product, index) => (
          <motion.div
            key={product.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05, type: "spring", stiffness: 380, damping: 30 }}
          >
            <Card className="bg-white/5 border-white/5 p-4 backdrop-blur-sm hover:bg-white/[0.07] transition-colors">
              <img
                src={product.image}
                alt={product.title}
                className="w-full h-48 object-cover rounded-lg mb-4"
                onError={(e) => { (e.target as HTMLImageElement).src = "https://via.placeholder.com/300x200"; }}
              />
              <h3 className="text-lg font-semibold text-white mb-2">{product.title}</h3>
              <div className="flex items-center justify-between mb-4">
                <span className="text-2xl font-bold text-green-400">${product.price}</span>
                <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20">
                  <TrendingUp className="w-3 h-3 mr-1" />
                  Trending
                </Badge>
              </div>
              <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                <ShoppingCart className="w-4 h-4 mr-2" />
                Add to Store
              </Button>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* SEO Meta (simulate) */}
      {searchQuery && (
        <div className="bg-white/5 border-white/5 p-6 backdrop-blur-sm rounded-lg">
          <h3 className="text-lg font-semibold text-white mb-4">SEO Optimization</h3>
          <div className="space-y-2">
            <p className="text-sm text-zinc-400">Page Title: {searchQuery} - Best Dropshipping Products</p>
            <p className="text-sm text-zinc-400">Meta Description: Discover high-converting {searchQuery} for your dropshipping store. Fast shipping from AliExpress suppliers.</p>
          </div>
        </div>
      )}

      {/* Heatmap (mock) */}
      <div className="bg-white/5 border-white/5 p-6 backdrop-blur-sm rounded-lg">
        <h3 className="text-lg font-semibold text-white mb-4">Conversion Heatmap</h3>
        <div className="relative w-full h-64 bg-gray-800 rounded-lg overflow-hidden">
          <div className="absolute top-10 left-10 w-20 h-20 bg-red-500 opacity-50 rounded-full"></div>
          <div className="absolute top-20 right-20 w-15 h-15 bg-yellow-500 opacity-50 rounded-full"></div>
          <div className="absolute bottom-10 left-1/2 w-25 h-25 bg-green-500 opacity-50 rounded-full"></div>
          <p className="absolute bottom-2 left-2 text-xs text-zinc-400">Hot areas: Search bar, Add to Store buttons</p>
        </div>
      </div>
    </div>
  );
}
