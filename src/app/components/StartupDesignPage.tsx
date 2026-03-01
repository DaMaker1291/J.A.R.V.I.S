import { useState } from "react";
import { motion } from "motion/react";
import { Rocket, Layout, Eye } from "lucide-react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Badge } from "./ui/badge";

export function StartupDesignPage() {
  const [selectedPage, setSelectedPage] = useState<string | null>(null);

  const pages = [
    {
      name: "Dashboard",
      description: "Main control center with metrics and quick actions",
      wireframe: (
        <div className="bg-gray-100 p-4 rounded-lg">
          <div className="flex justify-between mb-4">
            <div className="bg-gray-300 h-6 w-24 rounded"></div>
            <div className="bg-gray-300 h-6 w-16 rounded"></div>
          </div>
          <div className="grid grid-cols-4 gap-4 mb-4">
            {Array(4).fill(0).map((_, i) => (
              <div key={i} className="bg-gray-300 h-16 rounded"></div>
            ))}
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-300 h-32 rounded"></div>
            <div className="bg-gray-300 h-32 rounded"></div>
          </div>
        </div>
      ),
    },
    {
      name: "Dropshipping",
      description: "Product search and store builder interface",
      wireframe: (
        <div className="bg-gray-100 p-4 rounded-lg">
          <div className="bg-gray-300 h-10 w-full rounded mb-4"></div>
          <div className="grid grid-cols-3 gap-4">
            {Array(3).fill(0).map((_, i) => (
              <div key={i} className="bg-gray-300 h-40 rounded"></div>
            ))}
          </div>
        </div>
      ),
    },
    {
      name: "Handwriting",
      description: "Worksheet analysis and text overlay tool",
      wireframe: (
        <div className="bg-gray-100 p-4 rounded-lg">
          <div className="bg-gray-300 h-20 w-full rounded mb-4"></div>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-300 h-32 rounded"></div>
            <div className="bg-gray-300 h-32 rounded"></div>
          </div>
        </div>
      ),
    },
  ];

  return (
    <div className="p-8 lg:pt-8 pt-20 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-white mb-1">Startup Design System</h1>
          <p className="text-sm text-zinc-400">Live previews and wireframes for all application pages</p>
        </div>
        <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20">
          <Rocket className="w-3 h-3 mr-1" />
          Tech Stack: React + FastAPI + Gemini AI
        </Badge>
      </div>

      {/* Tech Stack */}
      <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm">
        <h3 className="text-lg font-semibold text-white mb-4">Full Tech Stack Overview</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="bg-blue-600/20 w-12 h-12 rounded-lg flex items-center justify-center mx-auto mb-2">
              <Layout className="w-6 h-6 text-blue-400" />
            </div>
            <p className="text-sm text-white font-medium">Frontend</p>
            <p className="text-xs text-zinc-400">React + TypeScript</p>
          </div>
          <div className="text-center">
            <div className="bg-green-600/20 w-12 h-12 rounded-lg flex items-center justify-center mx-auto mb-2">
              <Rocket className="w-6 h-6 text-green-400" />
            </div>
            <p className="text-sm text-white font-medium">Backend</p>
            <p className="text-xs text-zinc-400">FastAPI + Python</p>
          </div>
          <div className="text-center">
            <div className="bg-purple-600/20 w-12 h-12 rounded-lg flex items-center justify-center mx-auto mb-2">
              <Eye className="w-6 h-6 text-purple-400" />
            </div>
            <p className="text-sm text-white font-medium">AI Engine</p>
            <p className="text-xs text-zinc-400">Gemini 2.5 Flash</p>
          </div>
          <div className="text-center">
            <div className="bg-orange-600/20 w-12 h-12 rounded-lg flex items-center justify-center mx-auto mb-2">
              <Layout className="w-6 h-6 text-orange-400" />
            </div>
            <p className="text-sm text-white font-medium">Database</p>
            <p className="text-xs text-zinc-400">SQLite + Prisma</p>
          </div>
        </div>
      </Card>

      {/* Page Previews */}
      <div className="space-y-6">
        <h3 className="text-lg font-semibold text-white">Page Wireframes & Previews</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {pages.map((page, index) => (
            <motion.div
              key={page.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1, type: "spring", stiffness: 380, damping: 30 }}
            >
              <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm hover:bg-white/[0.07] transition-colors">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-lg font-semibold text-white">{page.name}</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedPage(selectedPage === page.name ? null : page.name)}
                  >
                    <Eye className="w-4 h-4" />
                  </Button>
                </div>
                <p className="text-sm text-zinc-400 mb-4">{page.description}</p>
                <div className="bg-gray-800 rounded-lg overflow-hidden">
                  {page.wireframe}
                </div>
                <Button className="w-full mt-4" variant="outline">
                  View Live Preview
                </Button>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Full Preview Modal */}
      {selectedPage && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-zinc-900 rounded-lg p-6 max-w-4xl w-full mx-4">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-semibold text-white">{selectedPage} - Full Preview</h3>
              <Button variant="ghost" onClick={() => setSelectedPage(null)}>×</Button>
            </div>
            <div className="bg-white rounded-lg p-4">
              <div className="text-center text-gray-500 py-20">
                Interactive {selectedPage} page preview would render here
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
