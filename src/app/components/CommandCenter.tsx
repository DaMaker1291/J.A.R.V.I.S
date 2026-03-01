import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
    Terminal,
    Globe,
    Pen,
    ShoppingBag,
    Activity,
    Search,
    Zap,
    Eye,
    TrendingUp,
    CheckCircle2,
    XCircle,
    Loader2,
    RefreshCw,
    Download,
    ExternalLink,
    BarChart3,
    Cpu,
    Database,
    Shield,
    Layers,
    ArrowRight,
    Crosshair,
    FileText,
    Image,
} from "lucide-react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";

// ─── TYPES ──────────────────────────────────────────────────────

interface EngineStats {
    spider: {
        pages_crawled: number;
        data_points_extracted: number;
        errors: number;
        cache_hits: number;
        last_crawl_time: string | null;
        active: boolean;
        cache_size: number;
        visited_count: number;
    };
    visual: {
        worksheets_processed: number;
        debug_loops_total: number;
        average_quality_score: number;
        available_fonts: number;
    };
    dropshipping: {
        products_sourced: number;
        pages_generated: number;
        seo_score_avg: number;
        catalog_size: number;
    };
}

interface CrawlResult {
    url: string;
    title?: string;
    products?: any[];
    text_content?: string;
    response_time_ms?: number;
    error?: string;
}

type ModuleTab = "overview" | "spider" | "visual" | "dropship" | "startup";

// ─── API HELPERS ────────────────────────────────────────────────

const API_BASE = "http://localhost:5050/api";

async function apiPost(endpoint: string, body: any) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        return await res.json();
    } catch (e: any) {
        return { error: e.message || "API request failed" };
    }
}

async function apiGet(endpoint: string) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`);
        return await res.json();
    } catch (e: any) {
        return { error: e.message || "API request failed" };
    }
}

// ─── SUB-COMPONENTS ─────────────────────────────────────────────

function GlowOrb({ color, size = 200, top, left, opacity = 0.15 }: any) {
    return (
        <div
            style={{
                position: "absolute",
                top,
                left,
                width: size,
                height: size,
                borderRadius: "50%",
                background: color,
                filter: `blur(${size / 2}px)`,
                opacity,
                pointerEvents: "none",
                zIndex: 0,
            }}
        />
    );
}

function StatCard({
    icon: Icon,
    label,
    value,
    subtext,
    color,
    animate = true,
}: {
    icon: any;
    label: string;
    value: string | number;
    subtext?: string;
    color: string;
    animate?: boolean;
}) {
    return (
        <motion.div
            initial={animate ? { opacity: 0, y: 20 } : {}}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
        >
            <Card
                style={{
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(255,255,255,0.06)",
                    padding: "24px",
                    borderRadius: "16px",
                }}
            >
                <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "12px" }}>
                    <div
                        style={{
                            width: 40,
                            height: 40,
                            borderRadius: 10,
                            background: `${color}18`,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                        }}
                    >
                        <Icon size={20} color={color} />
                    </div>
                    <span style={{ fontSize: 13, color: "#737373", fontWeight: 500 }}>{label}</span>
                </div>
                <div style={{ fontSize: 32, fontWeight: 800, color: "#fff" }}>{value}</div>
                {subtext && (
                    <div style={{ fontSize: 12, color: "#525252", marginTop: 4 }}>{subtext}</div>
                )}
            </Card>
        </motion.div>
    );
}

function ActivityLine({
    text,
    status,
    time,
}: {
    text: string;
    status: "success" | "error" | "pending" | "info";
    time: string;
}) {
    const colors = {
        success: "#22c55e",
        error: "#ef4444",
        pending: "#f59e0b",
        info: "#6366f1",
    };
    const icons = {
        success: CheckCircle2,
        error: XCircle,
        pending: Loader2,
        info: Activity,
    };
    const StatusIcon = icons[status];

    return (
        <motion.div
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                padding: "10px 16px",
                borderRadius: 10,
                background: "rgba(255,255,255,0.02)",
                borderLeft: `3px solid ${colors[status]}`,
                marginBottom: 8,
            }}
        >
            <StatusIcon size={16} color={colors[status]} />
            <span style={{ flex: 1, fontSize: 13, color: "#a3a3a3" }}>{text}</span>
            <span style={{ fontSize: 11, color: "#525252" }}>{time}</span>
        </motion.div>
    );
}

// ─── MAIN COMPONENT ──────────────────────────────────────────────

export function CommandCenter() {
    const [activeTab, setActiveTab] = useState<ModuleTab>("overview");
    const [stats, setStats] = useState<EngineStats | null>(null);
    const [loading, setLoading] = useState(false);
    const [backendOnline, setBackendOnline] = useState(false);

    // Spider state
    const [spiderUrl, setSpiderUrl] = useState("");
    const [spiderQuery, setSpiderQuery] = useState("");
    const [spiderResults, setSpiderResults] = useState<CrawlResult | null>(null);
    const [spiderLoading, setSpiderLoading] = useState(false);

    // Dropship state
    const [dropshipNiche, setDropshipNiche] = useState("tech gadgets");
    const [dropshipProducts, setDropshipProducts] = useState<any[]>([]);
    const [dropshipLoading, setDropshipLoading] = useState(false);
    const [generatedPage, setGeneratedPage] = useState<any>(null);

    // Visual state
    const [worksheetPath, setWorksheetPath] = useState("");
    const [referencePath, setReferencePath] = useState("");
    const [visualResult, setVisualResult] = useState<any>(null);
    const [visualLoading, setVisualLoading] = useState(false);

    // Activity log
    const [activityLog, setActivityLog] = useState<
        { text: string; status: "success" | "error" | "pending" | "info"; time: string }[]
    >([
        { text: "Autonomous Execution Engine initialized", status: "success", time: "now" },
        { text: "Spider Crawler module loaded", status: "info", time: "now" },
        { text: "Visual Replication Engine ready", status: "info", time: "now" },
        { text: "Dropshipping Engine ready", status: "info", time: "now" },
    ]);

    const addLog = useCallback(
        (text: string, status: "success" | "error" | "pending" | "info") => {
            setActivityLog((prev) => [
                { text, status, time: new Date().toLocaleTimeString() },
                ...prev.slice(0, 19),
            ]);
        },
        []
    );

    // Check backend health
    useEffect(() => {
        const checkHealth = async () => {
            const res = await apiGet("/health");
            setBackendOnline(!res.error);
            if (!res.error) {
                const statsRes = await apiGet("/stats");
                if (!statsRes.error) setStats(statsRes);
            }
        };
        checkHealth();
        const interval = setInterval(checkHealth, 10000);
        return () => clearInterval(interval);
    }, []);

    // ─── SPIDER ACTIONS ──────────────────────────────────────────

    const handleSpiderCrawl = async () => {
        if (!spiderUrl) return;
        setSpiderLoading(true);
        addLog(`Crawling: ${spiderUrl}`, "pending");
        const result = await apiPost("/spider/crawl", { url: spiderUrl });
        setSpiderResults(result);
        setSpiderLoading(false);
        addLog(
            result.error ? `Crawl failed: ${result.error}` : `Crawled ${spiderUrl} successfully`,
            result.error ? "error" : "success"
        );
    };

    const handleSpiderSearch = async () => {
        if (!spiderQuery) return;
        setSpiderLoading(true);
        addLog(`Searching suppliers for: ${spiderQuery}`, "pending");
        const result = await apiPost("/spider/search-suppliers", { query: spiderQuery });
        setSpiderResults(result);
        setSpiderLoading(false);
        addLog(
            result.error
                ? `Search failed: ${result.error}`
                : `Found ${result.total_products || 0} products`,
            result.error ? "error" : "success"
        );
    };

    // ─── DROPSHIP ACTIONS ─────────────────────────────────────────

    const handleSourceProducts = async () => {
        setDropshipLoading(true);
        addLog(`Sourcing products for: ${dropshipNiche}`, "pending");
        const result = await apiPost("/dropship/source-products", {
            niche: dropshipNiche,
            max_products: 12,
        });
        if (result.products) {
            setDropshipProducts(result.products);
            addLog(`Sourced ${result.count} products for "${dropshipNiche}"`, "success");
        } else {
            addLog(`Product sourcing failed: ${result.error}`, "error");
        }
        setDropshipLoading(false);
    };

    const handleGeneratePage = async () => {
        setDropshipLoading(true);
        addLog("Generating high-conversion landing page...", "pending");
        const result = await apiPost("/dropship/generate-page", {
            page_type: "product",
            niche: dropshipNiche,
        });
        if (!result.error) {
            setGeneratedPage(result);
            addLog(`Landing page generated (SEO Score: ${result.seo_score}/100)`, "success");
        } else {
            addLog(`Page generation failed: ${result.error}`, "error");
        }
        setDropshipLoading(false);
    };

    // ─── VISUAL ENGINE ACTIONS ────────────────────────────────────

    const handleFillWorksheet = async () => {
        if (!worksheetPath) return;
        setVisualLoading(true);
        addLog("Starting worksheet fill pipeline...", "pending");
        const result = await apiPost("/visual/fill-worksheet", {
            blank_worksheet_path: worksheetPath,
            reference_style_path: referencePath || undefined,
        });
        setVisualResult(result);
        setVisualLoading(false);
        if (result.error) {
            addLog(`Worksheet fill failed: ${result.error}`, "error");
        } else {
            addLog(
                `Worksheet filled: quality ${result.quality_score}/10, ${result.debug_iterations} debug loop(s)`,
                "success"
            );
        }
    };

    // ─── TABS CONFIG ──────────────────────────────────────────────

    const tabs: { id: ModuleTab; label: string; icon: any; color: string }[] = [
        { id: "overview", label: "Command Center", icon: Cpu, color: "#6366f1" },
        { id: "spider", label: "Spider Crawler", icon: Globe, color: "#22c55e" },
        { id: "visual", label: "Visual Engine", icon: Pen, color: "#f59e0b" },
        { id: "dropship", label: "Dropship Store", icon: ShoppingBag, color: "#ec4899" },
        { id: "startup", label: "Startup Builder", icon: Layers, color: "#8b5cf6" },
    ];

    return (
        <div
            style={{
                minHeight: "100vh",
                background: "#050505",
                color: "#e5e5e5",
                fontFamily: "'Inter', system-ui, sans-serif",
                position: "relative",
                overflow: "hidden",
            }}
        >
            {/* Background Glow Effects */}
            <GlowOrb color="#6366f1" size={400} top="-200px" left="-100px" opacity={0.08} />
            <GlowOrb color="#8b5cf6" size={300} top="50%" left="80%" opacity={0.06} />
            <GlowOrb color="#22c55e" size={250} top="70%" left="10%" opacity={0.04} />

            {/* Header */}
            <header
                style={{
                    padding: "20px 32px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    borderBottom: "1px solid rgba(255,255,255,0.05)",
                    background: "rgba(5,5,5,0.9)",
                    backdropFilter: "blur(20px)",
                    position: "sticky",
                    top: 0,
                    zIndex: 50,
                }}
            >
                <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                    <motion.div
                        animate={{ rotate: [0, 360] }}
                        transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                    >
                        <Cpu size={28} color="#6366f1" />
                    </motion.div>
                    <div>
                        <h1
                            style={{
                                fontSize: 20,
                                fontWeight: 800,
                                background: "linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899)",
                                WebkitBackgroundClip: "text",
                                WebkitTextFillColor: "transparent",
                                lineHeight: 1.2,
                            }}
                        >
                            J.A.R.V.I.S. AUTONOMOUS ENGINE
                        </h1>
                        <p style={{ fontSize: 12, color: "#525252", marginTop: 2 }}>
                            Execution Protocol v6.1.0 | {new Date().toLocaleDateString()}
                        </p>
                    </div>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                    <Badge
                        style={{
                            background: backendOnline ? "rgba(34,197,94,0.15)" : "rgba(239,68,68,0.15)",
                            color: backendOnline ? "#22c55e" : "#ef4444",
                            border: `1px solid ${backendOnline ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)"}`,
                            padding: "6px 14px",
                            fontSize: 12,
                            fontWeight: 600,
                        }}
                    >
                        <div
                            style={{
                                width: 8,
                                height: 8,
                                borderRadius: "50%",
                                background: backendOnline ? "#22c55e" : "#ef4444",
                                marginRight: 8,
                                animation: backendOnline ? "pulse 2s infinite" : "none",
                            }}
                        />
                        {backendOnline ? "ENGINE ONLINE" : "ENGINE OFFLINE"}
                    </Badge>
                </div>
            </header>

            <div style={{ display: "flex", position: "relative", zIndex: 1 }}>
                {/* Sidebar Nav */}
                <nav
                    style={{
                        width: 240,
                        minHeight: "calc(100vh - 73px)",
                        borderRight: "1px solid rgba(255,255,255,0.05)",
                        padding: "20px 12px",
                        background: "rgba(255,255,255,0.01)",
                    }}
                >
                    {tabs.map((tab) => (
                        <motion.button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            whileHover={{ x: 4 }}
                            whileTap={{ scale: 0.98 }}
                            style={{
                                display: "flex",
                                alignItems: "center",
                                gap: 12,
                                width: "100%",
                                padding: "12px 16px",
                                borderRadius: 12,
                                border: "none",
                                cursor: "pointer",
                                marginBottom: 4,
                                fontSize: 14,
                                fontWeight: activeTab === tab.id ? 600 : 400,
                                color: activeTab === tab.id ? "#fff" : "#737373",
                                background:
                                    activeTab === tab.id
                                        ? `linear-gradient(135deg, ${tab.color}18, ${tab.color}08)`
                                        : "transparent",
                                borderLeft: activeTab === tab.id ? `3px solid ${tab.color}` : "3px solid transparent",
                                transition: "all 0.2s",
                                fontFamily: "inherit",
                            }}
                        >
                            <tab.icon size={18} color={activeTab === tab.id ? tab.color : "#525252"} />
                            {tab.label}
                        </motion.button>
                    ))}

                    {/* Activity Log (Sidebar) */}
                    <div
                        style={{
                            marginTop: 32,
                            padding: "16px 0",
                            borderTop: "1px solid rgba(255,255,255,0.05)",
                        }}
                    >
                        <div
                            style={{
                                fontSize: 11,
                                fontWeight: 700,
                                color: "#525252",
                                textTransform: "uppercase",
                                letterSpacing: 1,
                                padding: "0 16px",
                                marginBottom: 12,
                            }}
                        >
                            Live Activity
                        </div>
                        <div style={{ maxHeight: 300, overflow: "auto" }}>
                            {activityLog.slice(0, 8).map((log, i) => (
                                <div
                                    key={i}
                                    style={{
                                        padding: "6px 16px",
                                        fontSize: 11,
                                        color: "#525252",
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 8,
                                    }}
                                >
                                    <div
                                        style={{
                                            width: 6,
                                            height: 6,
                                            borderRadius: "50%",
                                            background:
                                                log.status === "success"
                                                    ? "#22c55e"
                                                    : log.status === "error"
                                                        ? "#ef4444"
                                                        : log.status === "pending"
                                                            ? "#f59e0b"
                                                            : "#6366f1",
                                            flexShrink: 0,
                                        }}
                                    />
                                    <span style={{ flex: 1, lineHeight: 1.3 }}>{log.text}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </nav>

                {/* Main Content */}
                <main style={{ flex: 1, padding: "28px 32px", overflow: "auto" }}>
                    <AnimatePresence mode="wait">
                        {/* ─── OVERVIEW TAB ──────────────────────────────── */}
                        {activeTab === "overview" && (
                            <motion.div
                                key="overview"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                transition={{ duration: 0.3 }}
                            >
                                <h2 style={{ fontSize: 28, fontWeight: 800, marginBottom: 8 }}>
                                    Autonomous Execution Engine
                                </h2>
                                <p style={{ color: "#525252", marginBottom: 32, fontSize: 14 }}>
                                    Real-time intelligence gathering, visual replication, and automated store generation.
                                </p>

                                {/* Stat Cards */}
                                <div
                                    style={{
                                        display: "grid",
                                        gridTemplateColumns: "repeat(4, 1fr)",
                                        gap: 20,
                                        marginBottom: 32,
                                    }}
                                >
                                    <StatCard
                                        icon={Globe}
                                        label="Pages Crawled"
                                        value={stats?.spider?.pages_crawled ?? 0}
                                        subtext={`${stats?.spider?.cache_hits ?? 0} cache hits`}
                                        color="#22c55e"
                                    />
                                    <StatCard
                                        icon={Database}
                                        label="Data Points"
                                        value={stats?.spider?.data_points_extracted ?? 0}
                                        subtext="Extracted from web"
                                        color="#6366f1"
                                    />
                                    <StatCard
                                        icon={Pen}
                                        label="Worksheets"
                                        value={stats?.visual?.worksheets_processed ?? 0}
                                        subtext={`Avg quality: ${(stats?.visual?.average_quality_score ?? 0).toFixed(1)}/10`}
                                        color="#f59e0b"
                                    />
                                    <StatCard
                                        icon={ShoppingBag}
                                        label="Products Sourced"
                                        value={stats?.dropshipping?.products_sourced ?? 0}
                                        subtext={`${stats?.dropshipping?.pages_generated ?? 0} pages generated`}
                                        color="#ec4899"
                                    />
                                </div>

                                {/* Module Cards */}
                                <div
                                    style={{
                                        display: "grid",
                                        gridTemplateColumns: "repeat(3, 1fr)",
                                        gap: 20,
                                        marginBottom: 32,
                                    }}
                                >
                                    {[
                                        {
                                            title: "Spider Crawler",
                                            desc: "Deep-web data acquisition with live pricing and market trend analysis",
                                            icon: Globe,
                                            color: "#22c55e",
                                            features: ["Live supplier search", "SEO analysis", "Deep crawl", "Market trends"],
                                            tab: "spider" as ModuleTab,
                                        },
                                        {
                                            title: "Visual Replication",
                                            desc: "AI-powered handwriting analysis and coordinate-based worksheet filling",
                                            icon: Pen,
                                            color: "#f59e0b",
                                            features: [
                                                "Style analysis",
                                                "Worksheet detection",
                                                "Human variance",
                                                "Debug loop",
                                            ],
                                            tab: "visual" as ModuleTab,
                                        },
                                        {
                                            title: "Dropship Engine",
                                            desc: "Automated product sourcing and high-conversion landing page generation",
                                            icon: ShoppingBag,
                                            color: "#ec4899",
                                            features: ["Product sourcing", "SEO optimization", "Landing pages", "Price strategy"],
                                            tab: "dropship" as ModuleTab,
                                        },
                                    ].map((module) => (
                                        <motion.div
                                            key={module.title}
                                            whileHover={{ y: -6, scale: 1.02 }}
                                            transition={{ type: "spring", stiffness: 300 }}
                                        >
                                            <Card
                                                style={{
                                                    background: "rgba(255,255,255,0.03)",
                                                    border: "1px solid rgba(255,255,255,0.06)",
                                                    borderRadius: 20,
                                                    padding: 28,
                                                    cursor: "pointer",
                                                    transition: "border-color 0.3s",
                                                    height: "100%",
                                                }}
                                                onClick={() => setActiveTab(module.tab)}
                                            >
                                                <div
                                                    style={{
                                                        width: 52,
                                                        height: 52,
                                                        borderRadius: 14,
                                                        background: `${module.color}15`,
                                                        display: "flex",
                                                        alignItems: "center",
                                                        justifyContent: "center",
                                                        marginBottom: 16,
                                                    }}
                                                >
                                                    <module.icon size={24} color={module.color} />
                                                </div>
                                                <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 8 }}>
                                                    {module.title}
                                                </h3>
                                                <p
                                                    style={{
                                                        fontSize: 13,
                                                        color: "#737373",
                                                        lineHeight: 1.5,
                                                        marginBottom: 16,
                                                    }}
                                                >
                                                    {module.desc}
                                                </p>
                                                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                                                    {module.features.map((f) => (
                                                        <Badge
                                                            key={f}
                                                            style={{
                                                                background: `${module.color}10`,
                                                                color: module.color,
                                                                border: `1px solid ${module.color}25`,
                                                                fontSize: 11,
                                                                padding: "3px 10px",
                                                            }}
                                                        >
                                                            {f}
                                                        </Badge>
                                                    ))}
                                                </div>
                                                <div
                                                    style={{
                                                        display: "flex",
                                                        alignItems: "center",
                                                        gap: 6,
                                                        marginTop: 20,
                                                        color: module.color,
                                                        fontSize: 13,
                                                        fontWeight: 600,
                                                    }}
                                                >
                                                    Launch Module <ArrowRight size={14} />
                                                </div>
                                            </Card>
                                        </motion.div>
                                    ))}
                                </div>

                                {/* Activity Feed */}
                                <Card
                                    style={{
                                        background: "rgba(255,255,255,0.02)",
                                        border: "1px solid rgba(255,255,255,0.05)",
                                        borderRadius: 16,
                                        padding: 24,
                                    }}
                                >
                                    <h3
                                        style={{
                                            fontSize: 16,
                                            fontWeight: 700,
                                            marginBottom: 16,
                                            display: "flex",
                                            alignItems: "center",
                                            gap: 8,
                                        }}
                                    >
                                        <Activity size={18} color="#6366f1" /> Execution Log
                                    </h3>
                                    {activityLog.map((log, i) => (
                                        <ActivityLine key={i} text={log.text} status={log.status} time={log.time} />
                                    ))}
                                </Card>
                            </motion.div>
                        )}

                        {/* ─── SPIDER CRAWLER TAB ────────────────────────── */}
                        {activeTab === "spider" && (
                            <motion.div
                                key="spider"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                transition={{ duration: 0.3 }}
                            >
                                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
                                    <Globe size={28} color="#22c55e" />
                                    <h2 style={{ fontSize: 28, fontWeight: 800 }}>Spider Crawler</h2>
                                </div>
                                <p style={{ color: "#525252", marginBottom: 32, fontSize: 14 }}>
                                    Deep-web data acquisition engine — crawl, search, and analyze in real-time.
                                </p>

                                {/* URL Crawler */}
                                <Card
                                    style={{
                                        background: "rgba(255,255,255,0.03)",
                                        border: "1px solid rgba(255,255,255,0.06)",
                                        borderRadius: 16,
                                        padding: 24,
                                        marginBottom: 20,
                                    }}
                                >
                                    <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
                                        <Crosshair size={16} style={{ marginRight: 8, verticalAlign: "middle" }} />
                                        URL Crawler
                                    </h3>
                                    <div style={{ display: "flex", gap: 12 }}>
                                        <input
                                            type="text"
                                            value={spiderUrl}
                                            onChange={(e) => setSpiderUrl(e.target.value)}
                                            placeholder="https://example.com/products"
                                            style={{
                                                flex: 1,
                                                padding: "12px 16px",
                                                borderRadius: 10,
                                                border: "1px solid rgba(255,255,255,0.1)",
                                                background: "rgba(0,0,0,0.3)",
                                                color: "#e5e5e5",
                                                fontSize: 14,
                                                fontFamily: "inherit",
                                                outline: "none",
                                            }}
                                            onKeyDown={(e) => e.key === "Enter" && handleSpiderCrawl()}
                                        />
                                        <Button
                                            onClick={handleSpiderCrawl}
                                            disabled={spiderLoading}
                                            style={{
                                                background: "linear-gradient(135deg, #22c55e, #16a34a)",
                                                color: "#fff",
                                                padding: "12px 24px",
                                                borderRadius: 10,
                                                fontWeight: 600,
                                            }}
                                        >
                                            {spiderLoading ? (
                                                <Loader2 size={16} className="animate-spin" />
                                            ) : (
                                                <>
                                                    <Search size={16} style={{ marginRight: 8 }} /> Crawl
                                                </>
                                            )}
                                        </Button>
                                    </div>
                                </Card>

                                {/* Supplier Search */}
                                <Card
                                    style={{
                                        background: "rgba(255,255,255,0.03)",
                                        border: "1px solid rgba(255,255,255,0.06)",
                                        borderRadius: 16,
                                        padding: 24,
                                        marginBottom: 20,
                                    }}
                                >
                                    <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
                                        <ShoppingBag size={16} style={{ marginRight: 8, verticalAlign: "middle" }} />
                                        Supplier Search
                                    </h3>
                                    <div style={{ display: "flex", gap: 12 }}>
                                        <input
                                            type="text"
                                            value={spiderQuery}
                                            onChange={(e) => setSpiderQuery(e.target.value)}
                                            placeholder="wireless earbuds, LED lights, phone cases..."
                                            style={{
                                                flex: 1,
                                                padding: "12px 16px",
                                                borderRadius: 10,
                                                border: "1px solid rgba(255,255,255,0.1)",
                                                background: "rgba(0,0,0,0.3)",
                                                color: "#e5e5e5",
                                                fontSize: 14,
                                                fontFamily: "inherit",
                                                outline: "none",
                                            }}
                                            onKeyDown={(e) => e.key === "Enter" && handleSpiderSearch()}
                                        />
                                        <Button
                                            onClick={handleSpiderSearch}
                                            disabled={spiderLoading}
                                            style={{
                                                background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
                                                color: "#fff",
                                                padding: "12px 24px",
                                                borderRadius: 10,
                                                fontWeight: 600,
                                            }}
                                        >
                                            {spiderLoading ? (
                                                <Loader2 size={16} className="animate-spin" />
                                            ) : (
                                                <>
                                                    <TrendingUp size={16} style={{ marginRight: 8 }} /> Search
                                                </>
                                            )}
                                        </Button>
                                    </div>
                                </Card>

                                {/* Results */}
                                {spiderResults && (
                                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                        <Card
                                            style={{
                                                background: "rgba(255,255,255,0.03)",
                                                border: "1px solid rgba(255,255,255,0.06)",
                                                borderRadius: 16,
                                                padding: 24,
                                            }}
                                        >
                                            <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
                                                <FileText size={16} style={{ marginRight: 8, verticalAlign: "middle" }} />
                                                Crawl Results
                                            </h3>
                                            <pre
                                                style={{
                                                    background: "rgba(0,0,0,0.4)",
                                                    borderRadius: 10,
                                                    padding: 16,
                                                    fontSize: 12,
                                                    color: "#a3a3a3",
                                                    overflow: "auto",
                                                    maxHeight: 400,
                                                    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                                                    lineHeight: 1.6,
                                                }}
                                            >
                                                {JSON.stringify(spiderResults, null, 2)}
                                            </pre>
                                        </Card>
                                    </motion.div>
                                )}
                            </motion.div>
                        )}

                        {/* ─── VISUAL ENGINE TAB ─────────────────────────── */}
                        {activeTab === "visual" && (
                            <motion.div
                                key="visual"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                transition={{ duration: 0.3 }}
                            >
                                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
                                    <Pen size={28} color="#f59e0b" />
                                    <h2 style={{ fontSize: 28, fontWeight: 800 }}>Visual Replication Engine</h2>
                                </div>
                                <p style={{ color: "#525252", marginBottom: 32, fontSize: 14 }}>
                                    AI handwriting analysis + coordinate-based drawing with iterative visual debugging.
                                </p>

                                {/* Worksheet Fill */}
                                <Card
                                    style={{
                                        background: "rgba(255,255,255,0.03)",
                                        border: "1px solid rgba(255,255,255,0.06)",
                                        borderRadius: 16,
                                        padding: 24,
                                        marginBottom: 20,
                                    }}
                                >
                                    <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
                                        <Image size={16} style={{ marginRight: 8, verticalAlign: "middle" }} />
                                        Fill Worksheet Pipeline
                                    </h3>
                                    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                                        <input
                                            type="text"
                                            value={worksheetPath}
                                            onChange={(e) => setWorksheetPath(e.target.value)}
                                            placeholder="Path to blank worksheet image..."
                                            style={{
                                                padding: "12px 16px",
                                                borderRadius: 10,
                                                border: "1px solid rgba(255,255,255,0.1)",
                                                background: "rgba(0,0,0,0.3)",
                                                color: "#e5e5e5",
                                                fontSize: 14,
                                                fontFamily: "inherit",
                                                outline: "none",
                                            }}
                                        />
                                        <input
                                            type="text"
                                            value={referencePath}
                                            onChange={(e) => setReferencePath(e.target.value)}
                                            placeholder="Path to reference handwriting image (optional)..."
                                            style={{
                                                padding: "12px 16px",
                                                borderRadius: 10,
                                                border: "1px solid rgba(255,255,255,0.1)",
                                                background: "rgba(0,0,0,0.3)",
                                                color: "#e5e5e5",
                                                fontSize: 14,
                                                fontFamily: "inherit",
                                                outline: "none",
                                            }}
                                        />
                                        <Button
                                            onClick={handleFillWorksheet}
                                            disabled={visualLoading || !worksheetPath}
                                            style={{
                                                background: "linear-gradient(135deg, #f59e0b, #d97706)",
                                                color: "#000",
                                                padding: "14px 24px",
                                                borderRadius: 10,
                                                fontWeight: 700,
                                                alignSelf: "flex-start",
                                            }}
                                        >
                                            {visualLoading ? (
                                                <>
                                                    <Loader2 size={16} className="animate-spin" style={{ marginRight: 8 }} />
                                                    Processing...
                                                </>
                                            ) : (
                                                <>
                                                    <Eye size={16} style={{ marginRight: 8 }} /> Execute Pipeline
                                                </>
                                            )}
                                        </Button>
                                    </div>

                                    {/* Pipeline info */}
                                    <div
                                        style={{
                                            marginTop: 20,
                                            display: "grid",
                                            gridTemplateColumns: "repeat(5, 1fr)",
                                            gap: 8,
                                        }}
                                    >
                                        {[
                                            "1. Style Analysis",
                                            "2. Layout Detection",
                                            "3. Render Text",
                                            "4. Visual Debug",
                                            "5. Proof of Life",
                                        ].map((step, i) => (
                                            <div
                                                key={step}
                                                style={{
                                                    padding: "10px 12px",
                                                    borderRadius: 8,
                                                    background: "rgba(245,158,11,0.05)",
                                                    border: "1px solid rgba(245,158,11,0.1)",
                                                    fontSize: 11,
                                                    color: "#a3a3a3",
                                                    textAlign: "center",
                                                }}
                                            >
                                                {step}
                                            </div>
                                        ))}
                                    </div>
                                </Card>

                                {/* Visual Results */}
                                {visualResult && (
                                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                        <Card
                                            style={{
                                                background: "rgba(255,255,255,0.03)",
                                                border: `1px solid ${visualResult.quality_score >= 7
                                                    ? "rgba(34,197,94,0.3)"
                                                    : "rgba(245,158,11,0.3)"
                                                    }`,
                                                borderRadius: 16,
                                                padding: 24,
                                            }}
                                        >
                                            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
                                                {visualResult.quality_score >= 7 ? (
                                                    <CheckCircle2 size={24} color="#22c55e" />
                                                ) : (
                                                    <RefreshCw size={24} color="#f59e0b" />
                                                )}
                                                <h3 style={{ fontSize: 16, fontWeight: 700 }}>
                                                    {visualResult.quality_score >= 7
                                                        ? "✅ Proof of Life — PASS"
                                                        : "⚠️ Quality Below Threshold"}
                                                </h3>
                                            </div>
                                            <div
                                                style={{
                                                    display: "grid",
                                                    gridTemplateColumns: "repeat(3, 1fr)",
                                                    gap: 16,
                                                    marginBottom: 16,
                                                }}
                                            >
                                                <div
                                                    style={{
                                                        padding: 16,
                                                        borderRadius: 10,
                                                        background: "rgba(0,0,0,0.3)",
                                                        textAlign: "center",
                                                    }}
                                                >
                                                    <div style={{ fontSize: 28, fontWeight: 800, color: "#f59e0b" }}>
                                                        {visualResult.quality_score}/10
                                                    </div>
                                                    <div style={{ fontSize: 12, color: "#525252" }}>Quality Score</div>
                                                </div>
                                                <div
                                                    style={{
                                                        padding: 16,
                                                        borderRadius: 10,
                                                        background: "rgba(0,0,0,0.3)",
                                                        textAlign: "center",
                                                    }}
                                                >
                                                    <div style={{ fontSize: 28, fontWeight: 800, color: "#6366f1" }}>
                                                        {visualResult.debug_iterations}
                                                    </div>
                                                    <div style={{ fontSize: 12, color: "#525252" }}>Debug Loops</div>
                                                </div>
                                                <div
                                                    style={{
                                                        padding: 16,
                                                        borderRadius: 10,
                                                        background: "rgba(0,0,0,0.3)",
                                                        textAlign: "center",
                                                    }}
                                                >
                                                    <div style={{ fontSize: 28, fontWeight: 800, color: "#22c55e" }}>
                                                        {visualResult.status}
                                                    </div>
                                                    <div style={{ fontSize: 12, color: "#525252" }}>Status</div>
                                                </div>
                                            </div>
                                            {visualResult.output_path && (
                                                <div style={{ fontSize: 13, color: "#737373" }}>
                                                    📁 Output: <code style={{ color: "#a3a3a3" }}>{visualResult.output_path}</code>
                                                </div>
                                            )}
                                        </Card>
                                    </motion.div>
                                )}
                            </motion.div>
                        )}

                        {/* ─── DROPSHIP TAB ──────────────────────────────── */}
                        {activeTab === "dropship" && (
                            <motion.div
                                key="dropship"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                transition={{ duration: 0.3 }}
                            >
                                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
                                    <ShoppingBag size={28} color="#ec4899" />
                                    <h2 style={{ fontSize: 28, fontWeight: 800 }}>Dropshipping Engine</h2>
                                </div>
                                <p style={{ color: "#525252", marginBottom: 32, fontSize: 14 }}>
                                    Source products, optimize pricing, and generate high-conversion landing pages.
                                </p>

                                {/* Niche Search */}
                                <Card
                                    style={{
                                        background: "rgba(255,255,255,0.03)",
                                        border: "1px solid rgba(255,255,255,0.06)",
                                        borderRadius: 16,
                                        padding: 24,
                                        marginBottom: 20,
                                    }}
                                >
                                    <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
                                        <Search size={16} style={{ marginRight: 8, verticalAlign: "middle" }} />
                                        Product Sourcing
                                    </h3>
                                    <div style={{ display: "flex", gap: 12 }}>
                                        <input
                                            type="text"
                                            value={dropshipNiche}
                                            onChange={(e) => setDropshipNiche(e.target.value)}
                                            placeholder="Enter niche: tech gadgets, home decor, fashion..."
                                            style={{
                                                flex: 1,
                                                padding: "12px 16px",
                                                borderRadius: 10,
                                                border: "1px solid rgba(255,255,255,0.1)",
                                                background: "rgba(0,0,0,0.3)",
                                                color: "#e5e5e5",
                                                fontSize: 14,
                                                fontFamily: "inherit",
                                                outline: "none",
                                            }}
                                        />
                                        <Button
                                            onClick={handleSourceProducts}
                                            disabled={dropshipLoading}
                                            style={{
                                                background: "linear-gradient(135deg, #ec4899, #db2777)",
                                                color: "#fff",
                                                padding: "12px 24px",
                                                borderRadius: 10,
                                                fontWeight: 600,
                                            }}
                                        >
                                            {dropshipLoading ? (
                                                <Loader2 size={16} className="animate-spin" />
                                            ) : (
                                                <>
                                                    <Zap size={16} style={{ marginRight: 8 }} /> Source
                                                </>
                                            )}
                                        </Button>
                                        <Button
                                            onClick={handleGeneratePage}
                                            disabled={dropshipLoading}
                                            style={{
                                                background: "linear-gradient(135deg, #8b5cf6, #7c3aed)",
                                                color: "#fff",
                                                padding: "12px 24px",
                                                borderRadius: 10,
                                                fontWeight: 600,
                                            }}
                                        >
                                            {dropshipLoading ? (
                                                <Loader2 size={16} className="animate-spin" />
                                            ) : (
                                                <>
                                                    <FileText size={16} style={{ marginRight: 8 }} /> Generate Page
                                                </>
                                            )}
                                        </Button>
                                    </div>
                                </Card>

                                {/* Product Grid */}
                                {dropshipProducts.length > 0 && (
                                    <div style={{ marginBottom: 20 }}>
                                        <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
                                            Sourced Products ({dropshipProducts.length})
                                        </h3>
                                        <div
                                            style={{
                                                display: "grid",
                                                gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
                                                gap: 16,
                                            }}
                                        >
                                            {dropshipProducts.map((product, i) => (
                                                <motion.div
                                                    key={i}
                                                    initial={{ opacity: 0, y: 20 }}
                                                    animate={{ opacity: 1, y: 0 }}
                                                    transition={{ delay: i * 0.05 }}
                                                >
                                                    <Card
                                                        style={{
                                                            background: "rgba(255,255,255,0.03)",
                                                            border: "1px solid rgba(255,255,255,0.06)",
                                                            borderRadius: 14,
                                                            padding: 20,
                                                        }}
                                                    >
                                                        <h4
                                                            style={{
                                                                fontSize: 14,
                                                                fontWeight: 600,
                                                                marginBottom: 12,
                                                                lineHeight: 1.4,
                                                            }}
                                                        >
                                                            {product.seo_title}
                                                        </h4>
                                                        <div
                                                            style={{
                                                                display: "flex",
                                                                justifyContent: "space-between",
                                                                marginBottom: 8,
                                                            }}
                                                        >
                                                            <span style={{ fontSize: 22, fontWeight: 800, color: "#22c55e" }}>
                                                                ${product.selling_price}
                                                            </span>
                                                            <span
                                                                style={{
                                                                    fontSize: 14,
                                                                    color: "#525252",
                                                                    textDecoration: "line-through",
                                                                }}
                                                            >
                                                                ${product.compare_at_price}
                                                            </span>
                                                        </div>
                                                        <div
                                                            style={{
                                                                display: "flex",
                                                                justifyContent: "space-between",
                                                                fontSize: 12,
                                                                color: "#737373",
                                                            }}
                                                        >
                                                            <span>Cost: ${product.cost}</span>
                                                            <Badge
                                                                style={{
                                                                    background: "rgba(34,197,94,0.15)",
                                                                    color: "#22c55e",
                                                                    fontSize: 11,
                                                                }}
                                                            >
                                                                {product.estimated_margin_pct}% margin
                                                            </Badge>
                                                        </div>
                                                    </Card>
                                                </motion.div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Generated Page */}
                                {generatedPage && (
                                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                        <Card
                                            style={{
                                                background: "rgba(255,255,255,0.03)",
                                                border: "1px solid rgba(139,92,246,0.3)",
                                                borderRadius: 16,
                                                padding: 24,
                                            }}
                                        >
                                            <div
                                                style={{
                                                    display: "flex",
                                                    alignItems: "center",
                                                    justifyContent: "space-between",
                                                    marginBottom: 12,
                                                }}
                                            >
                                                <h3 style={{ fontSize: 16, fontWeight: 700 }}>
                                                    <CheckCircle2
                                                        size={18}
                                                        color="#22c55e"
                                                        style={{ marginRight: 8, verticalAlign: "middle" }}
                                                    />
                                                    Landing Page Generated
                                                </h3>
                                                <Badge
                                                    style={{
                                                        background: "rgba(99,102,241,0.15)",
                                                        color: "#6366f1",
                                                        fontSize: 12,
                                                        fontWeight: 700,
                                                    }}
                                                >
                                                    SEO Score: {generatedPage.seo_score}/100
                                                </Badge>
                                            </div>
                                            <p style={{ fontSize: 13, color: "#737373" }}>
                                                📁 Saved to:{" "}
                                                <code style={{ color: "#a3a3a3" }}>{generatedPage.filepath}</code>
                                            </p>
                                            <p style={{ fontSize: 12, color: "#525252", marginTop: 4 }}>
                                                Size: {(generatedPage.html_length / 1024).toFixed(1)} KB | Type:{" "}
                                                {generatedPage.page_type}
                                            </p>
                                        </Card>
                                    </motion.div>
                                )}
                            </motion.div>
                        )}

                        {/* ─── STARTUP BUILDER TAB ───────────────────────── */}
                        {activeTab === "startup" && (
                            <motion.div
                                key="startup"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                transition={{ duration: 0.3 }}
                            >
                                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
                                    <Layers size={28} color="#8b5cf6" />
                                    <h2 style={{ fontSize: 28, fontWeight: 800 }}>Startup Design Builder</h2>
                                </div>
                                <p style={{ color: "#525252", marginBottom: 32, fontSize: 14 }}>
                                    Generate full tech stacks, wireframes, and initial codebases for startup ideas.
                                </p>

                                {/* Tech Stack Summary */}
                                <div
                                    style={{
                                        display: "grid",
                                        gridTemplateColumns: "repeat(2, 1fr)",
                                        gap: 20,
                                        marginBottom: 24,
                                    }}
                                >
                                    <Card
                                        style={{
                                            background: "rgba(255,255,255,0.03)",
                                            border: "1px solid rgba(255,255,255,0.06)",
                                            borderRadius: 16,
                                            padding: 24,
                                        }}
                                    >
                                        <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
                                            Current Tech Stack
                                        </h3>
                                        {[
                                            { label: "Frontend", value: "React 18 + Vite + TailwindCSS", icon: "⚛️" },
                                            { label: "Backend", value: "Python Flask + Gemini AI", icon: "🐍" },
                                            { label: "Crawler", value: "BeautifulSoup + Requests", icon: "🕷️" },
                                            { label: "Vision", value: "Pillow + Gemini Vision API", icon: "👁️" },
                                            { label: "AI/ML", value: "scikit-learn + Gemini 2.0 Flash", icon: "🧠" },
                                            { label: "Deployment", value: "Vite Build + Netlify", icon: "🚀" },
                                        ].map((item) => (
                                            <div
                                                key={item.label}
                                                style={{
                                                    display: "flex",
                                                    alignItems: "center",
                                                    gap: 12,
                                                    padding: "10px 0",
                                                    borderBottom: "1px solid rgba(255,255,255,0.04)",
                                                }}
                                            >
                                                <span style={{ fontSize: 20 }}>{item.icon}</span>
                                                <span style={{ fontSize: 13, color: "#737373", width: 80 }}>
                                                    {item.label}
                                                </span>
                                                <span style={{ fontSize: 13, fontWeight: 600 }}>{item.value}</span>
                                            </div>
                                        ))}
                                    </Card>

                                    <Card
                                        style={{
                                            background: "rgba(255,255,255,0.03)",
                                            border: "1px solid rgba(255,255,255,0.06)",
                                            borderRadius: 16,
                                            padding: 24,
                                        }}
                                    >
                                        <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
                                            J.A.R.V.I.S. Modules
                                        </h3>
                                        {[
                                            { name: "Spider Crawler", status: "Active", color: "#22c55e" },
                                            { name: "Visual Replication", status: "Active", color: "#22c55e" },
                                            { name: "Dropship Engine", status: "Active", color: "#22c55e" },
                                            { name: "Cipher Protocol", status: "Loaded", color: "#6366f1" },
                                            { name: "Forge Protocol", status: "Loaded", color: "#6366f1" },
                                            { name: "Oracle Protocol", status: "Loaded", color: "#6366f1" },
                                            { name: "Watchtower", status: "Loaded", color: "#6366f1" },
                                            { name: "Ghost Sweep", status: "Loaded", color: "#6366f1" },
                                        ].map((mod) => (
                                            <div
                                                key={mod.name}
                                                style={{
                                                    display: "flex",
                                                    alignItems: "center",
                                                    justifyContent: "space-between",
                                                    padding: "8px 0",
                                                    borderBottom: "1px solid rgba(255,255,255,0.04)",
                                                }}
                                            >
                                                <span style={{ fontSize: 13 }}>{mod.name}</span>
                                                <Badge
                                                    style={{
                                                        background: `${mod.color}15`,
                                                        color: mod.color,
                                                        fontSize: 11,
                                                    }}
                                                >
                                                    {mod.status}
                                                </Badge>
                                            </div>
                                        ))}
                                    </Card>
                                </div>

                                {/* Architecture Diagram */}
                                <Card
                                    style={{
                                        background: "rgba(255,255,255,0.03)",
                                        border: "1px solid rgba(255,255,255,0.06)",
                                        borderRadius: 16,
                                        padding: 24,
                                    }}
                                >
                                    <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
                                        System Architecture
                                    </h3>
                                    <div
                                        style={{
                                            display: "grid",
                                            gridTemplateColumns: "repeat(5, 1fr)",
                                            gap: 8,
                                            textAlign: "center",
                                        }}
                                    >
                                        {[
                                            { label: "React UI", emoji: "🖥️", sub: "Dashboard" },
                                            { label: "→", emoji: "⚡", sub: "REST API" },
                                            { label: "Flask API", emoji: "🔧", sub: "Routing" },
                                            { label: "→", emoji: "🧠", sub: "AI Core" },
                                            { label: "Engines", emoji: "🚀", sub: "Execution" },
                                        ].map((node, i) => (
                                            <div
                                                key={i}
                                                style={{
                                                    padding: 16,
                                                    borderRadius: 10,
                                                    background: i % 2 === 0 ? "rgba(99,102,241,0.06)" : "transparent",
                                                    border:
                                                        i % 2 === 0 ? "1px solid rgba(99,102,241,0.15)" : "none",
                                                }}
                                            >
                                                <div style={{ fontSize: 28, marginBottom: 4 }}>{node.emoji}</div>
                                                <div style={{ fontSize: 13, fontWeight: 600 }}>{node.label}</div>
                                                <div style={{ fontSize: 11, color: "#525252" }}>{node.sub}</div>
                                            </div>
                                        ))}
                                    </div>
                                </Card>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </main>
            </div>

            {/* Inline keyframes */}
            <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        input::placeholder {
          color: #404040;
        }
        input:focus {
          border-color: rgba(99,102,241,0.5) !important;
          box-shadow: 0 0 0 3px rgba(99,102,241,0.1);
        }
        ::-webkit-scrollbar {
          width: 6px;
        }
        ::-webkit-scrollbar-track {
          background: transparent;
        }
        ::-webkit-scrollbar-thumb {
          background: rgba(255,255,255,0.1);
          border-radius: 3px;
        }
      `}</style>
        </div>
    );
}
