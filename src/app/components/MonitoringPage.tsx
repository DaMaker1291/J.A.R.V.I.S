import { useState, useEffect } from "react";
import { Link } from "react-router";
import { 
  Terminal, 
  ArrowLeft, 
  Cpu, 
  HardDrive, 
  Activity, 
  Wifi,
  RefreshCw,
  Server,
  Zap,
  TrendingUp,
  TrendingDown
} from "lucide-react";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface SystemMetrics {
  cpu: number;
  memory: number;
  disk: number;
  network: number;
}

interface ProcessInfo {
  id: number;
  name: string;
  cpu: number;
  memory: number;
}

export function MonitoringPage() {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    cpu: 24.3,
    memory: 68.5,
    disk: 45.2,
    network: 12.4
  });

  const [cpuHistory, setCpuHistory] = useState<Array<{ time: string; value: number }>>([
    { time: "00:00", value: 20 },
    { time: "00:05", value: 25 },
    { time: "00:10", value: 22 },
    { time: "00:15", value: 28 },
    { time: "00:20", value: 24 }
  ]);

  const [memoryHistory, setMemoryHistory] = useState<Array<{ time: string; value: number }>>([
    { time: "00:00", value: 65 },
    { time: "00:05", value: 67 },
    { time: "00:10", value: 66 },
    { time: "00:15", value: 69 },
    { time: "00:20", value: 68.5 }
  ]);

  const [processes, setProcesses] = useState<ProcessInfo[]>([
    { id: 5678, name: "VS Code", cpu: 8.7, memory: 512 },
    { id: 4521, name: "Safari", cpu: 12.4, memory: 245 },
    { id: 7890, name: "Slack", cpu: 2.1, memory: 334 },
    { id: 3456, name: "Terminal", cpu: 0.8, memory: 89 },
    { id: 1234, name: "Finder", cpu: 0.5, memory: 128 },
  ]);

  const [isRefreshing, setIsRefreshing] = useState(false);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => ({
        cpu: Math.max(10, Math.min(90, prev.cpu + (Math.random() - 0.5) * 8)),
        memory: Math.max(50, Math.min(85, prev.memory + (Math.random() - 0.5) * 4)),
        disk: Math.max(40, Math.min(60, prev.disk + (Math.random() - 0.5) * 2)),
        network: Math.max(5, Math.min(30, prev.network + (Math.random() - 0.5) * 6))
      }));

      // Update CPU history
      setCpuHistory(prev => {
        const newData = [...prev.slice(-19)];
        const now = new Date();
        newData.push({
          time: `${now.getMinutes()}:${now.getSeconds().toString().padStart(2, '0')}`,
          value: Math.max(10, Math.min(90, prev[prev.length - 1].value + (Math.random() - 0.5) * 8))
        });
        return newData;
      });

      // Update memory history
      setMemoryHistory(prev => {
        const newData = [...prev.slice(-19)];
        const now = new Date();
        newData.push({
          time: `${now.getMinutes()}:${now.getSeconds().toString().padStart(2, '0')}`,
          value: Math.max(50, Math.min(85, prev[prev.length - 1].value + (Math.random() - 0.5) * 4))
        });
        return newData;
      });

      // Occasionally update processes
      if (Math.random() > 0.7) {
        setProcesses(prev => prev.map(p => ({
          ...p,
          cpu: Math.max(0, Math.min(20, p.cpu + (Math.random() - 0.5) * 2)),
          memory: Math.max(50, Math.min(600, p.memory + (Math.random() - 0.5) * 20))
        })));
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => setIsRefreshing(false), 1000);
  };

  const getStatusColor = (value: number, type: "cpu" | "memory" | "disk" | "network") => {
    if (type === "cpu" || type === "memory") {
      if (value > 80) return "text-red-600";
      if (value > 60) return "text-yellow-600";
      return "text-emerald-600";
    }
    if (type === "disk") {
      if (value > 80) return "text-red-600";
      if (value > 60) return "text-yellow-600";
      return "text-emerald-600";
    }
    return "text-[#002FA7]";
  };

  const getProgressColor = (value: number) => {
    if (value > 80) return "bg-red-600";
    if (value > 60) return "bg-yellow-500";
    return "bg-emerald-600";
  };

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-xl border-b border-neutral-200/60">
        <div className="max-w-[1400px] mx-auto px-8 py-4">
          <div className="flex items-center justify-between">
            <Link to="/" className="flex items-center gap-3">
              <div className="size-8 bg-[#002FA7] rounded flex items-center justify-center">
                <Terminal className="size-4 text-white" strokeWidth={2.5} />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold tracking-tight text-neutral-900">J.A.S.O.N.</span>
                <span className="text-xs text-neutral-400">System Monitor</span>
              </div>
            </Link>
            <div className="flex items-center gap-2">
              <Badge className="bg-emerald-50 text-emerald-700 border-emerald-200 text-xs">
                <div className="size-1.5 rounded-full bg-emerald-500 mr-1.5 animate-pulse" />
                Live
              </Badge>
              <Button
                onClick={handleRefresh}
                variant="ghost"
                className="text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100 text-sm h-9"
              >
                <RefreshCw className={`size-4 ${isRefreshing ? 'animate-spin' : ''}`} strokeWidth={2} />
              </Button>
              <Link to="/">
                <Button variant="ghost" className="text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100 text-sm h-9">
                  <ArrowLeft className="size-4 mr-2" strokeWidth={2} />
                  Back
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      <div className="pt-24 pb-12 px-8">
        <div className="max-w-[1400px] mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-semibold tracking-tight text-neutral-900 mb-3">System Monitor</h1>
            <p className="text-neutral-500 text-lg">
              Real-time performance metrics and process management
            </p>
          </div>

          {/* Bento Grid Layout */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            {/* CPU - Large Card */}
            <div className="col-span-2 row-span-2 p-6 bg-white border border-neutral-200/60 rounded-xl">
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="size-10 bg-neutral-100 rounded-lg flex items-center justify-center">
                    <Cpu className="size-5 text-neutral-700" strokeWidth={2} />
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-neutral-500">CPU Usage</h3>
                    <div className={`text-3xl font-semibold tracking-tight ${getStatusColor(metrics.cpu, "cpu")}`}>
                      {metrics.cpu.toFixed(1)}%
                    </div>
                  </div>
                </div>
                <Badge className="bg-neutral-100 text-neutral-600 border-neutral-200 text-xs">8 Cores</Badge>
              </div>
              
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={cpuHistory}>
                  <defs>
                    <linearGradient id="cpuGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#002FA7" stopOpacity={0.15}/>
                      <stop offset="95%" stopColor="#002FA7" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
                  <XAxis dataKey="time" stroke="#9ca3af" fontSize={11} tickLine={false} />
                  <YAxis stroke="#9ca3af" fontSize={11} domain={[0, 100]} tickLine={false} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'white', 
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      fontSize: '12px'
                    }}
                    formatter={(value: number) => [`${value.toFixed(1)}%`, 'CPU']}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#002FA7" 
                    strokeWidth={2}
                    fill="url(#cpuGrad)" 
                  />
                </AreaChart>
              </ResponsiveContainer>

              <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-neutral-200">
                <div>
                  <div className="text-xs text-neutral-500 mb-1">Temperature</div>
                  <div className="text-sm font-semibold text-neutral-900">42°C</div>
                </div>
                <div>
                  <div className="text-xs text-neutral-500 mb-1">Processes</div>
                  <div className="text-sm font-semibold text-neutral-900">247</div>
                </div>
                <div>
                  <div className="text-xs text-neutral-500 mb-1">Type</div>
                  <div className="text-sm font-semibold text-neutral-900">M2 Pro</div>
                </div>
              </div>
            </div>

            {/* Memory */}
            <div className="p-6 bg-white border border-neutral-200/60 rounded-xl">
              <div className="flex items-center gap-3 mb-4">
                <div className="size-8 bg-neutral-100 rounded-lg flex items-center justify-center">
                  <Server className="size-4 text-neutral-700" strokeWidth={2} />
                </div>
                <div className="flex-1">
                  <div className="text-xs text-neutral-500">Memory</div>
                  <div className={`text-xl font-semibold tracking-tight ${getStatusColor(metrics.memory, "memory")}`}>
                    {metrics.memory.toFixed(1)}%
                  </div>
                </div>
              </div>
              <Progress value={metrics.memory} className="h-2 mb-3" indicatorClassName={getProgressColor(metrics.memory)} />
              <div className="flex items-center justify-between text-xs">
                <span className="text-neutral-500">11.2 GB</span>
                <span className="text-neutral-500">/ 16.0 GB</span>
              </div>
            </div>

            {/* Disk */}
            <div className="p-6 bg-white border border-neutral-200/60 rounded-xl">
              <div className="flex items-center gap-3 mb-4">
                <div className="size-8 bg-neutral-100 rounded-lg flex items-center justify-center">
                  <HardDrive className="size-4 text-neutral-700" strokeWidth={2} />
                </div>
                <div className="flex-1">
                  <div className="text-xs text-neutral-500">Disk Usage</div>
                  <div className={`text-xl font-semibold tracking-tight ${getStatusColor(metrics.disk, "disk")}`}>
                    {metrics.disk.toFixed(1)}%
                  </div>
                </div>
              </div>
              <Progress value={metrics.disk} className="h-2 mb-3" indicatorClassName={getProgressColor(metrics.disk)} />
              <div className="flex items-center justify-between text-xs">
                <span className="text-neutral-500">226 GB</span>
                <span className="text-neutral-500">/ 500 GB</span>
              </div>
            </div>

            {/* Network */}
            <div className="p-6 bg-white border border-neutral-200/60 rounded-xl">
              <div className="flex items-center gap-3 mb-4">
                <div className="size-8 bg-neutral-100 rounded-lg flex items-center justify-center">
                  <Wifi className="size-4 text-neutral-700" strokeWidth={2} />
                </div>
                <div className="flex-1">
                  <div className="text-xs text-neutral-500">Network</div>
                  <div className="text-xl font-semibold tracking-tight text-[#002FA7]">
                    {metrics.network.toFixed(1)} MB/s
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2 text-xs text-neutral-500">
                <TrendingUp className="size-3" strokeWidth={2} />
                <span>Download</span>
              </div>
            </div>

            {/* System Info */}
            <div className="p-6 bg-neutral-50 border border-neutral-200/60 rounded-xl">
              <div className="text-xs font-medium text-neutral-500 mb-3">SYSTEM INFO</div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-neutral-600">Uptime</span>
                  <span className="font-medium text-neutral-900">12d 4h</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-neutral-600">OS</span>
                  <span className="font-medium text-neutral-900">macOS</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-neutral-600">Platform</span>
                  <span className="font-medium text-neutral-900">arm64</span>
                </div>
              </div>
            </div>
          </div>

          {/* Memory History Chart */}
          <div className="mb-6">
            <Card className="bg-white border border-neutral-200/60">
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="size-8 bg-neutral-100 rounded-lg flex items-center justify-center">
                      <Activity className="size-4 text-neutral-700" strokeWidth={2} />
                    </div>
                    <div>
                      <CardTitle className="text-base text-neutral-900">Memory Utilization</CardTitle>
                      <CardDescription className="text-xs text-neutral-500">Real-time tracking over last 60 seconds</CardDescription>
                    </div>
                  </div>
                  <Badge className="bg-neutral-100 text-neutral-600 border-neutral-200 text-xs">LPDDR5</Badge>
                </div>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={180}>
                  <LineChart data={memoryHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
                    <XAxis dataKey="time" stroke="#9ca3af" fontSize={11} tickLine={false} />
                    <YAxis stroke="#9ca3af" fontSize={11} domain={[0, 100]} tickLine={false} />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'white', 
                        border: '1px solid #e5e7eb',
                        borderRadius: '8px',
                        fontSize: '12px'
                      }}
                      formatter={(value: number) => [`${value.toFixed(1)}%`, 'Memory']}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#002FA7" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Process List */}
          <Card className="bg-white border border-neutral-200/60">
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="size-8 bg-neutral-100 rounded-lg flex items-center justify-center">
                    <Zap className="size-4 text-neutral-700" strokeWidth={2} />
                  </div>
                  <div>
                    <CardTitle className="text-base text-neutral-900">Active Processes</CardTitle>
                    <CardDescription className="text-xs text-neutral-500">Top 5 by resource consumption</CardDescription>
                  </div>
                </div>
                <Badge className="bg-neutral-100 text-neutral-600 border-neutral-200 text-xs">247 Total</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-1">
                {/* Header Row */}
                <div className="grid grid-cols-12 gap-4 px-4 py-2 text-xs font-medium text-neutral-500 border-b border-neutral-200">
                  <div className="col-span-1">PID</div>
                  <div className="col-span-4">Process Name</div>
                  <div className="col-span-2 text-right">CPU %</div>
                  <div className="col-span-3"></div>
                  <div className="col-span-2 text-right">Memory</div>
                </div>
                
                {/* Process Rows */}
                {processes.map((process) => (
                  <div key={process.id} className="grid grid-cols-12 gap-4 px-4 py-3 hover:bg-neutral-50 rounded-lg transition-colors items-center">
                    <div className="col-span-1 text-xs text-neutral-500 font-mono">{process.id}</div>
                    <div className="col-span-4 text-sm text-neutral-900 font-medium">{process.name}</div>
                    <div className="col-span-2 text-right text-sm font-semibold text-[#002FA7]">{process.cpu.toFixed(1)}%</div>
                    <div className="col-span-3">
                      <Progress value={process.cpu * 5} className="h-1.5" indicatorClassName="bg-[#002FA7]" />
                    </div>
                    <div className="col-span-2 text-right text-sm font-medium text-neutral-600">{process.memory} MB</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}