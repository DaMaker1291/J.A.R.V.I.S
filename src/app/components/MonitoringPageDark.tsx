import { useState, useEffect } from "react";
import { motion } from "motion/react";
import { 
  Cpu, 
  HardDrive, 
  Activity, 
  Wifi,
  RefreshCw,
  Server,
  Zap,
  TrendingUp,
} from "lucide-react";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import { Skeleton } from "./ui/skeleton";
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

export function MonitoringPageDark() {
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
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate initial loading
    setTimeout(() => setIsLoading(false), 1500);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => ({
        cpu: Math.max(10, Math.min(90, prev.cpu + (Math.random() - 0.5) * 8)),
        memory: Math.max(50, Math.min(85, prev.memory + (Math.random() - 0.5) * 4)),
        disk: Math.max(40, Math.min(60, prev.disk + (Math.random() - 0.5) * 2)),
        network: Math.max(5, Math.min(30, prev.network + (Math.random() - 0.5) * 6))
      }));

      setCpuHistory(prev => {
        const newData = [...prev.slice(-19)];
        const now = new Date();
        newData.push({
          time: `${now.getMinutes()}:${now.getSeconds().toString().padStart(2, '0')}`,
          value: Math.max(10, Math.min(90, prev[prev.length - 1].value + (Math.random() - 0.5) * 8))
        });
        return newData;
      });

      setMemoryHistory(prev => {
        const newData = [...prev.slice(-19)];
        const now = new Date();
        newData.push({
          time: `${now.getMinutes()}:${now.getSeconds().toString().padStart(2, '0')}`,
          value: Math.max(50, Math.min(85, prev[prev.length - 1].value + (Math.random() - 0.5) * 4))
        });
        return newData;
      });

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
      if (value > 80) return "text-red-400";
      if (value > 60) return "text-amber-400";
      return "text-emerald-400";
    }
    if (type === "disk") {
      if (value > 80) return "text-red-400";
      if (value > 60) return "text-amber-400";
      return "text-emerald-400";
    }
    return "text-blue-400";
  };

  const getProgressColor = (value: number) => {
    if (value > 80) return "bg-red-500";
    if (value > 60) return "bg-amber-500";
    return "bg-emerald-500";
  };

  const StatusIndicator = ({ status }: { status: string }) => (
    <span className="relative flex h-2 w-2">
      <span className={`animate-ping absolute inline-flex h-full w-full rounded-full bg-${status === "optimal" ? "emerald" : status === "warning" ? "amber" : "red"}-500 opacity-75`} />
      <span className={`relative inline-flex rounded-full h-2 w-2 bg-${status === "optimal" ? "emerald" : status === "warning" ? "amber" : "red"}-500`} />
    </span>
  );

  if (isLoading) {
    return (
      <div className="p-8 lg:pt-8 pt-20 space-y-8">
        <div className="space-y-2">
          <Skeleton className="h-9 w-64 bg-white/5" />
          <Skeleton className="h-4 w-96 bg-white/5" />
        </div>
        <div className="grid grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <Skeleton key={i} className="h-32 bg-white/5" />
          ))}
        </div>
        <div className="grid grid-cols-2 gap-4">
          {[...Array(2)].map((_, i) => (
            <Skeleton key={i} className="h-64 bg-white/5" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="p-8 lg:pt-8 pt-20 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-white mb-1">System Monitor</h1>
          <p className="text-sm text-zinc-400">Real-time performance metrics and process management</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 px-3 py-1">
            <span className="relative flex h-2 w-2 mr-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
            </span>
            Live Monitoring
          </Badge>
          <motion.div whileTap={{ scale: 0.95 }}>
            <Button
              onClick={handleRefresh}
              variant="ghost"
              className="text-zinc-400 hover:text-white hover:bg-white/5"
            >
              <RefreshCw className={`size-4 ${isRefreshing ? 'animate-spin' : ''}`} strokeWidth={2} />
            </Button>
          </motion.div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ type: "spring", stiffness: 380, damping: 30 }}
        >
          <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm">
            <div className="flex items-start justify-between mb-4">
              <div className="space-y-1">
                <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">CPU Usage</p>
                <div className={`text-2xl font-semibold tracking-tight ${getStatusColor(metrics.cpu, "cpu")}`}>
                  {metrics.cpu.toFixed(1)}%
                </div>
              </div>
              <div className="size-10 rounded-lg bg-blue-600/20 flex items-center justify-center">
                <Cpu className="size-5 text-blue-400" strokeWidth={2} />
              </div>
            </div>
            <Progress value={metrics.cpu} className="h-2 mb-3" indicatorClassName={getProgressColor(metrics.cpu)} />
            <div className="flex items-center justify-between text-xs text-zinc-500">
              <span>8 Cores</span>
              <span>42°C</span>
            </div>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05, type: "spring", stiffness: 380, damping: 30 }}
        >
          <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm">
            <div className="flex items-start justify-between mb-4">
              <div className="space-y-1">
                <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Memory</p>
                <div className={`text-2xl font-semibold tracking-tight ${getStatusColor(metrics.memory, "memory")}`}>
                  {metrics.memory.toFixed(1)}%
                </div>
              </div>
              <div className="size-10 rounded-lg bg-purple-600/20 flex items-center justify-center">
                <Server className="size-5 text-purple-400" strokeWidth={2} />
              </div>
            </div>
            <Progress value={metrics.memory} className="h-2 mb-3" indicatorClassName={getProgressColor(metrics.memory)} />
            <div className="flex items-center justify-between text-xs text-zinc-500">
              <span>11.2 GB</span>
              <span>/ 16.0 GB</span>
            </div>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, type: "spring", stiffness: 380, damping: 30 }}
        >
          <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm">
            <div className="flex items-start justify-between mb-4">
              <div className="space-y-1">
                <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Disk Usage</p>
                <div className={`text-2xl font-semibold tracking-tight ${getStatusColor(metrics.disk, "disk")}`}>
                  {metrics.disk.toFixed(1)}%
                </div>
              </div>
              <div className="size-10 rounded-lg bg-emerald-600/20 flex items-center justify-center">
                <HardDrive className="size-5 text-emerald-400" strokeWidth={2} />
              </div>
            </div>
            <Progress value={metrics.disk} className="h-2 mb-3" indicatorClassName={getProgressColor(metrics.disk)} />
            <div className="flex items-center justify-between text-xs text-zinc-500">
              <span>226 GB</span>
              <span>/ 500 GB</span>
            </div>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15, type: "spring", stiffness: 380, damping: 30 }}
        >
          <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm">
            <div className="flex items-start justify-between mb-4">
              <div className="space-y-1">
                <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Network</p>
                <div className="text-2xl font-semibold tracking-tight text-blue-400">
                  {metrics.network.toFixed(1)} MB/s
                </div>
              </div>
              <div className="size-10 rounded-lg bg-orange-600/20 flex items-center justify-center">
                <Wifi className="size-5 text-orange-400" strokeWidth={2} />
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <TrendingUp className="size-3" strokeWidth={2} />
              <span>Download</span>
            </div>
          </Card>
        </motion.div>
      </div>

      {/* Charts */}
      <div className="grid lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, type: "spring", stiffness: 380, damping: 30 }}
        >
          <Card className="bg-white/5 border-white/5 backdrop-blur-sm">
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="size-8 rounded-lg bg-blue-600/20 flex items-center justify-center">
                    <Cpu className="size-4 text-blue-400" strokeWidth={2} />
                  </div>
                  <div>
                    <CardTitle className="text-base text-white">CPU History</CardTitle>
                    <CardDescription className="text-xs text-zinc-400">Real-time usage over last 60s</CardDescription>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={cpuHistory}>
                  <defs>
                    <linearGradient id="cpuGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                  <XAxis dataKey="time" stroke="#52525b" fontSize={11} tickLine={false} />
                  <YAxis stroke="#52525b" fontSize={11} domain={[0, 100]} tickLine={false} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#18181b', 
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '8px',
                      fontSize: '12px',
                      color: '#fff'
                    }}
                    formatter={(value: number) => [`${value.toFixed(1)}%`, 'CPU']}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#3b82f6" 
                    strokeWidth={2}
                    fill="url(#cpuGradient)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25, type: "spring", stiffness: 380, damping: 30 }}
        >
          <Card className="bg-white/5 border-white/5 backdrop-blur-sm">
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="size-8 rounded-lg bg-purple-600/20 flex items-center justify-center">
                    <Server className="size-4 text-purple-400" strokeWidth={2} />
                  </div>
                  <div>
                    <CardTitle className="text-base text-white">Memory History</CardTitle>
                    <CardDescription className="text-xs text-zinc-400">Real-time usage over last 60s</CardDescription>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={memoryHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                  <XAxis dataKey="time" stroke="#52525b" fontSize={11} tickLine={false} />
                  <YAxis stroke="#52525b" fontSize={11} domain={[0, 100]} tickLine={false} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#18181b', 
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '8px',
                      fontSize: '12px',
                      color: '#fff'
                    }}
                    formatter={(value: number) => [`${value.toFixed(1)}%`, 'Memory']}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#a855f7" 
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Process List */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, type: "spring", stiffness: 380, damping: 30 }}
      >
        <Card className="bg-white/5 border-white/5 backdrop-blur-sm">
          <CardHeader className="pb-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="size-8 rounded-lg bg-emerald-600/20 flex items-center justify-center">
                  <Zap className="size-4 text-emerald-400" strokeWidth={2} />
                </div>
                <div>
                  <CardTitle className="text-base text-white">Active Processes</CardTitle>
                  <CardDescription className="text-xs text-zinc-400">Top 5 by resource consumption</CardDescription>
                </div>
              </div>
              <Badge className="bg-white/5 text-zinc-300 border-white/10 text-xs">247 Total</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              <div className="grid grid-cols-12 gap-4 px-4 py-2 text-xs font-medium text-zinc-500 border-b border-white/5">
                <div className="col-span-1">PID</div>
                <div className="col-span-4">Process Name</div>
                <div className="col-span-2 text-right">CPU %</div>
                <div className="col-span-3"></div>
                <div className="col-span-2 text-right">Memory</div>
              </div>
              
              {processes.map((process, index) => (
                <motion.div
                  key={process.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.05 * index, type: "spring", stiffness: 500, damping: 40 }}
                  whileHover={{ backgroundColor: "rgba(255,255,255,0.05)" }}
                  className="grid grid-cols-12 gap-4 px-4 py-3 rounded-lg transition-colors items-center"
                >
                  <div className="col-span-1 text-xs text-zinc-500 font-mono">{process.id}</div>
                  <div className="col-span-4 text-sm text-white font-medium">{process.name}</div>
                  <div className="col-span-2 text-right text-sm font-semibold text-blue-400">{process.cpu.toFixed(1)}%</div>
                  <div className="col-span-3">
                    <Progress value={process.cpu * 5} className="h-1.5" indicatorClassName="bg-blue-500" />
                  </div>
                  <div className="col-span-2 text-right text-sm font-medium text-zinc-300">{process.memory} MB</div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
