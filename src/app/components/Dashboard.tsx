import { useState, useEffect } from "react";
import { Link } from "react-router";
import { motion } from "motion/react";
import {
  Terminal,
  Cpu,
  Activity,
  Zap,
  Calendar,
  Layout,
  ArrowRight,
  TrendingUp,
  CheckCircle2,
  Clock,
  Play,
  AlertCircle,
} from "lucide-react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import { AreaChart, Area, ResponsiveContainer, LineChart, Line } from "recharts";

export function Dashboard() {
  const [cpuData, setCpuData] = useState([
    { value: 20 },
    { value: 25 },
    { value: 22 },
    { value: 28 },
    { value: 24 },
    { value: 26 },
    { value: 23 },
  ]);

  const [metrics, setMetrics] = useState({
    cpu: 0,
    memory: 0,
    activeProcesses: 0,
    automations: 0,
  });

  const [backendStatus, setBackendStatus] = useState('checking');

  // Check backend status on mount
  useEffect(() => {
    checkBackendStatus();
    fetchRealMetrics();
  }, []);

  const checkBackendStatus = async () => {
    const apiBase = (import.meta as any).env?.VITE_API_BASE_URL as string | undefined;
    if (!apiBase) {
      setBackendStatus('disconnected');
      return;
    }
    try {
      const response = await fetch(`${apiBase.replace(/\/$/, '')}/status`);
      if (response.ok) {
        setBackendStatus('connected');
      } else {
        setBackendStatus('error');
      }
    } catch (error) {
      setBackendStatus('disconnected');
    }
  };

  const executeCommand = async (command: string) => {
    const apiBase = (import.meta as any).env?.VITE_API_BASE_URL as string | undefined;
    if (!apiBase) {
      return { error: 'Backend is not configured' };
    }
    try {
      const response = await fetch(`${apiBase.replace(/\/$/, '')}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          command: command,
          require_clarification: false,
        }),
      });
      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Failed to execute command:', error);
      return { error: 'Failed to connect to backend' };
    }
  };

  // Fetch real metrics periodically
  useEffect(() => {
    const interval = setInterval(() => {
      fetchRealMetrics();
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const fetchRealMetrics = async () => {
    const apiBase = (import.meta as any).env?.VITE_API_BASE_URL as string | undefined;
    if (!apiBase) return;
    
    try {
      const response = await fetch(`${apiBase.replace(/\/$/, '')}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          command: 'monitor cpu',
        }),
      });
      const result = await response.json();
      if (result && result.result) {
        // Parse the CPU monitoring result
        const lines = result.result.split('\n');
        let cpuUsage = 0;
        let processes = 0;
        
        for (const line of lines) {
          if (line.startsWith('Usage:')) {
            const match = line.match(/Usage:\s*([\d.]+)/);
            if (match) cpuUsage = parseFloat(match[1]);
          }
          if (line.startsWith('Processes:')) {
            const match = line.match(/Processes:\s*(\d+)/);
            if (match) processes = parseInt(match[1]);
          }
        }
        
        // Update CPU data for chart
        setCpuData(prev => {
          const newData = [...prev.slice(1)];
          newData.push({ value: cpuUsage });
          return newData;
        });
        
        setMetrics(prev => ({
          ...prev,
          cpu: cpuUsage,
          activeProcesses: processes,
        }));
      }
    } catch (error) {
      console.error('Failed to fetch real metrics:', error);
    }
  };

  const recentAutomations = [
    {
      id: 1,
      name: "Productivity Boost",
      status: "completed",
      time: "2 min ago",
      apps: 4,
    },
    {
      id: 2,
      name: "Window Grid Arrangement",
      status: "completed",
      time: "8 min ago",
      apps: 5,
    },
    {
      id: 3,
      name: "System Cleanup",
      status: "running",
      time: "Running...",
      progress: 67,
    },
    {
      id: 4,
      name: "Calendar Sync",
      status: "scheduled",
      time: "In 15 min",
      apps: 2,
    },
  ];

  const quickActions = [
    {
      icon: Zap,
      label: "Boost Productivity",
      description: "Close distractions, launch work apps",
      color: "blue",
    },
    {
      icon: Layout,
      label: "Arrange Windows",
      description: "Grid layout for active apps",
      color: "purple",
    },
    {
      icon: Calendar,
      label: "Create Event",
      description: "Add to Calendar.app",
      color: "green",
    },
    {
      icon: Terminal,
      label: "Run Command",
      description: "Execute custom automation",
      color: "orange",
    },
  ];

  const systemStats = [
    {
      label: "CPU Usage",
      value: `${metrics.cpu.toFixed(1)}%`,
      data: cpuData,
      color: "blue",
      status: metrics.cpu > 80 ? "critical" : metrics.cpu > 60 ? "warning" : "optimal",
    },
    {
      label: "Memory",
      value: `${metrics.memory.toFixed(1)}%`,
      data: cpuData.map((d) => ({ value: Math.max(50, Math.min(85, d.value + 20)) })),
      color: "purple",
      status: metrics.memory > 80 ? "critical" : metrics.memory > 60 ? "warning" : "optimal",
    },
    {
      label: "Active Processes",
      value: metrics.activeProcesses,
      change: "+3",
      status: "optimal",
    },
    {
      label: "Automations Today",
      value: metrics.automations,
      change: "+2",
      status: "optimal",
    },
  ];

  const StatusIndicator = ({ status }: { status: string }) => {
    const colors = {
      optimal: "bg-emerald-500",
      warning: "bg-amber-500",
      critical: "bg-red-500",
      running: "bg-blue-500",
      completed: "bg-emerald-500",
      scheduled: "bg-zinc-500",
    };

    return (
      <span className="relative flex h-2 w-2">
        <span
          className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${colors[status as keyof typeof colors]}`}
        />
        <span
          className={`relative inline-flex rounded-full h-2 w-2 ${colors[status as keyof typeof colors]}`}
        />
      </span>
    );
  };

  return (
    <div className="p-8 lg:pt-8 pt-20 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-white mb-1">Dashboard</h1>
          <p className="text-sm text-zinc-400">Monitor and control your system automations</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 px-3 py-1">
            <StatusIndicator status="optimal" />
            <span className="ml-2">All Systems Operational</span>
          </Badge>
        </div>
      </div>

      {/* System Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {systemStats.map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05, type: "spring", stiffness: 380, damping: 30 }}
          >
            <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm hover:bg-white/[0.07] transition-colors">
              <div className="flex items-start justify-between mb-4">
                <div className="space-y-1">
                  <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">
                    {stat.label}
                  </p>
                  <div className="flex items-baseline gap-2">
                    <p className="text-2xl font-semibold tracking-tight text-white">{stat.value}</p>
                    {stat.change && (
                      <span className="text-xs font-medium text-emerald-400">{stat.change}</span>
                    )}
                  </div>
                </div>
                <StatusIndicator status={stat.status} />
              </div>
              {stat.data && (
                <div className="h-12 -mb-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={stat.data}>
                      <defs>
                        <linearGradient id={`gradient-${index}`} x1="0" y1="0" x2="0" y2="1">
                          <stop
                            offset="5%"
                            stopColor={stat.color === "blue" ? "#3b82f6" : "#a855f7"}
                            stopOpacity={0.3}
                          />
                          <stop
                            offset="95%"
                            stopColor={stat.color === "blue" ? "#3b82f6" : "#a855f7"}
                            stopOpacity={0}
                          />
                        </linearGradient>
                      </defs>
                      <Area
                        type="monotone"
                        dataKey="value"
                        stroke={stat.color === "blue" ? "#3b82f6" : "#a855f7"}
                        strokeWidth={2}
                        fill={`url(#gradient-${index})`}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              )}
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, type: "spring", stiffness: 380, damping: 30 }}
          className="lg:col-span-2"
        >
          <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-lg font-semibold text-white">Quick Actions</h3>
                <p className="text-sm text-zinc-400 mt-1">Execute common automation workflows</p>
              </div>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              {quickActions.map((action, index) => {
                const Icon = action.icon;
                const handleClick = async () => {
                  let command = '';
                  switch (action.label) {
                    case 'Boost Productivity':
                      command = 'boost productivity';
                      break;
                    case 'Arrange Windows':
                      command = 'arrange windows in grid';
                      break;
                    case 'Create Event':
                      command = 'create calendar event';
                      break;
                    case 'Run Command':
                      command = 'show system status';
                      break;
                    default:
                      command = action.label.toLowerCase();
                  }
                  const result = await executeCommand(command);
                  console.log('Command result:', result);
                  // You could add toast notifications here to show the result
                };
                return (
                  <motion.button
                    key={action.label}
                    onClick={handleClick}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="group flex items-start gap-4 p-4 rounded-lg border border-white/5 bg-white/5 hover:bg-white/10 transition-all text-left"
                  >
                    <div className="size-10 rounded-lg bg-blue-600/20 flex items-center justify-center flex-shrink-0 group-hover:bg-blue-600/30 transition-colors">
                      <Icon className="size-5 text-blue-400" strokeWidth={2} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="text-sm font-medium text-white mb-1">{action.label}</h4>
                      <p className="text-xs text-zinc-400">{action.description}</p>
                    </div>
                    <ArrowRight className="size-4 text-zinc-500 group-hover:text-zinc-300 transition-colors flex-shrink-0 mt-1" />
                  </motion.button>
                );
              })}
            </div>
          </Card>
        </motion.div>

        {/* Recent Activity */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25, type: "spring", stiffness: 380, damping: 30 }}
        >
          <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-white">Recent Activity</h3>
              <Link to="/monitoring">
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-zinc-400 hover:text-white h-8 text-xs"
                >
                  View All
                  <ArrowRight className="ml-1 size-3" />
                </Button>
              </Link>
            </div>
            <div className="space-y-3">
              {recentAutomations.map((automation) => (
                <motion.div
                  key={automation.id}
                  whileHover={{ x: 2 }}
                  className="flex items-start gap-3 p-3 rounded-lg hover:bg-white/5 transition-colors cursor-pointer"
                >
                  <div className="mt-0.5">
                    {automation.status === "completed" && (
                      <CheckCircle2 className="size-4 text-emerald-400" strokeWidth={2} />
                    )}
                    {automation.status === "running" && (
                      <div className="size-4 flex items-center justify-center">
                        <StatusIndicator status="running" />
                      </div>
                    )}
                    {automation.status === "scheduled" && (
                      <Clock className="size-4 text-zinc-400" strokeWidth={2} />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white mb-1">{automation.name}</p>
                    <div className="flex items-center gap-2">
                      <p className="text-xs text-zinc-400">{automation.time}</p>
                      {automation.apps && (
                        <>
                          <span className="text-zinc-600">•</span>
                          <p className="text-xs text-zinc-500">{automation.apps} apps</p>
                        </>
                      )}
                    </div>
                    {automation.progress !== undefined && (
                      <Progress
                        value={automation.progress}
                        className="h-1 mt-2"
                        indicatorClassName="bg-blue-500"
                      />
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          </Card>
        </motion.div>
      </div>

      {/* Performance Insights */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, type: "spring", stiffness: 380, damping: 30 }}
      >
        <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="size-10 rounded-lg bg-purple-600/20 flex items-center justify-center">
                <TrendingUp className="size-5 text-purple-400" strokeWidth={2} />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">Performance Insights</h3>
                <p className="text-sm text-zinc-400">System efficiency and optimization opportunities</p>
              </div>
            </div>
            <Link to="/monitoring">
              <Button variant="ghost" className="text-zinc-400 hover:text-white">
                View Details
                <ArrowRight className="ml-2 size-4" />
              </Button>
            </Link>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-zinc-400">Commands Executed Today</span>
                <span className="text-sm font-semibold text-white">1,247</span>
              </div>
              <Progress value={78} className="h-1.5" indicatorClassName="bg-blue-500" />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-zinc-400">Success Rate</span>
                <span className="text-sm font-semibold text-emerald-400">99.8%</span>
              </div>
              <Progress value={99.8} className="h-1.5" indicatorClassName="bg-emerald-500" />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-zinc-400">Avg Response Time</span>
                <span className="text-sm font-semibold text-white">&lt;100ms</span>
              </div>
              <Progress value={95} className="h-1.5" indicatorClassName="bg-purple-500" />
            </div>
          </div>
        </Card>
      </motion.div>
    </div>
  );
}
