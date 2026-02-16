import { Link } from "react-router";
import { 
  Terminal, 
  Cpu, 
  Calendar, 
  Zap, 
  Shield, 
  Layout, 
  Activity,
  ArrowRight,
  Github,
  Play,
  CheckCircle2,
  TrendingUp
} from "lucide-react";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";

export function LandingPage() {
  const capabilities = [
    {
      icon: Terminal,
      title: "Desktop Integration",
      description: "Launch, quit, and control desktop applications with precision",
      metric: "247 processes",
      status: "Active"
    },
    {
      icon: Calendar,
      title: "Native Scheduling",
      description: "Create calendar events directly in Calendar.app, Fantastical, and BusyCal",
      metric: "12 events today",
      status: "Synced"
    },
    {
      icon: Activity,
      title: "System Monitoring",
      description: "Real-time CPU, memory, disk, and network monitoring",
      metric: "24.3% CPU",
      status: "Optimal"
    },
    {
      icon: Layout,
      title: "Window Management",
      description: "Arrange windows in professional grid layouts",
      metric: "5 apps arranged",
      status: "Active"
    },
    {
      icon: Zap,
      title: "Automation Workflows",
      description: "Productivity boost mode and system maintenance automation",
      metric: "3 workflows",
      status: "Ready"
    },
    {
      icon: Shield,
      title: "Zero-API Processing",
      description: "Deterministic command execution without external APIs",
      metric: "100% local",
      status: "Secure"
    }
  ];

  const automationActions = [
    { label: "Launch Development Environment", apps: 4, time: "2s" },
    { label: "Optimize System Performance", freed: "1.08 GB", time: "12s" },
    { label: "Arrange Windows Grid Layout", windows: 5, time: "1s" },
    { label: "Schedule Team Sync", when: "Tomorrow 2pm", time: "1s" }
  ];

  const systemStatus = [
    { label: "CPU Usage", value: "24.3%", status: "optimal" },
    { label: "Memory", value: "11.2/16 GB", status: "normal" },
    { label: "Active Processes", value: "247", status: "optimal" },
    { label: "Uptime", value: "12d 4h", status: "excellent" }
  ];

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-xl border-b border-neutral-200/60">
        <div className="max-w-[1400px] mx-auto px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="size-8 bg-[#002FA7] rounded flex items-center justify-center">
                <Terminal className="size-4 text-white" strokeWidth={2.5} />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold tracking-tight text-neutral-900">J.A.S.O.N.</span>
                <span className="text-xs text-neutral-400">v2.1.0</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Link to="/demo">
                <Button variant="ghost" className="text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100 text-sm h-9">
                  Demo
                </Button>
              </Link>
              <Link to="/monitoring">
                <Button variant="ghost" className="text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100 text-sm h-9">
                  Monitor
                </Button>
              </Link>
              <Button variant="ghost" className="text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100 text-sm h-9">
                <Github className="size-4" strokeWidth={2} />
              </Button>
              <Link to="/demo">
                <Button className="bg-[#002FA7] hover:bg-[#001f75] text-white text-sm h-9 px-4">
                  Get Started
                  <ArrowRight className="ml-2 size-4" strokeWidth={2} />
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-16 px-8">
        <div className="max-w-[1400px] mx-auto">
          <div className="max-w-3xl">
            <Badge className="mb-6 bg-neutral-100 text-neutral-700 border-neutral-200 hover:bg-neutral-100 text-xs font-medium px-3 py-1">
              Desktop Automation Platform
            </Badge>
            <h1 className="text-[56px] leading-[1.1] font-semibold tracking-tight text-neutral-900 mb-6">
              Professional-grade system control through intelligent automation
            </h1>
            <p className="text-lg text-neutral-500 mb-8 leading-relaxed max-w-2xl">
              J.A.S.O.N. executes deterministic commands without external APIs. Native macOS integration, real-time monitoring, and zero-dependency processing.
            </p>
            <div className="flex gap-3">
              <Link to="/demo">
                <Button className="bg-[#002FA7] hover:bg-[#001f75] text-white h-11 px-6">
                  Try Interactive Demo
                  <ArrowRight className="ml-2 size-4" strokeWidth={2} />
                </Button>
              </Link>
              <Link to="/monitoring">
                <Button variant="outline" className="border-neutral-300 text-neutral-700 hover:bg-neutral-50 h-11 px-6">
                  View System Stats
                </Button>
              </Link>
            </div>
          </div>

          {/* System Status Bar */}
          <div className="mt-16 grid grid-cols-4 gap-4">
            {systemStatus.map((stat) => (
              <div key={stat.label} className="p-4 bg-white border border-neutral-200/60 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-neutral-500 font-medium">{stat.label}</span>
                  <div className="size-1.5 rounded-full bg-emerald-500" />
                </div>
                <div className="text-2xl font-semibold tracking-tight text-neutral-900">{stat.value}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Bento Grid - Core Capabilities */}
      <section className="py-16 px-8 bg-neutral-50/50">
        <div className="max-w-[1400px] mx-auto">
          <div className="mb-12">
            <h2 className="text-3xl font-semibold tracking-tight text-neutral-900 mb-3">Core Capabilities</h2>
            <p className="text-neutral-500">Complete desktop automation through native system integration</p>
          </div>

          <div className="grid grid-cols-3 gap-4">
            {/* Large card - Desktop Integration */}
            <div className="col-span-2 row-span-2 p-8 bg-white/60 backdrop-blur-xl border border-neutral-200/60 rounded-xl">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="size-10 bg-neutral-100 rounded-lg flex items-center justify-center">
                      <Terminal className="size-5 text-neutral-700" strokeWidth={2} />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-neutral-900">Desktop Integration</h3>
                      <p className="text-sm text-neutral-500">Native macOS automation via AppleScript</p>
                    </div>
                  </div>
                </div>
                <Badge className="bg-emerald-50 text-emerald-700 border-emerald-200 text-xs">Active</Badge>
              </div>
              
              <div className="space-y-3">
                <div className="text-sm text-neutral-600 mb-6">
                  Launch, quit, and control desktop applications with precision. Real desktop app management and workflow automation without external dependencies.
                </div>
                
                <div className="space-y-2">
                  <div className="text-xs font-medium text-neutral-500 mb-3">SUGGESTED ACTIONS</div>
                  {automationActions.map((action) => (
                    <button
                      key={action.label}
                      className="w-full p-3 bg-neutral-50 hover:bg-neutral-100 border border-neutral-200/60 rounded-lg transition-colors text-left group"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <CheckCircle2 className="size-4 text-neutral-400 group-hover:text-[#002FA7]" strokeWidth={2} />
                          <span className="text-sm text-neutral-700 font-medium">{action.label}</span>
                        </div>
                        <div className="flex items-center gap-3">
                          {action.apps && <span className="text-xs text-neutral-400">{action.apps} apps</span>}
                          {action.freed && <span className="text-xs text-neutral-400">{action.freed}</span>}
                          {action.windows && <span className="text-xs text-neutral-400">{action.windows} windows</span>}
                          {action.when && <span className="text-xs text-neutral-400">{action.when}</span>}
                          <span className="text-xs text-neutral-400">{action.time}</span>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Other capability cards */}
            {capabilities.slice(1, 3).map((capability) => (
              <div key={capability.title} className="p-6 bg-white/60 backdrop-blur-xl border border-neutral-200/60 rounded-xl hover:border-neutral-300 transition-colors">
                <div className="flex items-start justify-between mb-4">
                  <div className="size-10 bg-neutral-100 rounded-lg flex items-center justify-center">
                    <capability.icon className="size-5 text-neutral-700" strokeWidth={2} />
                  </div>
                  <Badge className="bg-neutral-100 text-neutral-600 border-neutral-200 text-xs">{capability.status}</Badge>
                </div>
                <h3 className="text-base font-semibold text-neutral-900 mb-2">{capability.title}</h3>
                <p className="text-sm text-neutral-500 mb-4">{capability.description}</p>
                <div className="flex items-center justify-between pt-3 border-t border-neutral-200">
                  <span className="text-xs text-neutral-400">Current</span>
                  <span className="text-sm font-medium text-neutral-900">{capability.metric}</span>
                </div>
              </div>
            ))}

            {capabilities.slice(3).map((capability) => (
              <div key={capability.title} className="p-6 bg-white/60 backdrop-blur-xl border border-neutral-200/60 rounded-xl hover:border-neutral-300 transition-colors">
                <div className="flex items-start justify-between mb-4">
                  <div className="size-10 bg-neutral-100 rounded-lg flex items-center justify-center">
                    <capability.icon className="size-5 text-neutral-700" strokeWidth={2} />
                  </div>
                  <Badge className="bg-neutral-100 text-neutral-600 border-neutral-200 text-xs">{capability.status}</Badge>
                </div>
                <h3 className="text-base font-semibold text-neutral-900 mb-2">{capability.title}</h3>
                <p className="text-sm text-neutral-500 mb-4">{capability.description}</p>
                <div className="flex items-center justify-between pt-3 border-t border-neutral-200">
                  <span className="text-xs text-neutral-400">Status</span>
                  <span className="text-sm font-medium text-neutral-900">{capability.metric}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Technical Specifications */}
      <section className="py-16 px-8">
        <div className="max-w-[1400px] mx-auto">
          <div className="grid grid-cols-2 gap-8">
            <div>
              <h2 className="text-3xl font-semibold tracking-tight text-neutral-900 mb-3">Technical Foundation</h2>
              <p className="text-neutral-500 mb-8">Built for performance, reliability, and maintainability</p>
              
              <div className="space-y-4">
                <div className="flex items-start gap-4 p-4 border-l-2 border-[#002FA7]">
                  <div className="flex-1">
                    <h4 className="text-sm font-semibold text-neutral-900 mb-1">Python-Based Architecture</h4>
                    <p className="text-sm text-neutral-500">Robust performance with enterprise-grade reliability</p>
                  </div>
                </div>
                <div className="flex items-start gap-4 p-4 border-l-2 border-neutral-200">
                  <div className="flex-1">
                    <h4 className="text-sm font-semibold text-neutral-900 mb-1">Real-Time System Monitoring</h4>
                    <p className="text-sm text-neutral-500">Advanced tracking using psutil library</p>
                  </div>
                </div>
                <div className="flex items-start gap-4 p-4 border-l-2 border-neutral-200">
                  <div className="flex-1">
                    <h4 className="text-sm font-semibold text-neutral-900 mb-1">Native macOS Integration</h4>
                    <p className="text-sm text-neutral-500">Seamless automation via AppleScript</p>
                  </div>
                </div>
                <div className="flex items-start gap-4 p-4 border-l-2 border-neutral-200">
                  <div className="flex-1">
                    <h4 className="text-sm font-semibold text-neutral-900 mb-1">Zero External Dependencies</h4>
                    <p className="text-sm text-neutral-500">No cloud services or external APIs required</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="p-8 bg-neutral-50 rounded-xl border border-neutral-200/60">
              <div className="flex items-center gap-3 mb-6">
                <TrendingUp className="size-5 text-[#002FA7]" strokeWidth={2} />
                <h3 className="text-lg font-semibold text-neutral-900">Performance Metrics</h3>
              </div>
              
              <div className="space-y-6">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-neutral-600">Command Execution</span>
                    <span className="text-sm font-semibold text-neutral-900">&lt;100ms</span>
                  </div>
                  <div className="h-1.5 bg-neutral-200 rounded-full overflow-hidden">
                    <div className="h-full bg-[#002FA7] rounded-full" style={{ width: '95%' }} />
                  </div>
                </div>
                
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-neutral-600">System Response Time</span>
                    <span className="text-sm font-semibold text-neutral-900">&lt;50ms</span>
                  </div>
                  <div className="h-1.5 bg-neutral-200 rounded-full overflow-hidden">
                    <div className="h-full bg-[#002FA7] rounded-full" style={{ width: '98%' }} />
                  </div>
                </div>
                
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-neutral-600">Resource Overhead</span>
                    <span className="text-sm font-semibold text-neutral-900">~50MB RAM</span>
                  </div>
                  <div className="h-1.5 bg-neutral-200 rounded-full overflow-hidden">
                    <div className="h-full bg-emerald-500 rounded-full" style={{ width: '15%' }} />
                  </div>
                </div>

                <div className="pt-4 border-t border-neutral-200 grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-neutral-500 mb-1">Commands/Day</div>
                    <div className="text-2xl font-semibold text-neutral-900">1,247</div>
                  </div>
                  <div>
                    <div className="text-xs text-neutral-500 mb-1">Success Rate</div>
                    <div className="text-2xl font-semibold text-neutral-900">99.8%</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 px-8 bg-neutral-50/50">
        <div className="max-w-[1400px] mx-auto">
          <div className="p-12 bg-white border border-neutral-200/60 rounded-xl">
            <div className="max-w-2xl mx-auto text-center">
              <h2 className="text-3xl font-semibold tracking-tight text-neutral-900 mb-4">
                Experience J.A.S.O.N.
              </h2>
              <p className="text-neutral-500 mb-8">
                Try the interactive demo or explore real-time system monitoring
              </p>
              <div className="flex gap-3 justify-center">
                <Link to="/demo">
                  <Button className="bg-[#002FA7] hover:bg-[#001f75] text-white h-11 px-6">
                    <Terminal className="mr-2 size-4" strokeWidth={2} />
                    Interactive Demo
                  </Button>
                </Link>
                <Link to="/monitoring">
                  <Button variant="outline" className="border-neutral-300 text-neutral-700 hover:bg-neutral-50 h-11 px-6">
                    <Activity className="mr-2 size-4" strokeWidth={2} />
                    System Dashboard
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-8 border-t border-neutral-200">
        <div className="max-w-[1400px] mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="size-7 bg-[#002FA7] rounded flex items-center justify-center">
                <Terminal className="size-4 text-white" strokeWidth={2.5} />
              </div>
              <div>
                <span className="text-sm font-semibold text-neutral-900">J.A.S.O.N.</span>
                <p className="text-xs text-neutral-400">Professional desktop automation</p>
              </div>
            </div>
            <p className="text-xs text-neutral-400">
              © 2026 J.A.S.O.N. Platform. Python-powered native automation.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}