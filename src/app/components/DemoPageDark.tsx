import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Terminal, Send, Zap, CheckCircle2, AlertCircle, Info } from "lucide-react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";
import { Skeleton } from "./ui/skeleton";

interface CommandOutput {
  command: string;
  output: string;
  timestamp: string;
  type: "success" | "error" | "info";
}

export function DemoPageDark() {
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<CommandOutput[]>([
    {
      command: "system",
      output: "J.A.S.O.N. Desktop Automation Platform v2.1.0\nReady for commands. Type 'help' for available commands.",
      timestamp: new Date().toLocaleTimeString(),
      type: "info"
    }
  ]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [backendAvailable, setBackendAvailable] = useState(true); // Assume available initially
  const scrollRef = useRef<HTMLDivElement>(null);

  const exampleCommands = [
    { cmd: "help", desc: "Show all commands", icon: Info },
    { cmd: "app launch Safari", desc: "Launch Safari browser", icon: Terminal },
    { cmd: "boost productivity", desc: "Activate productivity mode", icon: Zap },
    { cmd: "window arrange grid", desc: "Arrange windows in grid", icon: Terminal },
    { cmd: "monitor cpu", desc: "Monitor CPU usage", icon: Terminal },
  ];

  const commandResponses: Record<string, { output: string; type: "success" | "error" | "info" }> = {
    help: {
      output: `Available Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🖥️  Desktop Integration
  app launch <name>     - Launch application
  app quit <name>       - Quit application
  app list              - List running apps

📅 Calendar Management
  calendar event <details> - Create calendar event
  calendar list         - Show upcoming events

🪟 Window Management
  window arrange <layout> - Arrange windows (grid/left/right)
  window focus <app>    - Focus specific application

📊 System Monitoring
  monitor <resource>    - Monitor CPU/memory/disk/network
  process list          - List running processes
  process kill <id>     - Terminate process

⚡ Automation Workflows
  boost productivity    - Activate productivity mode
  system cleanup        - Run maintenance tasks
  security scan         - Perform security check`,
      type: "info"
    },
    "app launch Safari": {
      output: `✓ Launching Safari...
✓ Application started successfully
✓ Window positioned: Main Display
Process ID: 4521`,
      type: "success"
    },
    "boost productivity": {
      output: `⚡ Activating Productivity Boost Mode...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Closing distractions (Slack, Messages)
✓ Launching work apps (VS Code, Terminal, Safari)
✓ Arranging windows in optimal layout
✓ Enabling Do Not Disturb
✓ Setting focus timer for 25 minutes

Productivity mode activated! 🚀`,
      type: "success"
    },
    "window arrange grid": {
      output: `✓ Arranging windows in grid layout...
✓ Safari → Top Left Quadrant
✓ VS Code → Top Right Quadrant
✓ Terminal → Bottom Left Quadrant
✓ Slack → Bottom Right Quadrant

Window arrangement complete!`,
      type: "success"
    },
    "monitor cpu": {
      output: `CPU Monitoring:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage: 24.3%
Cores: 8 (M2 Pro)
Temperature: 42°C
Processes: 247

Top CPU Consumers:
  Safari          12.4%
  VS Code         8.7%
  Slack           2.1%`,
      type: "info"
    },
    "system cleanup": {
      output: `🧹 Running System Maintenance...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Clearing cache files... (842 MB freed)
✓ Removing temp files... (234 MB freed)
✓ Optimizing disk space...
✓ Repairing permissions...
✓ Updating system databases...

Cleanup complete! Freed 1.08 GB of space.`,
      type: "success"
    },
  };

  useEffect(() => {
    // Check backend availability on component mount
    const checkBackend = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/status', {
          timeout: 5000
        });
        setBackendAvailable(response.ok);
      } catch (error) {
        setBackendAvailable(false);
      }
    };
    checkBackend();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing) return;

    const command = input.trim();
    setIsProcessing(true);

    setHistory(prev => [...prev, {
      command,
      output: "",
      timestamp: new Date().toLocaleTimeString(),
      type: "info"
    }]);

    setInput("");

    // If backend is not available, use demo mode
    if (!backendAvailable) {
      setTimeout(() => {
        let response = commandResponses[command];
        
        if (!response) {
          if (command.toLowerCase().startsWith("app launch")) {
            const appName = command.substring(11).trim();
            response = {
              output: `✓ Launching ${appName}...\n✓ Application started successfully\nProcess ID: ${Math.floor(Math.random() * 10000)}`,
              type: "success"
            };
          } else {
            response = {
              output: `Command not recognized: "${command}"\nType 'help' to see available commands.\n\nNote: Running in demo mode. For full functionality, run the app locally.`,
              type: "error"
            };
          }
        }

        setHistory(prev => {
          const newHistory = [...prev];
          newHistory[newHistory.length - 1] = {
            command,
            output: response.output,
            timestamp: new Date().toLocaleTimeString(),
            type: response.type
          };
          return newHistory;
        });

        setIsProcessing(false);
      }, 800);
      return;
    }

    // Backend is available, use real API
    try {
      const response = await fetch('http://127.0.0.1:8000/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          command: command,
          require_clarification: false,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      setHistory(prev => {
        const newHistory = [...prev];
        newHistory[newHistory.length - 1] = {
          command,
          output: typeof data.result === 'string' ? data.result : JSON.stringify(data.result, null, 2),
          timestamp: new Date().toLocaleTimeString(),
          type: data.status === "success" ? "success" : "info"
        };
        return newHistory;
      });
    } catch (error) {
      console.error('Failed to execute command:', error);
      setHistory(prev => {
        const newHistory = [...prev];
        newHistory[newHistory.length - 1] = {
          command,
          output: `Error: Failed to connect to J.A.S.O.N. Neural Core. Ensure backend is running.\n${error instanceof Error ? error.message : String(error)}`,
          timestamp: new Date().toLocaleTimeString(),
          type: "error"
        };
        return newHistory;
      });
    } finally {
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history]);

  return (
    <div className="p-8 lg:pt-8 pt-20 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-white mb-1">Interactive Demo</h1>
          <p className="text-sm text-zinc-400">Execute automation commands in real-time</p>
        </div>
        <Badge className={`px-3 py-1 ${backendAvailable ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-amber-500/10 text-amber-400 border-amber-500/20'}`}>
          <span className="relative flex h-2 w-2 mr-2">
            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${backendAvailable ? 'bg-emerald-500' : 'bg-amber-500'}`} />
            <span className={`relative inline-flex rounded-full h-2 w-2 ${backendAvailable ? 'bg-emerald-500' : 'bg-amber-500'}`} />
          </span>
          {backendAvailable ? 'Live Demo' : 'Demo Mode'}
        </Badge>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Terminal - Main Column */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ type: "spring", stiffness: 380, damping: 30 }}
          className="lg:col-span-2"
        >
          <Card className="bg-white/5 border-white/5 overflow-hidden backdrop-blur-sm">
            {/* Terminal Header */}
            <div className="bg-white/5 border-b border-white/5 px-6 py-3 flex items-center justify-between backdrop-blur-sm">
              <div className="flex items-center gap-3">
                <div className="flex gap-1.5">
                  <div className="size-3 rounded-full bg-red-500/80" />
                  <div className="size-3 rounded-full bg-amber-500/80" />
                  <div className="size-3 rounded-full bg-emerald-500/80" />
                </div>
                <span className="text-zinc-400 text-xs font-medium font-mono">jason-cli</span>
              </div>
              <Badge className="bg-blue-500/10 text-blue-400 border-blue-500/20 text-xs">
                <Terminal className="size-3 mr-1" />
                Active Session
              </Badge>
            </div>

            {/* Terminal Content */}
            <div 
              ref={scrollRef}
              className="p-6 h-[600px] overflow-y-auto font-mono text-sm bg-zinc-950/50"
            >
              <AnimatePresence mode="popLayout">
                {history.map((item, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ type: "spring", stiffness: 500, damping: 40 }}
                    className="mb-4"
                  >
                    {item.command && (
                      <div className="flex items-start gap-2 mb-2">
                        <span className="text-blue-400">$</span>
                        <span className="text-white">{item.command}</span>
                      </div>
                    )}
                    {item.output && (
                      <div className={`ml-4 whitespace-pre-wrap ${
                        item.type === "success" ? "text-emerald-400" : 
                        item.type === "error" ? "text-red-400" : 
                        "text-zinc-300"
                      }`}>
                        {item.output}
                      </div>
                    )}
                    {index < history.length - 1 && <div className="my-4 border-t border-white/5" />}
                  </motion.div>
                ))}
              </AnimatePresence>

              {isProcessing && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-center gap-2 text-zinc-400"
                >
                  <div className="flex gap-1">
                    <motion.div
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 1.5, repeat: Infinity, delay: 0 }}
                      className="size-1.5 bg-blue-400 rounded-full"
                    />
                    <motion.div
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 1.5, repeat: Infinity, delay: 0.3 }}
                      className="size-1.5 bg-blue-400 rounded-full"
                    />
                    <motion.div
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 1.5, repeat: Infinity, delay: 0.6 }}
                      className="size-1.5 bg-blue-400 rounded-full"
                    />
                  </div>
                  <span>Processing...</span>
                </motion.div>
              )}
            </div>

            {/* Input Area */}
            <div className="border-t border-white/5 bg-white/5 p-4 backdrop-blur-sm">
              <form onSubmit={handleSubmit} className="flex items-center gap-3">
                <span className="text-blue-400 font-mono">$</span>
                <Input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Type a command..."
                  className="flex-1 bg-white/5 border-white/10 text-white font-mono text-sm focus:border-blue-500 focus-visible:ring-blue-500/20 placeholder:text-zinc-500"
                  disabled={isProcessing}
                  autoFocus
                />
                <motion.div whileTap={{ scale: 0.95 }}>
                  <Button 
                    type="submit" 
                    disabled={!input.trim() || isProcessing}
                    className="bg-blue-600 hover:bg-blue-700 h-9"
                  >
                    <Send className="size-4" strokeWidth={2} />
                  </Button>
                </motion.div>
              </form>
            </div>
          </Card>
        </motion.div>

        {/* Sidebar */}
        <div className="space-y-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, type: "spring", stiffness: 380, damping: 30 }}
          >
            <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm">
              <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                <Zap className="size-4 text-blue-400" strokeWidth={2} />
                Quick Commands
              </h3>
              <div className="space-y-2">
                {exampleCommands.map((item, index) => {
                  const Icon = item.icon;
                  return (
                    <motion.button
                      key={item.cmd}
                      whileHover={{ x: 2 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => setInput(item.cmd)}
                      className="w-full p-3 bg-white/5 hover:bg-white/10 border border-white/5 rounded-lg transition-all text-left group"
                    >
                      <div className="flex items-start gap-2 mb-1">
                        <Icon className="size-3.5 text-zinc-400 mt-0.5 group-hover:text-blue-400 transition-colors" strokeWidth={2} />
                        <span className="text-xs font-mono text-white">{item.cmd}</span>
                      </div>
                      <p className="text-xs text-zinc-500 ml-5">{item.desc}</p>
                    </motion.button>
                  );
                })}
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15, type: "spring", stiffness: 380, damping: 30 }}
          >
            <Card className="bg-white/5 border-white/5 p-6 backdrop-blur-sm">
              <h3 className="text-sm font-semibold text-white mb-4">Stats</h3>
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-zinc-400">Commands Today</span>
                    <span className="text-sm font-semibold text-white">1,247</span>
                  </div>
                  <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: "78%" }}
                      transition={{ duration: 1, ease: "easeOut" }}
                      className="h-full bg-blue-500 rounded-full"
                    />
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-zinc-400">Success Rate</span>
                    <span className="text-sm font-semibold text-emerald-400">99.8%</span>
                  </div>
                  <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: "99.8%" }}
                      transition={{ duration: 1, delay: 0.2, ease: "easeOut" }}
                      className="h-full bg-emerald-500 rounded-full"
                    />
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-zinc-400">Avg Response</span>
                    <span className="text-sm font-semibold text-white">&lt;100ms</span>
                  </div>
                  <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: "95%" }}
                      transition={{ duration: 1, delay: 0.4, ease: "easeOut" }}
                      className="h-full bg-purple-500 rounded-full"
                    />
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 380, damping: 30 }}
          >
            <Card className="bg-gradient-to-br from-blue-600/20 to-purple-600/20 border-blue-500/20 p-6 backdrop-blur-sm">
              <div className="flex items-start gap-3">
                <div className="size-10 bg-blue-600/30 rounded-lg flex items-center justify-center flex-shrink-0">
                  <CheckCircle2 className="size-5 text-blue-400" strokeWidth={2} />
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-white mb-1">Instant Execution</h4>
                  <p className="text-xs text-zinc-300">
                    All commands execute deterministically without external APIs.
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
