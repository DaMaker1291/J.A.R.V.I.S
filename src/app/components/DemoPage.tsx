import { useState, useEffect, useRef } from "react";
import { Link } from "react-router";
import { Terminal, ArrowLeft, Send, Zap, CheckCircle2 } from "lucide-react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";

interface CommandOutput {
  command: string;
  output: string;
  timestamp: string;
  type: "success" | "error" | "info";
}

export function DemoPage() {
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
  const scrollRef = useRef<HTMLDivElement>(null);

  const exampleCommands = [
    { cmd: "app launch Safari", desc: "Launch Safari browser" },
    { cmd: "window arrange grid", desc: "Arrange windows in grid" },
    { cmd: "monitor cpu", desc: "Monitor CPU usage" },
    { cmd: "boost productivity", desc: "Activate productivity mode" },
    { cmd: "help", desc: "Show all commands" }
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
  security scan         - Perform security check

Type any command to try it out!`,
      type: "info"
    },
    "app launch Safari": {
      output: `✓ Launching Safari...
✓ Application started successfully
✓ Window positioned: Main Display
Process ID: 4521`,
      type: "success"
    },
    "app launch Chrome": {
      output: `✓ Launching Google Chrome...
✓ Application started successfully
✓ Window positioned: Main Display
Process ID: 4622`,
      type: "success"
    },
    "app list": {
      output: `Running Applications:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Safari          PID: 4521   RAM: 245 MB
Finder          PID: 1234   RAM: 128 MB
Terminal        PID: 3456   RAM: 89 MB
VS Code         PID: 5678   RAM: 512 MB
Slack           PID: 7890   RAM: 334 MB

Total: 5 applications running`,
      type: "info"
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
    "monitor memory": {
      output: `Memory Monitoring:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: 16.0 GB
Used: 11.2 GB (70%)
Free: 4.8 GB
Swap: 2.1 GB

Largest Consumers:
  VS Code         512 MB
  Slack           334 MB
  Safari          245 MB`,
      type: "info"
    },
    "calendar event 'Team Meeting' tomorrow 2pm": {
      output: `✓ Creating calendar event...
✓ Event: Team Meeting
✓ Date: February 17, 2026
✓ Time: 2:00 PM
✓ Duration: 1 hour
✓ Added to Calendar.app

Event created successfully!`,
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
    "security scan": {
      output: `🔒 Security Scan In Progress...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Checking for malware... Clear
✓ Verifying system integrity... OK
✓ Scanning network connections... Secure
✓ Reviewing file permissions... Correct
✓ Checking firewall status... Active

No security issues detected. System secure! ✓`,
      type: "success"
    },
    "process list": {
      output: `Running Processes:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PID    Name              CPU%   MEM
4521   Safari            12.4%  245 MB
5678   VS Code           8.7%   512 MB
7890   Slack             2.1%   334 MB
3456   Terminal          0.8%   89 MB
1234   Finder            0.5%   128 MB

Total: 247 processes`,
      type: "info"
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing) return;

    const command = input.trim();
    setIsProcessing(true);

    // Add command to history immediately
    setHistory(prev => [...prev, {
      command,
      output: "",
      timestamp: new Date().toLocaleTimeString(),
      type: "info"
    }]);

    setInput("");

    // Simulate processing delay
    setTimeout(() => {
      let response = commandResponses[command];
      
      if (!response) {
        // Check for partial matches
        if (command.toLowerCase().startsWith("app launch")) {
          const appName = command.substring(11).trim();
          response = {
            output: `✓ Launching ${appName}...\n✓ Application started successfully\nProcess ID: ${Math.floor(Math.random() * 10000)}`,
            type: "success"
          };
        } else if (command.toLowerCase().startsWith("monitor")) {
          response = {
            output: `Monitoring ${command.substring(8).trim()}...\nReal-time data stream active.`,
            type: "info"
          };
        } else if (command.toLowerCase().startsWith("calendar event")) {
          response = {
            output: `✓ Calendar event created successfully!\n✓ Event details: ${command.substring(15)}`,
            type: "success"
          };
        } else {
          response = {
            output: `Command not recognized: "${command}"\nType 'help' to see available commands.`,
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
  };

  const handleExampleClick = (command: string) => {
    setInput(command);
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history]);

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
                <span className="text-xs text-neutral-400">Interactive Demo</span>
              </div>
            </Link>
            <Link to="/">
              <Button variant="ghost" className="text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100 text-sm h-9">
                <ArrowLeft className="size-4 mr-2" strokeWidth={2} />
                Back
              </Button>
            </Link>
          </div>
        </div>
      </nav>

      <div className="pt-24 pb-12 px-8">
        <div className="max-w-[1400px] mx-auto">
          {/* Header */}
          <div className="mb-8">
            <Badge className="mb-4 bg-emerald-50 text-emerald-700 border-emerald-200 text-xs font-medium px-3 py-1">
              <CheckCircle2 className="size-3 mr-1" strokeWidth={2} />
              Live Interactive Demo
            </Badge>
            <h1 className="text-4xl font-semibold tracking-tight text-neutral-900 mb-3">Command Interface</h1>
            <p className="text-neutral-500 text-lg">
              Execute automation commands in real-time. Type 'help' for available commands.
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            {/* Terminal - Main Column */}
            <div className="lg:col-span-2">
              <Card className="bg-white border border-neutral-200/60 overflow-hidden">
                {/* Terminal Header */}
                <div className="bg-neutral-50 border-b border-neutral-200/60 px-6 py-3 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="flex gap-1.5">
                      <div className="size-3 rounded-full bg-neutral-300" />
                      <div className="size-3 rounded-full bg-neutral-300" />
                      <div className="size-3 rounded-full bg-neutral-300" />
                    </div>
                    <span className="text-neutral-500 text-xs font-medium font-mono">jason-cli</span>
                  </div>
                  <Badge className="bg-emerald-50 text-emerald-700 border-emerald-200 text-xs">
                    <div className="size-1.5 rounded-full bg-emerald-500 mr-1.5" />
                    Active
                  </Badge>
                </div>

                {/* Terminal Content */}
                <div 
                  ref={scrollRef}
                  className="p-6 h-[600px] overflow-y-auto font-mono text-sm bg-neutral-50/30"
                >
                  {history.map((item, index) => (
                    <div key={index} className="mb-4">
                      {item.command && (
                        <div className="flex items-start gap-2 mb-2">
                          <span className="text-[#002FA7]">$</span>
                          <span className="text-neutral-900">{item.command}</span>
                        </div>
                      )}
                      {item.output && (
                        <div className={`ml-4 whitespace-pre-wrap ${
                          item.type === "success" ? "text-emerald-700" : 
                          item.type === "error" ? "text-red-600" : 
                          "text-neutral-600"
                        }`}>
                          {item.output}
                        </div>
                      )}
                      {index < history.length - 1 && <div className="my-4 border-t border-neutral-200/40" />}
                    </div>
                  ))}

                  {isProcessing && (
                    <div className="flex items-center gap-2 text-neutral-500">
                      <div className="size-1.5 bg-[#002FA7] rounded-full animate-pulse" />
                      <span>Processing...</span>
                    </div>
                  )}
                </div>

                {/* Input Area */}
                <div className="border-t border-neutral-200/60 bg-white p-4">
                  <form onSubmit={handleSubmit} className="flex items-center gap-3">
                    <span className="text-[#002FA7] font-mono">$</span>
                    <Input
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder="Type a command..."
                      className="flex-1 bg-neutral-50 border-neutral-200 text-neutral-900 font-mono text-sm focus:border-[#002FA7] focus-visible:ring-[#002FA7]"
                      disabled={isProcessing}
                      autoFocus
                    />
                    <Button 
                      type="submit" 
                      disabled={!input.trim() || isProcessing}
                      className="bg-[#002FA7] hover:bg-[#001f75] h-9"
                    >
                      <Send className="size-4" strokeWidth={2} />
                    </Button>
                  </form>
                </div>
              </Card>
            </div>

            {/* Sidebar - Commands */}
            <div className="space-y-6">
              <div>
                <h3 className="text-sm font-semibold text-neutral-900 mb-4">Quick Commands</h3>
                <div className="space-y-2">
                  {exampleCommands.map((item) => (
                    <button
                      key={item.cmd}
                      onClick={() => handleExampleClick(item.cmd)}
                      className="w-full p-3 bg-white hover:bg-neutral-50 border border-neutral-200/60 rounded-lg transition-colors text-left group"
                    >
                      <div className="flex items-start gap-2 mb-1">
                        <Terminal className="size-3.5 text-neutral-400 mt-0.5 group-hover:text-[#002FA7]" strokeWidth={2} />
                        <span className="text-xs font-mono text-neutral-900">{item.cmd}</span>
                      </div>
                      <p className="text-xs text-neutral-500 ml-5">{item.desc}</p>
                    </button>
                  ))}
                </div>
              </div>

              <div className="p-4 bg-neutral-50 border border-neutral-200/60 rounded-lg">
                <div className="flex items-start gap-3 mb-3">
                  <div className="size-8 bg-white rounded-lg flex items-center justify-center flex-shrink-0 border border-neutral-200/60">
                    <Zap className="size-4 text-[#002FA7]" strokeWidth={2} />
                  </div>
                  <div>
                    <h4 className="text-sm font-semibold text-neutral-900 mb-1">Instant Execution</h4>
                    <p className="text-xs text-neutral-500">
                      All commands execute deterministically without external APIs.
                    </p>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-neutral-50 border border-neutral-200/60 rounded-lg">
                <div className="text-xs font-medium text-neutral-500 mb-3">COMMAND STATS</div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-neutral-600">Executed Today</span>
                    <span className="text-sm font-semibold text-neutral-900">1,247</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-neutral-600">Success Rate</span>
                    <span className="text-sm font-semibold text-emerald-700">99.8%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-neutral-600">Avg Response</span>
                    <span className="text-sm font-semibold text-neutral-900">&lt;100ms</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}