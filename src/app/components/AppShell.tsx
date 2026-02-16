import { useState } from "react";
import { Link, useLocation } from "react-router";
import { motion, AnimatePresence } from "motion/react";
import {
  Terminal,
  Activity,
  Play,
  Menu,
  X,
  Search,
  Command,
  ChevronDown,
  Zap,
  Layout,
  Calendar,
  Settings,
  FileText,
  Github,
} from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "./ui/collapsible";
import { Sheet, SheetContent, SheetTrigger } from "./ui/sheet";

interface AppShellProps {
  children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const location = useLocation();
  const [automationOpen, setAutomationOpen] = useState(true);
  const [mobileOpen, setMobileOpen] = useState(false);

  const navigation = [
    { name: "Dashboard", href: "/", icon: Terminal },
    { name: "Demo", href: "/demo", icon: Play },
    { name: "Monitor", href: "/monitoring", icon: Activity },
  ];

  const automationActions = [
    { name: "Productivity Mode", icon: Zap, shortcut: "⌘B" },
    { name: "Arrange Windows", icon: Layout, shortcut: "⌘W" },
    { name: "Create Event", icon: Calendar, shortcut: "⌘E" },
  ];

  const isActive = (path: string) => location.pathname === path;

  const Sidebar = () => (
    <div className="flex h-full flex-col gap-y-6">
      {/* Logo */}
      <div className="flex h-14 items-center gap-3 px-4">
        <div className="size-8 bg-blue-600 rounded flex items-center justify-center">
          <Terminal className="size-4 text-white" strokeWidth={2.5} />
        </div>
        <div className="flex-1">
          <h1 className="text-sm font-semibold text-white">J.A.S.O.N.</h1>
          <p className="text-xs text-zinc-500">v2.1.0</p>
        </div>
      </div>

      {/* Search */}
      <div className="px-4">
        <button className="w-full group flex items-center gap-3 rounded-lg border border-white/5 bg-white/5 px-3 py-2 text-sm text-zinc-400 hover:bg-white/10 transition-colors">
          <Search className="size-4" strokeWidth={2} />
          <span className="flex-1 text-left">Search...</span>
          <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border border-white/10 bg-white/5 px-1.5 font-mono text-[10px] font-medium">
            <Command className="size-2.5" />K
          </kbd>
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 space-y-1">
        {navigation.map((item) => {
          const Icon = item.icon;
          const active = isActive(item.href);
          return (
            <Link key={item.name} to={item.href}>
              <motion.div
                whileHover={{ x: 2 }}
                whileTap={{ scale: 0.98 }}
                className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                  active
                    ? "bg-white/10 text-white"
                    : "text-zinc-400 hover:bg-white/5 hover:text-zinc-200"
                }`}
              >
                <Icon className="size-4" strokeWidth={2} />
                <span>{item.name}</span>
              </motion.div>
            </Link>
          );
        })}

        {/* Automation Actions */}
        <Collapsible open={automationOpen} onOpenChange={setAutomationOpen} className="mt-6">
          <CollapsibleTrigger className="flex w-full items-center gap-2 px-3 py-2 text-xs font-semibold uppercase tracking-wider text-zinc-500 hover:text-zinc-400 transition-colors">
            <span className="flex-1 text-left">Automation</span>
            <ChevronDown
              className={`size-3 transition-transform ${automationOpen ? "rotate-180" : ""}`}
            />
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-1 pt-2">
            {automationActions.map((action) => {
              const Icon = action.icon;
              return (
                <motion.button
                  key={action.name}
                  whileHover={{ x: 2 }}
                  whileTap={{ scale: 0.98 }}
                  className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-zinc-400 hover:bg-white/5 hover:text-zinc-200 transition-colors"
                >
                  <Icon className="size-4" strokeWidth={2} />
                  <span className="flex-1 text-left">{action.name}</span>
                  <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border border-white/10 bg-white/5 px-1.5 font-mono text-[10px] font-medium text-white/50">
                    {action.shortcut}
                  </kbd>
                </motion.button>
              );
            })}
          </CollapsibleContent>
        </Collapsible>
      </nav>

      {/* Footer */}
      <div className="border-t border-white/5 px-4 py-4 space-y-1">
        <motion.button
          whileHover={{ x: 2 }}
          whileTap={{ scale: 0.98 }}
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-zinc-400 hover:bg-white/5 hover:text-zinc-200 transition-colors"
        >
          <Settings className="size-4" strokeWidth={2} />
          <span>Settings</span>
        </motion.button>
        <motion.button
          whileHover={{ x: 2 }}
          whileTap={{ scale: 0.98 }}
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-zinc-400 hover:bg-white/5 hover:text-zinc-200 transition-colors"
        >
          <FileText className="size-4" strokeWidth={2} />
          <span>Docs</span>
        </motion.button>
        <motion.a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          whileHover={{ x: 2 }}
          whileTap={{ scale: 0.98 }}
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-zinc-400 hover:bg-white/5 hover:text-zinc-200 transition-colors"
        >
          <Github className="size-4" strokeWidth={2} />
          <span>GitHub</span>
        </motion.a>
      </div>
    </div>
  );

  return (
    <div className="flex h-screen bg-zinc-950 text-white overflow-hidden">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex lg:w-64 lg:flex-col border-r border-white/5 bg-zinc-950/50 backdrop-blur-md">
        <Sidebar />
      </aside>

      {/* Mobile Header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-50 flex items-center justify-between border-b border-white/5 bg-zinc-950/80 backdrop-blur-xl px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="size-8 bg-blue-600 rounded flex items-center justify-center">
            <Terminal className="size-4 text-white" strokeWidth={2.5} />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-white">J.A.S.O.N.</h1>
          </div>
        </div>
        <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon" className="text-zinc-400">
              <Menu className="size-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="w-64 bg-zinc-950 border-white/5 p-0">
            <Sidebar />
          </SheetContent>
        </Sheet>
      </div>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{
              type: "spring",
              stiffness: 380,
              damping: 30,
            }}
            className="min-h-full"
          >
            {children}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
