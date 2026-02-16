import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface LogEntry {
  id: number;
  message: string;
  timestamp: Date;
}

const mockLogs = [
  "File pruned: cache/old_config.json",
  "Network sniff: suspicious packet from 192.168.1.100",
  "Autonomous scan: room temperature 22°C",
  "Voice embedding verified: match 95.2%",
  "IR command sent: TV power toggle",
  "Memory optimized: freed 512MB",
  "Threat level updated: 0.1",
  "YOLO detection: object at [1.2, 0.5, -0.8]",
];

const GhostSweepLog: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);

  useEffect(() => {
    // Initial logs
    const initialLogs = mockLogs.slice(0, 5).map((msg, idx) => ({
      id: idx,
      message: msg,
      timestamp: new Date(Date.now() - (5 - idx) * 1000),
    }));
    setLogs(initialLogs);

    // Add new logs periodically
    const interval = setInterval(() => {
      const newLog = {
        id: Date.now(),
        message: mockLogs[Math.floor(Math.random() * mockLogs.length)],
        timestamp: new Date(),
      };
      setLogs(prev => [...prev.slice(-9), newLog]); // Keep last 10
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-full bg-black/50 rounded p-2 overflow-hidden font-mono text-xs">
      <AnimatePresence>
        {logs.map((log, index) => (
          <motion.div
            key={log.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.3 }}
            className="text-stark-cyan/80 mb-1"
            style={{
              animationDelay: `${index * 0.1}s`,
            }}
          >
            <span className="text-alert-red/60">[{log.timestamp.toLocaleTimeString()}]</span> {log.message}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
};

export default GhostSweepLog;
