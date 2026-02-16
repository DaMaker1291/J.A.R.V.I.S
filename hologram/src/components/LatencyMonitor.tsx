import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const LatencyMonitor: React.FC = () => {
  const [latency, setLatency] = useState(250); // Mock latency in ms

  // Simulate latency changes
  useEffect(() => {
    const interval = setInterval(() => {
      setLatency(Math.random() * 400 + 100); // 100-500ms
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const isHighLatency = latency > 300;
  const progressWidth = Math.min((latency / 500) * 100, 100);

  return (
    <div className="h-full flex flex-col justify-center">
      <div className="flex justify-between text-xs mb-2">
        <span>Latency</span>
        <span>{latency.toFixed(0)}ms</span>
      </div>
      <div className="w-full bg-black/50 rounded-full h-3 overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${isHighLatency ? 'bg-neutral-orange' : 'bg-stark-cyan'}`}
          initial={{ width: 0 }}
          animate={{ width: `${progressWidth}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>
      <div className="text-xs mt-1 text-center">
        {isHighLatency ? 'Local Fallback Active' : 'Neural Online'}
      </div>
    </div>
  );
};

export default LatencyMonitor;
