import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const VoiceInterface: React.FC = () => {
  const [bars, setBars] = useState<number[]>(Array(20).fill(0));

  useEffect(() => {
    const interval = setInterval(() => {
      setBars(prev => prev.map(() => Math.random() * 100));
    }, 100);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-full flex items-end justify-center space-x-1">
      {bars.map((height, index) => (
        <motion.div
          key={index}
          className="w-1 bg-stark-cyan rounded-t"
          initial={{ height: 0 }}
          animate={{ height: `${height}%` }}
          transition={{ duration: 0.1 }}
        />
      ))}
    </div>
  );
};

export default VoiceInterface;
