import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const BioLockStatus: React.FC = () => {
  const [matchPercentage, setMatchPercentage] = useState(95); // Mock match percentage

  // Simulate match changes (in real implementation, from voice verification)
  useEffect(() => {
    const interval = setInterval(() => {
      setMatchPercentage(Math.random() * 40 + 60); // 60-100%
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const isAuthorized = matchPercentage >= 80;
  const progressWidth = (matchPercentage / 100) * 100;

  return (
    <div className="h-full flex flex-col justify-center">
      <div className="flex justify-between text-xs mb-2">
        <span>Bio-Match</span>
        <span>{matchPercentage.toFixed(1)}%</span>
      </div>
      <div className="w-full bg-black/50 rounded-full h-3 overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${isAuthorized ? 'bg-stark-cyan' : 'bg-alert-red'}`}
          initial={{ width: 0 }}
          animate={{ width: `${progressWidth}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>
      <div className={`text-xs mt-1 text-center ${isAuthorized ? 'text-stark-cyan' : 'text-alert-red'}`}>
        {isAuthorized ? 'Authorized' : 'Guest Detected'}
      </div>
    </div>
  );
};

export default BioLockStatus;
