import React, { useState } from 'react';
import { motion } from 'framer-motion';

interface Command {
  id: string;
  label: string;
  icon: string;
}

const commands: Command[] = [
  { id: 'power_on', label: 'Power On', icon: '🔌' },
  { id: 'power_off', label: 'Power Off', icon: '⚡' },
  { id: 'volume_up', label: 'Vol +', icon: '🔊' },
  { id: 'volume_down', label: 'Vol -', icon: '🔉' },
  { id: 'channel_up', label: 'Ch +', icon: '📺' },
  { id: 'channel_down', label: 'Ch -', icon: '📺' },
  { id: 'mute', label: 'Mute', icon: '🔇' },
  { id: 'input', label: 'Input', icon: '🔄' },
];

const SpectrumHub: React.FC = () => {
  const [activeCommand, setActiveCommand] = useState<string | null>(null);

  const handleCommand = (commandId: string) => {
    setActiveCommand(commandId);
    // In real implementation, send command to Broadlink RM4
    console.log(`Sending command: ${commandId}`);

    // Reset after animation
    setTimeout(() => setActiveCommand(null), 1000);
  };

  return (
    <div className="grid grid-cols-4 gap-2 h-full">
      {commands.map((command) => (
        <motion.button
          key={command.id}
          className={`relative bg-black/30 border border-stark-cyan/30 rounded-lg p-2 text-stark-cyan hover:bg-stark-cyan/10 transition-colors flex flex-col items-center justify-center text-xs ${
            activeCommand === command.id ? 'bg-stark-cyan/20 border-stark-cyan' : ''
          }`}
          onClick={() => handleCommand(command.id)}
          whileTap={{ scale: 0.95 }}
        >
          <span className="text-lg mb-1">{command.icon}</span>
          <span>{command.label}</span>

          {/* Signal Burst Animation */}
          {activeCommand === command.id && (
            <motion.div
              className="absolute inset-0 rounded-lg border-2 border-stark-cyan"
              initial={{ scale: 1, opacity: 1 }}
              animate={{ scale: 1.5, opacity: 0 }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
            />
          )}
        </motion.button>
      ))}
    </div>
  );
};

export default SpectrumHub;
