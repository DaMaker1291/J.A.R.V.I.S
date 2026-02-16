import React from 'react';
import { Activity, Shield, Zap, Eye, Mic, Terminal } from 'lucide-react';
import SpatialMapping from './components/SpatialMapping';
import SpectrumHub from './components/SpectrumHub';
import LatencyMonitor from './components/LatencyMonitor';
import BioLockStatus from './components/BioLockStatus';
import GhostSweepLog from './components/GhostSweepLog';
import useSocket from './hooks/useSocket';

function App() {
  const { isConnected, status } = useSocket();

  return (
    <div className="h-screen w-screen bg-gradient-to-br from-black to-gray-900 text-stark-cyan font-mono overflow-hidden">
      {/* Header */}
      <header className="h-16 bg-black/20 backdrop-blur-md border-b border-stark-cyan/20 flex items-center px-6">
        <h1 className="text-2xl font-bold text-stark-cyan">J.A.S.O.N. Tactical OS Interface</h1>
        <div className="ml-auto flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full animate-pulse ${
              status.mood === 'red' ? 'bg-alert-red' :
              status.mood === 'orange' ? 'bg-neutral-orange' :
              'bg-stark-cyan'
            }`}></div>
            <span className="text-sm">{status.status.toUpperCase()}</span>
          </div>
          <div className="text-sm">
            Threat: {status.threat_level.toFixed(1)}
          </div>
          <div className="text-sm">
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-4rem)]">
        {/* Sidebar */}
        <aside className="w-64 bg-black/10 backdrop-blur-md border-r border-stark-cyan/20 p-4">
          <nav className="space-y-2">
            <div className="flex items-center space-x-3 p-3 bg-stark-cyan/10 rounded-lg">
              <Eye className="w-5 h-5 text-stark-cyan" />
              <span>Spatial Mapping</span>
            </div>
            <div className="flex items-center space-x-3 p-3 hover:bg-stark-cyan/10 rounded-lg">
              <Zap className="w-5 h-5 text-stark-cyan" />
              <span>Spectrum Hub</span>
            </div>
            <div className="flex items-center space-x-3 p-3 hover:bg-stark-cyan/10 rounded-lg">
              <Activity className="w-5 h-5 text-stark-cyan" />
              <span>Neural Monitor</span>
            </div>
            <div className="flex items-center space-x-3 p-3 hover:bg-stark-cyan/10 rounded-lg">
              <Shield className="w-5 h-5 text-stark-cyan" />
              <span>Bio-Lock Status</span>
            </div>
            <div className="flex items-center space-x-3 p-3 hover:bg-stark-cyan/10 rounded-lg">
              <Terminal className="w-5 h-5 text-stark-cyan" />
              <span>Ghost Sweep Log</span>
            </div>
            <div className="flex items-center space-x-3 p-3 hover:bg-stark-cyan/10 rounded-lg">
              <Mic className="w-5 h-5 text-stark-cyan" />
              <span>Voice Interface</span>
            </div>
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-6 overflow-hidden">
          <div className="h-full bg-black/5 backdrop-blur-md rounded-lg border border-stark-cyan/20 p-4">
            <h2 className="text-xl font-semibold mb-4">Mission Control</h2>
            <div className="grid grid-cols-2 gap-4 h-full">
              {/* Spatial Mapping Overlay */}
              <div className="bg-black/20 rounded-lg p-4 border border-stark-cyan/30">
                <h3 className="text-lg font-medium mb-2">Spatial Mapping Overlay</h3>
                <div className="h-64 bg-black/30 rounded overflow-hidden">
                  <SpatialMapping />
                </div>
              </div>

              {/* Other panels */}
              <div className="space-y-4">
                <div className="bg-black/20 rounded-lg p-4 border border-stark-cyan/30">
                  <h3 className="text-lg font-medium mb-2">Spectrum Command Hub</h3>
                  <div className="h-24">
                    <SpectrumHub />
                  </div>
                </div>
                <div className="bg-black/20 rounded-lg p-4 border border-stark-cyan/30">
                  <h3 className="text-lg font-medium mb-2">Neural Latency Monitor</h3>
                  <div className="h-24">
                    <LatencyMonitor />
                  </div>
                </div>
              </div>

              {/* Bottom row */}
              <div className="bg-black/20 rounded-lg p-4 border border-stark-cyan/30">
                <h3 className="text-lg font-medium mb-2">Bio-Lock Status</h3>
                <div className="h-24">
                  <BioLockStatus />
                </div>
              </div>
              <div className="bg-black/20 rounded-lg p-4 border border-stark-cyan/30">
                <h3 className="text-lg font-medium mb-2">Ghost Sweep Log</h3>
                <div className="h-24">
                  <GhostSweepLog />
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
