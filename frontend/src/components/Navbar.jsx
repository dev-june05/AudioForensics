import React, { useEffect, useState } from 'react';
import { ShieldCheck, Activity, Cpu, Sparkles, Database } from 'lucide-react';
import axios from 'axios';

export default function Navbar() {
  const [backendStatus, setBackendStatus] = useState(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await axios.get('http://127.0.0.1:8000/health');
        setBackendStatus(res.data);
      } catch (err) {
        setBackendStatus({ status: 'offline' });
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const isOnline = backendStatus?.status === 'ok';

  return (
    <header className="w-full border-b border-slate-800/80 bg-slate-950/60 backdrop-blur-xl sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 shadow-lg shadow-sky-500/20 text-white">
            <ShieldCheck className="w-6 h-6" />
            <span className="absolute -top-1 -right-1 flex h-3 w-3">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${isOnline ? 'bg-emerald-400' : 'bg-rose-400'} opacity-75`}></span>
              <span className={`relative inline-flex rounded-full h-3 w-3 ${isOnline ? 'bg-emerald-500' : 'bg-rose-500'}`}></span>
            </span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-sky-300 bg-clip-text text-transparent">
                AudioForensics AI
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full font-semibold bg-sky-500/10 text-sky-400 border border-sky-500/30 tracking-wide uppercase">
                v2.4 SOTA
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Deepfake & Partial Speech Tampering Detection System
            </p>
          </div>
        </div>

        {/* System & Model Status Badges */}
        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-2 bg-slate-900/80 border border-slate-800 px-3 py-1.5 rounded-xl text-xs text-slate-300">
            <Database className="w-3.5 h-3.5 text-indigo-400" />
            <span className="text-slate-400">Benchmark:</span>
            <span className="font-medium text-slate-200">PartialSpoof LA</span>
            <span className="text-emerald-400 font-semibold bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
              EER: 5.49%
            </span>
          </div>

          <div className="flex items-center gap-2 bg-slate-900/90 border border-slate-800 px-3 py-1.5 rounded-xl text-xs">
            <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-400 shadow-sm shadow-emerald-400' : 'bg-rose-400'}`} />
            <span className="text-slate-300 font-medium">
              {isOnline ? 'Core Engine Online' : 'Engine Disconnected'}
            </span>
            {isOnline && (
              <span className="text-[11px] text-sky-400 border-l border-slate-700 pl-2">
                ResNet + BiLSTM + Transformers
              </span>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
