import React from 'react';
import { Clock, ShieldAlert, ShieldCheck, HelpCircle, VolumeX } from 'lucide-react';

export default function TamperingTimeline({ windows, selectedIndex, onSelectWindow, totalDuration }) {
  if (!windows || windows.length === 0) return null;

  const formatSeconds = (sec) => {
    const mins = Math.floor(sec / 60);
    const secs = Math.floor(sec % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const realCount = windows.filter((w) => w.prediction === 'Real').length;
  const spoofedCount = windows.filter((w) => w.prediction === 'Spoofed').length;
  const silentCount = windows.filter((w) => w.is_silent).length;
  const uncertainCount = windows.filter((w) => w.prediction === 'Uncertain' && !w.is_silent).length;

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 backdrop-blur-md p-5 shadow-xl shadow-black/30 space-y-4">
      {/* Header and Summary stats */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
        <div>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Clock className="w-4 h-4 text-sky-400" />
            <span>Temporal Tampering Timeline</span>
            <span className="text-[11px] font-normal text-slate-400">
              (PartialSpoof Segmentation)
            </span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Click any 30-second window block to inspect its spectrogram and model consensus.
          </p>
        </div>

        {/* Legend / Counter Badges */}
        <div className="flex flex-wrap items-center gap-2 text-[11px]">
          <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            {realCount} Clean
          </span>
          <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20 font-medium">
            <span className="w-2 h-2 rounded-full bg-rose-400" />
            {spoofedCount} Tampered
          </span>
          {uncertainCount > 0 && (
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 font-medium">
              <span className="w-2 h-2 rounded-full bg-amber-400" />
              {uncertainCount} Mixed
            </span>
          )}
          {silentCount > 0 && (
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700 font-medium">
              <span className="w-2 h-2 rounded-full bg-slate-500" />
              {silentCount} Silent
            </span>
          )}
        </div>
      </div>

      {/* The Visual Timeline Bar */}
      <div className="space-y-2">
        <div className="w-full h-8 rounded-xl bg-slate-950 p-1 flex gap-1 border border-slate-800 overflow-x-auto">
          {windows.map((w, idx) => {
            const isSelected = idx === selectedIndex;
            let bgColor = 'bg-slate-700/50 hover:bg-slate-700';
            let borderColor = 'border-transparent';

            if (w.is_silent) {
              bgColor = 'bg-slate-800 hover:bg-slate-700';
            } else if (w.prediction === 'Spoofed') {
              bgColor = 'bg-rose-500/80 hover:bg-rose-500';
            } else if (w.prediction === 'Real') {
              bgColor = 'bg-emerald-500/80 hover:bg-emerald-500';
            } else {
              bgColor = 'bg-amber-500/80 hover:bg-amber-500';
            }

            if (isSelected) {
              borderColor = 'border-white ring-2 ring-white/40 ring-offset-1 ring-offset-slate-950';
            }

            return (
              <button
                key={idx}
                type="button"
                onClick={() => onSelectWindow(idx)}
                title={`Window ${idx + 1} (${formatSeconds(w.start_time_sec)} - ${formatSeconds(w.end_time_sec)}): ${
                  w.is_silent ? 'Silent' : `${w.prediction} (${(w.ai_probability * 100).toFixed(1)}% AI)`
                }`}
                className={`flex-1 min-w-[20px] h-full rounded-lg transition-all duration-150 border ${bgColor} ${borderColor} relative group`}
              >
                {/* Micro indicator if selected */}
                {isSelected && (
                  <span className="absolute -top-1 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-white shadow-sm" />
                )}
              </button>
            );
          })}
        </div>

        {/* Timeline Start / End timestamps */}
        <div className="flex justify-between text-[11px] font-mono text-slate-500 px-1">
          <span>00:00</span>
          <span className="text-slate-400">Total Track: {formatSeconds(totalDuration)}</span>
          <span>{formatSeconds(totalDuration)}</span>
        </div>
      </div>
    </div>
  );
}
