import React, { useState } from 'react';
import { 
  ShieldCheck, 
  ShieldAlert, 
  AlertTriangle, 
  Activity, 
  Cpu, 
  Layers, 
  Maximize2, 
  ChevronRight, 
  ChevronLeft,
  VolumeX,
  Sparkles,
  Info
} from 'lucide-react';
import TamperingTimeline from './TamperingTimeline';

export default function ForensicsDashboard({ result, loading }) {
  const [selectedWindowIdx, setSelectedWindowIdx] = useState(0);
  const [isModalOpen, setIsModalOpen] = useState(false);

  if (loading) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-8 flex flex-col items-center justify-center text-center min-h-[460px] animate-pulse">
        <div className="w-16 h-16 rounded-2xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center text-sky-400 mb-4">
          <Activity className="w-8 h-8 animate-spin" />
        </div>
        <h3 className="text-base font-semibold text-white mb-2">
          Executing Forensic Extraction Pipeline
        </h3>
        <p className="text-xs text-slate-400 max-w-md">
          Decomposing audio into 30-second temporal windows, applying Mel-scale STFT transforms, and running multi-model inference across ResNet-18 and BiLSTM.
        </p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="rounded-2xl border border-slate-800/80 bg-slate-900/40 p-8 flex flex-col items-center justify-center text-center min-h-[460px]">
        <div className="w-16 h-16 rounded-2xl bg-slate-800/60 border border-slate-700/60 flex items-center justify-center text-slate-400 mb-4">
          <ShieldCheck className="w-8 h-8" />
        </div>
        <h3 className="text-base font-semibold text-slate-200 mb-2">
          Forensic Intelligence Standby
        </h3>
        <p className="text-xs text-slate-400 max-w-md mb-6">
          Upload an audio recording (.wav, .mp3, .m4a, .flac, etc.) to trigger multi-modal spoof detection, partial tampering localization, and spectrogram analysis.
        </p>

        {/* Feature Highlights Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full max-w-lg text-left">
          <div className="rounded-xl bg-slate-950/60 border border-slate-800/80 p-3">
            <span className="text-[11px] font-semibold text-sky-400 uppercase tracking-wider block mb-1">
              Spectral CNN
            </span>
            <p className="text-[11px] text-slate-400">
              ResNet-18 detects vocoder artifacts & high-frequency distortion.
            </p>
          </div>
          <div className="rounded-xl bg-slate-950/60 border border-slate-800/80 p-3">
            <span className="text-[11px] font-semibold text-indigo-400 uppercase tracking-wider block mb-1">
              Temporal BiLSTM
            </span>
            <p className="text-[11px] text-slate-400">
              Learned attention pooling pinpoints brief tampering timestamps.
            </p>
          </div>
          <div className="rounded-xl bg-slate-950/60 border border-slate-800/80 p-3">
            <span className="text-[11px] font-semibold text-emerald-400 uppercase tracking-wider block mb-1">
              PartialSpoof SOTA
            </span>
            <p className="text-[11px] text-slate-400">
              Benchmarked on ASVspoof LA protocols with 5.49% baseline EER.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const { overall_status, overall_ai_probability, total_duration_sec, windows = [] } = result;
  const currentWindow = windows[selectedWindowIdx] || windows[0] || {};

  const isReal = overall_status === 'Real';
  const isSpoofed = overall_status === 'Spoofed';
  const isMixed = overall_status === 'MIXED / SUSPICIOUS' || (!isReal && !isSpoofed);

  const formatSeconds = (sec) => {
    if (isNaN(sec)) return '00:00';
    const mins = Math.floor(sec / 60);
    const secs = Math.floor(sec % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="flex flex-col gap-5">
      {/* 1. Global Verdict Banner */}
      <div
        className={`relative overflow-hidden rounded-2xl border p-5 sm:p-6 backdrop-blur-md shadow-2xl transition-all duration-300 ${
          isReal
            ? 'border-emerald-500/40 bg-gradient-to-r from-emerald-950/40 via-slate-900/80 to-slate-900/60 shadow-emerald-500/10'
            : isSpoofed
            ? 'border-rose-500/40 bg-gradient-to-r from-rose-950/40 via-slate-900/80 to-slate-900/60 shadow-rose-500/10'
            : 'border-amber-500/40 bg-gradient-to-r from-amber-950/40 via-slate-900/80 to-slate-900/60 shadow-amber-500/10'
        }`}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-start gap-4">
            <div
              className={`w-12 h-12 rounded-2xl flex items-center justify-center shrink-0 border shadow-lg ${
                isReal
                  ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-400 shadow-emerald-500/20'
                  : isSpoofed
                  ? 'bg-rose-500/20 border-rose-500/40 text-rose-400 shadow-rose-500/20'
                  : 'bg-amber-500/20 border-amber-500/40 text-amber-400 shadow-amber-500/20'
              }`}
            >
              {isReal ? (
                <ShieldCheck className="w-7 h-7" />
              ) : isSpoofed ? (
                <ShieldAlert className="w-7 h-7" />
              ) : (
                <AlertTriangle className="w-7 h-7" />
              )}
            </div>

            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  Forensic Consensus Verdict
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                  {windows.length} {windows.length === 1 ? 'Window' : 'Windows'} Analyzed
                </span>
              </div>

              <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-white">
                {isReal && 'Genuine Human Speech'}
                {isSpoofed && 'Synthetic / AI-Generated Speech'}
                {isMixed && 'Partial Spoofing / Tampered Audio Detected'}
              </h2>

              <p className="text-xs text-slate-300 mt-1 max-w-lg">
                {isReal &&
                  'Acoustic and spectral patterns exhibit natural human vocal tract continuity with no neural vocoder artifacts detected.'}
                {isSpoofed &&
                  'Strong synthetic acoustic signatures, vocoder phase anomalies, or diffusion artifacts detected across the recording.'}
                {isMixed &&
                  'The recording contains mixed characteristics: certain segments exhibit high synthetic probability while others appear genuine.'}
              </p>
            </div>
          </div>

          {/* AI Probability Gauge Box */}
          <div className="sm:text-right shrink-0 bg-slate-950/70 border border-slate-800/80 rounded-xl p-3 sm:min-w-[170px]">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-0.5">
              AI Likelihood
            </span>
            <div className="text-2xl font-black font-mono tracking-tight text-white">
              {(overall_ai_probability * 100).toFixed(1)}%
            </div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full mt-1.5 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  isReal ? 'bg-emerald-400' : isSpoofed ? 'bg-rose-500' : 'bg-amber-400'
                }`}
                style={{ width: `${Math.max(overall_ai_probability * 100, 5)}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* 2. Interactive Tampering Timeline */}
      <TamperingTimeline
        windows={windows}
        selectedIndex={selectedWindowIdx}
        onSelectWindow={setSelectedWindowIdx}
        totalDuration={total_duration_sec}
      />

      {/* 3. Window Inspector Card */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/60 backdrop-blur-md p-5 sm:p-6 shadow-xl shadow-black/30 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-sky-400 uppercase tracking-wider">
                Window {selectedWindowIdx + 1} of {windows.length}
              </span>
              <span className="text-slate-400">·</span>
              <span className="text-xs font-mono text-slate-200">
                {formatSeconds(currentWindow.start_time_sec)} - {formatSeconds(currentWindow.end_time_sec)}
              </span>
              {currentWindow.is_silent && (
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700 flex items-center gap-1">
                  <VolumeX className="w-3 h-3" /> Silent
                </span>
              )}
            </div>
            <h3 className="text-base font-semibold text-white mt-0.5">
              Forensic Spectrum & Model Consensus
            </h3>
          </div>

          {/* Window Navigation Buttons */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={selectedWindowIdx === 0}
              onClick={() => setSelectedWindowIdx((prev) => Math.max(prev - 1, 0))}
              className="px-3 py-1.5 rounded-lg border border-slate-800 bg-slate-950 text-xs text-slate-300 hover:text-white hover:border-slate-700 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1"
            >
              <ChevronLeft className="w-3.5 h-3.5" /> Previous
            </button>
            <button
              type="button"
              disabled={selectedWindowIdx === windows.length - 1}
              onClick={() => setSelectedWindowIdx((prev) => Math.min(prev + 1, windows.length - 1))}
              className="px-3 py-1.5 rounded-lg border border-slate-800 bg-slate-950 text-xs text-slate-300 hover:text-white hover:border-slate-700 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1"
            >
              Next <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Model Consensus Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {/* ResNet-18 */}
          <div className="rounded-xl border border-slate-800/80 bg-slate-950/60 p-3.5 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-sky-300">
                <Cpu className="w-4 h-4 text-sky-400" />
                <span>ResNet-18</span>
              </div>
              <span className="text-[10px] font-medium text-slate-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                Spectral CNN
              </span>
            </div>
            <div>
              <div className="text-lg font-bold font-mono text-white">
                {(currentWindow.resnet_prob * 100).toFixed(1)}%
                <span className="text-xs font-normal text-slate-400 ml-1.5">AI Prob</span>
              </div>
              <div className="w-full bg-slate-800/80 h-1 rounded-full mt-2 overflow-hidden">
                <div
                  className="bg-sky-400 h-full rounded-full"
                  style={{ width: `${Math.max(currentWindow.resnet_prob * 100, 3)}%` }}
                />
              </div>
            </div>
          </div>

          {/* BiLSTM */}
          <div className="rounded-xl border border-slate-800/80 bg-slate-950/60 p-3.5 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-indigo-300">
                <Activity className="w-4 h-4 text-indigo-400" />
                <span>BiLSTM + Attn</span>
              </div>
              <span className="text-[10px] font-medium text-slate-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                Temporal RNN
              </span>
            </div>
            <div>
              <div className="text-lg font-bold font-mono text-white">
                {(currentWindow.lstm_prob * 100).toFixed(1)}%
                <span className="text-xs font-normal text-slate-400 ml-1.5">AI Prob</span>
              </div>
              <div className="w-full bg-slate-800/80 h-1 rounded-full mt-2 overflow-hidden">
                <div
                  className="bg-indigo-400 h-full rounded-full"
                  style={{ width: `${Math.max(currentWindow.lstm_prob * 100, 3)}%` }}
                />
              </div>
            </div>
          </div>

          {/* Meta-Ensemble */}
          <div className="rounded-xl border border-sky-500/30 bg-sky-950/20 p-3.5 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-300">
                <Layers className="w-4 h-4 text-emerald-400" />
                <span>Ensemble Blend</span>
              </div>
              <span className="text-[10px] font-semibold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/30">
                Fused Verdict
              </span>
            </div>
            <div>
              <div className="text-lg font-bold font-mono text-white flex items-center justify-between">
                <span>{(currentWindow.ai_probability * 100).toFixed(1)}%</span>
                <span
                  className={`text-xs px-2 py-0.5 rounded-full font-semibold uppercase ${
                    currentWindow.prediction === 'Real'
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      : currentWindow.prediction === 'Spoofed'
                      ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                      : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                  }`}
                >
                  {currentWindow.prediction}
                </span>
              </div>
              <div className="w-full bg-slate-800/80 h-1 rounded-full mt-2 overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    currentWindow.prediction === 'Real' ? 'bg-emerald-400' : 'bg-rose-500'
                  }`}
                  style={{ width: `${Math.max(currentWindow.ai_probability * 100, 3)}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Mel-Spectrogram Image Display */}
        <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <span>Log Mel-Spectrogram (128 Mel Bins · 2048 N_FFT)</span>
            </span>
            {currentWindow.spectrogram_image && (
              <button
                type="button"
                onClick={() => setIsModalOpen(true)}
                className="text-xs text-sky-400 hover:text-sky-300 flex items-center gap-1 px-2.5 py-1 rounded bg-slate-900 border border-slate-800 hover:border-slate-700 transition-colors"
              >
                <Maximize2 className="w-3.5 h-3.5" /> Expand High-Res
              </button>
            )}
          </div>

          {currentWindow.spectrogram_image ? (
            <div className="relative rounded-lg overflow-hidden border border-slate-800 bg-black flex items-center justify-center">
              <img
                src={currentWindow.spectrogram_image}
                alt={`Mel-Spectrogram Window ${selectedWindowIdx + 1}`}
                className="w-full max-h-[280px] object-contain cursor-pointer hover:opacity-95 transition-opacity"
                onClick={() => setIsModalOpen(true)}
              />
            </div>
          ) : (
            <div className="h-44 rounded-lg border border-dashed border-slate-800 flex items-center justify-center text-xs text-slate-500 italic">
              No spectrogram available for silent or unanalyzed window.
            </div>
          )}
        </div>
      </div>

      {/* High-Res Spectrogram Modal */}
      {isModalOpen && currentWindow.spectrogram_image && (
        <div
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4"
          onClick={() => setIsModalOpen(false)}
        >
          <div
            className="relative max-w-5xl w-full bg-slate-900 border border-slate-700 rounded-2xl p-6 space-y-4 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-white">
                  High-Resolution Log Mel-Spectrogram
                </h3>
                <p className="text-xs text-slate-400 font-mono">
                  Window {selectedWindowIdx + 1} ({formatSeconds(currentWindow.start_time_sec)} -{' '}
                  {formatSeconds(currentWindow.end_time_sec)}) · AI Likelihood:{' '}
                  {(currentWindow.ai_probability * 100).toFixed(1)}%
                </p>
              </div>
              <button
                type="button"
                onClick={() => setIsModalOpen(false)}
                className="px-3 py-1.5 text-xs rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 font-semibold"
              >
                Close
              </button>
            </div>
            <div className="bg-black rounded-xl overflow-hidden border border-slate-800 p-2">
              <img
                src={currentWindow.spectrogram_image}
                alt="High-Res Spectrogram"
                className="w-full max-h-[70vh] object-contain mx-auto"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
