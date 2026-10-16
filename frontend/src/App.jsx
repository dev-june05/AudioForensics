import React, { useState } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import AudioUploader from './components/AudioUploader';
import ForensicsDashboard from './components/ForensicsDashboard';
import { Sparkles, Shield, Cpu, Activity, Database, FileText } from 'lucide-react';

export default function App() {
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    setError('');
    setResult(null);

    if (!selected) {
      setFile(null);
      setAudioUrl('');
      return;
    }

    setFile(selected);
    const url = URL.createObjectURL(selected);
    setAudioUrl(url);
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please choose or drop an audio recording first.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://127.0.0.1:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
    } catch (err) {
      if (err.response?.data?.detail) {
        setError(`Analysis error: ${err.response.data.detail}`);
      } else {
        setError('Could not connect to forensic engine on port 8000. Ensure the backend server is running.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setAudioUrl('');
    setResult(null);
    setError('');
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#070b14] text-slate-100">
      {/* 1. Header / Navbar */}
      <Navbar />

      {/* 2. Hero Title Section */}
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8 pb-4">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-slate-800/80 pb-6">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-sky-500/30 bg-sky-500/10 px-3 py-1 text-xs font-semibold text-sky-300 mb-3 shadow-inner">
              <Sparkles className="w-3.5 h-3.5 text-sky-400" />
              <span>Multi-Modal Deepfake & Speech Tampering Forensics</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
              Audio Tampering Detection Studio
            </h1>
            <p className="mt-2 text-sm sm:text-base text-slate-400 max-w-2xl leading-relaxed">
              Analyzes raw waveforms and 2D Log Mel-Spectrograms using an ensemble of 
              <span className="text-slate-200 font-medium"> ResNet-18</span> and 
              <span className="text-slate-200 font-medium"> Attention-BiLSTM</span> to pinpoint 
              synthetic speech, voice clones, and partial splicing timestamps.
            </p>
          </div>

          {/* Quick Metrics Bar */}
          <div className="flex items-center gap-4 bg-slate-900/80 border border-slate-800/80 rounded-2xl p-3 shrink-0">
            <div className="px-3 border-r border-slate-800">
              <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider block">
                Benchmark EER
              </span>
              <span className="text-lg font-black font-mono text-emerald-400">
                5.49%
              </span>
            </div>
            <div className="px-3 border-r border-slate-800">
              <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider block">
                Validation Acc
              </span>
              <span className="text-lg font-black font-mono text-sky-400">
                98.27%
              </span>
            </div>
            <div className="px-3">
              <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider block">
                Window Step
              </span>
              <span className="text-lg font-black font-mono text-indigo-400">
                30.0s
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Main Workspace Grid */}
      <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Audio Ingestion & Player (5 Cols) */}
          <div className="lg:col-span-5">
            <AudioUploader
              file={file}
              audioUrl={audioUrl}
              loading={loading}
              error={error}
              onFileChange={handleFileChange}
              onAnalyze={handleAnalyze}
              onReset={handleReset}
            />
          </div>

          {/* Right Column: Forensics Dashboard & Timeline (7 Cols) */}
          <div className="lg:col-span-7">
            <ForensicsDashboard result={result} loading={loading} />
          </div>
        </div>
      </main>

      {/* 4. Footer */}
      <footer className="w-full border-t border-slate-800/80 bg-slate-950/80 py-6 mt-12 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-sky-400" />
            <span>Audio Deepfake Forensics Lab · Trained on PartialSpoof / ASVspoof 2019 LA</span>
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <span>ResNet-18 (CNN)</span>
            <span>·</span>
            <span>BiLSTM + Attention</span>
            <span>·</span>
            <span>Wav2Vec 2.0 / HuBERT</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
