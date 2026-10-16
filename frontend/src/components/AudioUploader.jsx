import React, { useRef, useState, useEffect } from 'react';
import { 
  UploadCloud, 
  FileAudio, 
  Play, 
  Pause, 
  Volume2, 
  VolumeX, 
  RotateCcw, 
  Loader2, 
  CheckCircle, 
  AlertCircle,
  Clock,
  Sparkles,
  Layers
} from 'lucide-react';

const SUPPORTED_FORMATS = [
  { ext: '.WAV', desc: 'Lossless PCM' },
  { ext: '.MP3', desc: 'MPEG Audio' },
  { ext: '.M4A', desc: 'AAC / Apple Audio' },
  { ext: '.FLAC', desc: 'Free Lossless' },
  { ext: '.OGG', desc: 'Ogg Vorbis' },
  { ext: '.WEBM', desc: 'Web Media' },
  { ext: '.AAC', desc: 'Advanced Audio' }
];

export default function AudioUploader({ 
  file, 
  audioUrl, 
  loading, 
  error, 
  onFileChange, 
  onAnalyze, 
  onReset 
}) {
  const fileInputRef = useRef(null);
  const audioRef = useRef(null);
  
  // Custom audio player state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [analysisStep, setAnalysisStep] = useState(0);

  // Stepper effect during loading
  useEffect(() => {
    if (!loading) {
      setAnalysisStep(0);
      return;
    }
    const timer1 = setTimeout(() => setAnalysisStep(1), 1200);
    const timer2 = setTimeout(() => setAnalysisStep(2), 2600);
    const timer3 = setTimeout(() => setAnalysisStep(3), 4200);
    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
    };
  }, [loading]);

  const handleAudioTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleAudioLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
    }
  };

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const toggleMute = () => {
    if (!audioRef.current) return;
    audioRef.current.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const handleSeek = (e) => {
    const targetTime = parseFloat(e.target.value);
    if (audioRef.current) {
      audioRef.current.currentTime = targetTime;
      setCurrentTime(targetTime);
    }
  };

  const formatTime = (seconds) => {
    if (isNaN(seconds) || seconds === 0) return '00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 KB';
    if (bytes >= 1024 * 1024) {
      return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    }
    return (bytes / 1024).toFixed(1) + ' KB';
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      onFileChange({ target: { files: [droppedFile] } });
    }
  };

  return (
    <div className="flex flex-col gap-5">
      {/* Upload Zone Card */}
      <div className="relative rounded-2xl border border-slate-800 bg-slate-900/60 backdrop-blur-md p-5 sm:p-6 shadow-xl shadow-black/40">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <UploadCloud className="w-5 h-5 text-sky-400" />
            <h2 className="text-base font-semibold text-white tracking-tight">
              Audio Ingestion & Forensics
            </h2>
          </div>
          <span className="text-[11px] font-medium text-slate-400 bg-slate-800/80 px-2.5 py-1 rounded-full border border-slate-700/60">
            Up to 100 MB · Any Length
          </span>
        </div>

        {/* Drag and Drop Container */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative group cursor-pointer rounded-xl border-2 border-dashed transition-all duration-200 p-6 flex flex-col items-center justify-center text-center ${
            dragOver
              ? 'border-sky-400 bg-sky-950/20'
              : file
              ? 'border-slate-700/80 bg-slate-800/20 hover:border-slate-600'
              : 'border-slate-700/60 bg-slate-950/40 hover:border-sky-500/50 hover:bg-slate-900/40'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.m4a,.flac,.ogg,.webm,.aac,audio/*"
            onChange={onFileChange}
            className="hidden"
          />

          <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-sky-500/20 to-indigo-500/20 border border-sky-500/30 flex items-center justify-center mb-3 text-sky-400 group-hover:scale-105 transition-transform duration-200 shadow-inner">
            <FileAudio className="w-7 h-7" />
          </div>

          <h3 className="text-sm font-semibold text-slate-200 mb-1">
            {file ? 'Replace audio file' : 'Drop audio recording here or browse'}
          </h3>
          <p className="text-xs text-slate-400 max-w-sm mb-4">
            Accepts speech files, call recordings, voice notes, and audio streams.
          </p>

          {/* Formats Badges Grid */}
          <div className="flex flex-wrap items-center justify-center gap-1.5 max-w-md">
            {SUPPORTED_FORMATS.map((fmt) => (
              <span
                key={fmt.ext}
                className="text-[10px] font-semibold tracking-wider px-2 py-0.5 rounded-md bg-slate-800/90 text-slate-300 border border-slate-700/60 hover:text-sky-300 hover:border-sky-500/40 transition-colors"
                title={fmt.desc}
              >
                {fmt.ext}
              </span>
            ))}
          </div>
        </div>

        {/* Selected Audio Information & Custom Player */}
        {file && audioUrl && (
          <div className="mt-5 space-y-3 pt-4 border-t border-slate-800/80">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2 truncate pr-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400" />
                <span className="font-semibold text-slate-200 truncate">{file.name}</span>
              </div>
              <span className="text-slate-400 shrink-0 font-mono">
                {formatFileSize(file.size)}
              </span>
            </div>

            {/* Hidden native audio element */}
            <audio
              ref={audioRef}
              src={audioUrl}
              onTimeUpdate={handleAudioTimeUpdate}
              onLoadedMetadata={handleAudioLoadedMetadata}
              onEnded={() => setIsPlaying(false)}
            />

            {/* Custom Sleek Player UI */}
            <div className="rounded-xl bg-slate-950/80 border border-slate-800 p-3.5 flex flex-col gap-2.5">
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={togglePlay}
                  className="w-9 h-9 rounded-full bg-sky-500 hover:bg-sky-400 text-slate-950 flex items-center justify-center shrink-0 shadow-md shadow-sky-500/20 transition-transform active:scale-95"
                >
                  {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current ml-0.5" />}
                </button>

                {/* Scrubber slider */}
                <div className="flex-1 flex flex-col gap-1">
                  <input
                    type="range"
                    min="0"
                    max={duration || 100}
                    step="0.01"
                    value={currentTime}
                    onChange={handleSeek}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-400"
                  />
                  <div className="flex justify-between text-[11px] font-mono text-slate-400">
                    <span>{formatTime(currentTime)}</span>
                    <span>{formatTime(duration)}</span>
                  </div>
                </div>

                {/* Mute button */}
                <button
                  type="button"
                  onClick={toggleMute}
                  className="p-1.5 text-slate-400 hover:text-slate-200 transition-colors"
                >
                  {isMuted ? <VolumeX className="w-4 h-4 text-rose-400" /> : <Volume2 className="w-4 h-4" />}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Primary Action Button */}
        <div className="mt-5 space-y-3">
          <button
            type="button"
            onClick={onAnalyze}
            disabled={loading || !file}
            className="w-full py-3.5 px-6 rounded-xl font-semibold text-sm tracking-wide bg-gradient-to-r from-sky-500 via-sky-400 to-indigo-500 text-slate-950 shadow-lg shadow-sky-500/25 hover:shadow-sky-500/40 hover:opacity-95 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-slate-950" />
                <span>Running Deep Forensic Analysis...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 fill-current" />
                <span>Analyze Audio Forensics</span>
              </>
            )}
          </button>

          {/* Stepper info during loading */}
          {loading && (
            <div className="rounded-xl bg-slate-950/60 border border-slate-800/80 p-3 space-y-2 text-xs">
              <div className="flex items-center gap-2 text-slate-300">
                <span className={`w-2 h-2 rounded-full ${analysisStep >= 0 ? 'bg-sky-400 animate-ping' : 'bg-slate-600'}`} />
                <span>{analysisStep === 0 && 'Ingesting audio stream & detecting sample frequency (16 kHz)...'}</span>
                <span>{analysisStep === 1 && 'Segmenting audio into 30s analysis windows...'}</span>
                <span>{analysisStep === 2 && 'Generating 128-band Mel-Spectrogram & temporal embeddings...'}</span>
                <span>{analysisStep >= 3 && 'Evaluating ResNet-18, BiLSTM & Ensemble weights...'}</span>
              </div>
              <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                <div 
                  className="bg-gradient-to-r from-sky-400 to-indigo-500 h-full transition-all duration-700 ease-out"
                  style={{ width: `${Math.min((analysisStep + 1) * 25, 95)}%` }}
                />
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="rounded-xl border border-rose-500/40 bg-rose-950/30 p-3 text-xs text-rose-300 flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
