"use client";

import { useRef, useState } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface LogEntry {
  step: string;
  message: string;
  timestamp: string;
}

interface DupTextPair {
  entry_a: { title: string; abstract_snippet: string; id: string };
  entry_b: { title: string; abstract_snippet: string; id: string };
  similarity: number;
}

interface DupImagePair {
  entry_a: { title: string; thumbnail: string; url: string };
  entry_b: { title: string; thumbnail: string; url: string };
  hamming_distance: number;
  similarity: number;
}

interface Savings {
  data_reduction_pct: number;
  duplicates_removed: number;
  gpu_hours_saved: number;
  energy_saved_kwh: number;
  cost_saved_usd: number;
  carbon_saved_kg_co2: number;
  water_saved_litres: number;
  baseline_gpu_hours: number;
}

interface Report {
  run_id: string;
  topic: string;
  counts: {
    raw_text: number;
    raw_image: number;
    final_text: number;
    final_image: number;
  };
  savings: Savings;
  text_dup_pairs: DupTextPair[];
  image_dup_pairs: DupImagePair[];
  sample_texts: {
    instruction?: string;
    input?: string;
    output?: string;
    title?: string;
    abstract_snippet?: string;
    text?: string;
  }[];
  sample_images: { title: string; thumbnail: string; license: string; source?: string }[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const QUICK_STARTS = [
  "I need training data about renewable energy",
  "I need training data about ocean & marine life",
  "I need training data about wildlife conservation",
];

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const STEP_LABELS: Record<string, string> = {
  intent: "Parsing intent",
  intent_done: "Intent parsed",
  collect: "Collecting data",
  collect_done: "Data collected",
  clean: "Cleaning data",
  clean_done: "Data cleaned",
  filter: "Quality filtering",
  filter_done: "Quality filter done",
  dedup: "Deduplicating",
  dedup_done: "Deduplication done",
  structure: "Structuring & synthesising",
  structure_done: "Output structured",
  done: "Pipeline complete",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmt(n: number, decimals = 2): string {
  return Number.isFinite(n) ? n.toFixed(decimals) : "0";
}

function copyToClipboard(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {});
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatCard({
  label,
  value,
  sub,
  icon,
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  icon?: string;
  accent?: boolean;
}) {
  return (
    <div
      className={`rounded-2xl p-5 flex flex-col gap-2 border transition-colors ${
        accent
          ? "bg-gradient-to-br from-indigo-900/40 to-violet-900/30 border-indigo-500/30 hover:border-indigo-400/50"
          : "bg-white/[0.02] border-white/10 hover:bg-white/[0.04]"
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
          {label}
        </span>
        {icon && <span className="text-base opacity-60">{icon}</span>}
      </div>
      <span
        className={`text-3xl font-extrabold tracking-tight ${
          accent ? "text-indigo-300" : "text-white"
        }`}
      >
        {value}
      </span>
      {sub && <span className="text-xs text-slate-500">{sub}</span>}
    </div>
  );
}

function SavingsCard({
  value,
  unit,
  label,
  icon,
}: {
  value: string;
  unit: string;
  label: string;
  icon: string;
}) {
  return (
    <div className="flex flex-col gap-1.5 p-4 rounded-xl bg-white/[0.02] border border-white/8 hover:bg-white/[0.04] transition-colors">
      <span className="text-xl">{icon}</span>
      <div className="flex items-baseline gap-1 mt-1">
        <span className="text-xl font-bold text-emerald-300">{value}</span>
        {unit && <span className="text-xs text-emerald-400/70 font-medium">{unit}</span>}
      </div>
      <span className="text-xs text-slate-500 leading-snug">{label}</span>
    </div>
  );
}

function TextDupProof({ pair }: { pair: DupTextPair }) {
  return (
    <div className="rounded-2xl border border-amber-500/20 bg-amber-500/[0.02] p-5 space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <span className="text-sm font-semibold text-amber-300">
          Near-duplicate text pair — live proof
        </span>
        <span className="text-sm font-bold text-amber-200 bg-amber-500/15 border border-amber-500/25 px-3 py-1 rounded-full tabular-nums">
          {(pair.similarity * 100).toFixed(1)}% match
        </span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="rounded-xl border-l-4 border-emerald-500 bg-emerald-950/20 border border-emerald-500/20 p-4">
          <span className="inline-block text-[10px] font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded mb-2 tracking-wider">
            ✓ KEPT
          </span>
          <p className="text-sm font-semibold text-white mb-1.5 line-clamp-2">
            {pair.entry_a.title}
          </p>
          <p className="text-xs text-slate-400 line-clamp-4 leading-relaxed">
            {pair.entry_a.abstract_snippet}
          </p>
        </div>
        <div className="rounded-xl border-l-4 border-rose-500 bg-rose-950/20 border border-rose-500/20 p-4 opacity-70">
          <span className="inline-block text-[10px] font-bold text-rose-400 bg-rose-500/10 border border-rose-500/20 px-2 py-0.5 rounded mb-2 tracking-wider">
            ✕ DROPPED
          </span>
          <p className="text-sm font-semibold text-white mb-1.5 line-clamp-2">
            {pair.entry_b.title}
          </p>
          <p className="text-xs text-slate-400 line-clamp-4 leading-relaxed">
            {pair.entry_b.abstract_snippet}
          </p>
        </div>
      </div>
    </div>
  );
}

function ImageDupProof({ pair }: { pair: DupImagePair }) {
  return (
    <div className="rounded-2xl border border-violet-500/20 bg-violet-500/[0.02] p-5 space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <span className="text-sm font-semibold text-violet-300">
          Near-duplicate image pair — live proof
        </span>
        <span className="text-sm font-bold text-violet-200 bg-violet-500/15 border border-violet-500/25 px-3 py-1 rounded-full tabular-nums">
          Hamming {pair.hamming_distance}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-xl border-l-4 border-emerald-500 bg-emerald-950/20 border border-emerald-500/20 p-3">
          <span className="inline-block text-[10px] font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded mb-2 tracking-wider">
            ✓ KEPT
          </span>
          {pair.entry_a.thumbnail ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={pair.entry_a.thumbnail}
              alt={pair.entry_a.title}
              className="w-full h-28 object-cover rounded-lg border border-white/10"
            />
          ) : (
            <div className="w-full h-28 bg-white/5 rounded-lg flex items-center justify-center text-slate-600 text-xs border border-white/5">
              No preview
            </div>
          )}
          <p className="text-xs text-slate-400 mt-2 line-clamp-1">
            {pair.entry_a.title || "Untitled"}
          </p>
        </div>
        <div className="rounded-xl border-l-4 border-rose-500 bg-rose-950/20 border border-rose-500/20 p-3 opacity-70">
          <span className="inline-block text-[10px] font-bold text-rose-400 bg-rose-500/10 border border-rose-500/20 px-2 py-0.5 rounded mb-2 tracking-wider">
            ✕ DROPPED
          </span>
          {pair.entry_b.thumbnail ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={pair.entry_b.thumbnail}
              alt={pair.entry_b.title}
              className="w-full h-28 object-cover rounded-lg border border-white/10"
            />
          ) : (
            <div className="w-full h-28 bg-white/5 rounded-lg flex items-center justify-center text-slate-600 text-xs border border-white/5">
              No preview
            </div>
          )}
          <p className="text-xs text-slate-400 mt-2 line-clamp-1">
            {pair.entry_b.title || "Untitled"}
          </p>
        </div>
      </div>
    </div>
  );
}

function AlpacaCard({
  record,
  index,
}: {
  record: {
    instruction?: string;
    input?: string;
    output?: string;
    title?: string;
    abstract_snippet?: string;
    text?: string;
  };
  index: number;
}) {
  const instruction = record.instruction || record.title || "";
  const input = record.input || "";
  const output = record.output || record.abstract_snippet || record.text || "";

  const json = JSON.stringify({ instruction, input, output }, null, 2);

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.02] hover:bg-white/[0.04] transition-colors overflow-hidden">
      {/* Card header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/8 bg-white/[0.02]">
        <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">
          Record #{index + 1}
        </span>
        <button
          onClick={() => copyToClipboard(json)}
          title="Copy JSON"
          className="text-[10px] text-slate-500 hover:text-slate-300 transition flex items-center gap-1 font-mono"
        >
          ⎘ Copy JSON
        </button>
      </div>

      <div className="p-4 space-y-3">
        {/* Instruction */}
        <div>
          <span className="text-[10px] font-bold text-white uppercase tracking-widest block mb-1.5">
            Instruction
          </span>
          <p className="text-sm font-semibold text-white leading-snug line-clamp-3">
            {instruction || <span className="text-slate-600 italic">—</span>}
          </p>
        </div>

        {/* Input / Context */}
        {input && (
          <div>
            <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-indigo-300 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded tracking-wider mb-1.5">
              &lt;/&gt; Input Context
            </span>
            <p className="text-xs text-slate-400 font-mono leading-relaxed line-clamp-2 bg-black/30 rounded-lg px-3 py-2 border border-white/5">
              {input}
            </p>
          </div>
        )}

        {/* Output */}
        {output && (
          <div>
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block mb-1.5">
              Generated Output
            </span>
            <div className="rounded-lg bg-slate-900/60 border border-white/8 px-3 py-2.5">
              <p className="text-xs text-slate-300 leading-relaxed line-clamp-3">
                {output}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [topic, setTopic] = useState<string | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const logBottomRef = useRef<HTMLDivElement>(null);

  function appendLog(step: string, message: string) {
    setLog((prev) => [
      ...prev,
      { step, message, timestamp: new Date().toLocaleTimeString() },
    ]);
    setTimeout(
      () => logBottomRef.current?.scrollIntoView({ behavior: "smooth" }),
      50
    );
  }

  async function startPipeline() {
    if (!prompt.trim()) return;
    setRunning(true);
    setLog([]);
    setReport(null);
    setTopic(null);
    setError(null);
    setRunId(null);

    try {
      const initRes = await fetch(`${API_BASE}/mine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!initRes.ok) throw new Error(`Server error: ${initRes.status}`);
      const { run_id } = await initRes.json();
      setRunId(run_id);

      const streamUrl = `${API_BASE}/mine/${run_id}/stream?prompt=${encodeURIComponent(prompt)}`;
      const evtSource = new EventSource(streamUrl);

      evtSource.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          const label = STEP_LABELS[data.step] ?? data.step;
          appendLog(
            data.step,
            label + (data.message ? ` — ${data.message}` : "")
          );
          if (data.step === "intent_done" && data.topic) setTopic(data.topic);
          if (data.step === "done" && data.report) {
            setReport(data.report);
            evtSource.close();
            setRunning(false);
          }
        } catch {
          // ignore parse errors
        }
      };

      evtSource.onerror = () => {
        evtSource.close();
        setRunning(false);
        setError("Connection lost. Please try again.");
      };
    } catch (err: unknown) {
      setRunning(false);
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  }

  const totalRaw = report
    ? report.counts.raw_text + report.counts.raw_image
    : 0;
  const totalFinal = report
    ? report.counts.final_text + report.counts.final_image
    : 0;

  return (
    <main className="min-h-screen bg-slate-950 text-white">

      {/* ── Ambient radial background ── */}
      <div
        className="pointer-events-none fixed inset-0"
        style={{
          background:
            "radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99,102,241,0.15) 0%, transparent 70%)",
        }}
      />
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute top-1/4 -left-64 w-[600px] h-[600px] rounded-full bg-indigo-900/10 blur-[140px]" />
        <div className="absolute bottom-0 right-0 w-[500px] h-[500px] rounded-full bg-violet-900/8 blur-[120px]" />
      </div>

      <div className="relative max-w-4xl mx-auto px-4 pb-28">

        {/* ── Header ── */}
        <header className="pt-16 pb-12 text-center space-y-5 relative">

          {/* Left decorative circuit SVG */}
          <svg
            className="pointer-events-none absolute left-0 top-8 w-48 opacity-20 hidden lg:block"
            viewBox="0 0 180 260"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <circle cx="16" cy="16" r="5" stroke="#818cf8" strokeWidth="1.5"/>
            <circle cx="16" cy="80" r="5" stroke="#818cf8" strokeWidth="1.5"/>
            <circle cx="16" cy="144" r="5" stroke="#818cf8" strokeWidth="1.5"/>
            <circle cx="16" cy="208" r="5" stroke="#818cf8" strokeWidth="1.5"/>
            <line x1="16" y1="21" x2="16" y2="75" stroke="#4f46e5" strokeWidth="1" strokeDasharray="4 3"/>
            <line x1="16" y1="85" x2="16" y2="139" stroke="#4f46e5" strokeWidth="1" strokeDasharray="4 3"/>
            <line x1="16" y1="149" x2="16" y2="203" stroke="#4f46e5" strokeWidth="1" strokeDasharray="4 3"/>
            <line x1="21" y1="16" x2="80" y2="16" stroke="#6366f1" strokeWidth="1"/>
            <line x1="21" y1="80" x2="120" y2="80" stroke="#6366f1" strokeWidth="1"/>
            <line x1="21" y1="144" x2="90" y2="144" stroke="#6366f1" strokeWidth="1"/>
            <line x1="21" y1="208" x2="110" y2="208" stroke="#6366f1" strokeWidth="1"/>
            <rect x="80" y="8" width="16" height="16" rx="3" stroke="#a5b4fc" strokeWidth="1.2"/>
            <rect x="120" y="72" width="16" height="16" rx="3" stroke="#a5b4fc" strokeWidth="1.2"/>
            <rect x="90" y="136" width="16" height="16" rx="3" stroke="#a5b4fc" strokeWidth="1.2"/>
            <rect x="110" y="200" width="16" height="16" rx="3" stroke="#a5b4fc" strokeWidth="1.2"/>
            <circle cx="160" cy="48" r="3" fill="#6366f1" opacity="0.6"/>
            <circle cx="140" cy="112" r="3" fill="#818cf8" opacity="0.5"/>
            <circle cx="155" cy="176" r="3" fill="#6366f1" opacity="0.6"/>
            <line x1="96" y1="24" x2="96" y2="72" stroke="#4f46e5" strokeWidth="1" strokeDasharray="3 4" opacity="0.7"/>
            <line x1="128" y1="80" x2="160" y2="48" stroke="#4f46e5" strokeWidth="1" opacity="0.5"/>
            <line x1="98" y1="152" x2="140" y2="112" stroke="#4f46e5" strokeWidth="1" opacity="0.5"/>
            <text x="85" y="20" fontSize="6" fill="#818cf8" opacity="0.8">01</text>
            <text x="125" y="84" fontSize="6" fill="#818cf8" opacity="0.8">10</text>
            <text x="95" y="148" fontSize="6" fill="#818cf8" opacity="0.8">11</text>
            <text x="115" y="212" fontSize="6" fill="#818cf8" opacity="0.8">00</text>
          </svg>

          {/* Right decorative circuit SVG */}
          <svg
            className="pointer-events-none absolute right-0 top-8 w-48 opacity-20 hidden lg:block"
            viewBox="0 0 180 260"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            style={{ transform: "scaleX(-1)" }}
          >
            <circle cx="16" cy="16" r="5" stroke="#818cf8" strokeWidth="1.5"/>
            <circle cx="16" cy="80" r="5" stroke="#818cf8" strokeWidth="1.5"/>
            <circle cx="16" cy="144" r="5" stroke="#818cf8" strokeWidth="1.5"/>
            <circle cx="16" cy="208" r="5" stroke="#818cf8" strokeWidth="1.5"/>
            <line x1="16" y1="21" x2="16" y2="75" stroke="#4f46e5" strokeWidth="1" strokeDasharray="4 3"/>
            <line x1="16" y1="85" x2="16" y2="139" stroke="#4f46e5" strokeWidth="1" strokeDasharray="4 3"/>
            <line x1="16" y1="149" x2="16" y2="203" stroke="#4f46e5" strokeWidth="1" strokeDasharray="4 3"/>
            <line x1="21" y1="16" x2="80" y2="16" stroke="#6366f1" strokeWidth="1"/>
            <line x1="21" y1="80" x2="120" y2="80" stroke="#6366f1" strokeWidth="1"/>
            <line x1="21" y1="144" x2="90" y2="144" stroke="#6366f1" strokeWidth="1"/>
            <line x1="21" y1="208" x2="110" y2="208" stroke="#6366f1" strokeWidth="1"/>
            <rect x="80" y="8" width="16" height="16" rx="3" stroke="#a5b4fc" strokeWidth="1.2"/>
            <rect x="120" y="72" width="16" height="16" rx="3" stroke="#a5b4fc" strokeWidth="1.2"/>
            <rect x="90" y="136" width="16" height="16" rx="3" stroke="#a5b4fc" strokeWidth="1.2"/>
            <rect x="110" y="200" width="16" height="16" rx="3" stroke="#a5b4fc" strokeWidth="1.2"/>
            <circle cx="160" cy="48" r="3" fill="#6366f1" opacity="0.6"/>
            <circle cx="140" cy="112" r="3" fill="#818cf8" opacity="0.5"/>
            <circle cx="155" cy="176" r="3" fill="#6366f1" opacity="0.6"/>
            <line x1="96" y1="24" x2="96" y2="72" stroke="#4f46e5" strokeWidth="1" strokeDasharray="3 4" opacity="0.7"/>
            <line x1="128" y1="80" x2="160" y2="48" stroke="#4f46e5" strokeWidth="1" opacity="0.5"/>
            <line x1="98" y1="152" x2="140" y2="112" stroke="#4f46e5" strokeWidth="1" opacity="0.5"/>
            <text x="85" y="20" fontSize="6" fill="#818cf8" opacity="0.8">01</text>
            <text x="125" y="84" fontSize="6" fill="#818cf8" opacity="0.8">10</text>
            <text x="95" y="148" fontSize="6" fill="#818cf8" opacity="0.8">11</text>
            <text x="115" y="212" fontSize="6" fill="#818cf8" opacity="0.8">00</text>
          </svg>

          <div className="inline-flex items-center gap-2 bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-[11px] font-semibold px-4 py-1.5 rounded-full uppercase tracking-widest">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse shadow-lg shadow-green-400/50" />
            IBM AI Builders Challenge · Future of Work
          </div>

          <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight leading-none">
            <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-500 bg-clip-text text-transparent">
              DataMine AI
            </span>
          </h1>

          <p className="text-slate-400 text-base md:text-lg max-w-xl mx-auto leading-relaxed">
            Describe what you need training data for. The AI agent pipeline
            collects, cleans, quality-filters, and deduplicates it —
            automatically.
          </p>
        </header>

        {/* ── Prompt card ── */}
        <section className="mb-5">
          <div className="bg-slate-900/60 backdrop-blur-xl border border-slate-800/80 rounded-2xl p-6 shadow-2xl shadow-indigo-950/20 space-y-4">
            <label
              htmlFor="prompt"
              className="block text-sm font-medium text-slate-300"
            >
              What do you need training data for?
            </label>

            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={running}
              rows={3}
              placeholder="e.g. I need training data about LLM security vulnerabilities and prompt injection attacks"
              className="w-full bg-slate-950/80 border border-slate-700/60 rounded-xl px-4 py-3 text-white placeholder-slate-600 text-sm resize-none focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 transition disabled:opacity-40"
            />

            <div className="flex flex-wrap gap-2">
              {QUICK_STARTS.map((q) => (
                <button
                  key={q}
                  onClick={() => setPrompt(q)}
                  disabled={running}
                  className="text-xs bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/60 hover:border-slate-600 text-slate-300 hover:text-white px-3 py-1.5 rounded-full transition-all duration-150 disabled:opacity-30 hover:scale-[1.02]"
                >
                  {q}
                </button>
              ))}
            </div>

            <button
              onClick={startPipeline}
              disabled={running || !prompt.trim()}
              className="w-full bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 disabled:from-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed text-white font-medium py-3.5 rounded-xl shadow-lg shadow-indigo-500/25 transition-all duration-200 hover:scale-[1.01] text-sm tracking-wide"
            >
              {running ? (
                <span className="flex items-center justify-center gap-2.5">
                  <span className="w-4 h-4 rounded-full border-2 border-white/25 border-t-white animate-spin" />
                  Pipeline running…
                </span>
              ) : (
                "Start Mining →"
              )}
            </button>
          </div>
        </section>

        {/* ── Topic badge ── */}
        {topic && (
          <div className="mb-5 rounded-xl border border-indigo-500/25 bg-indigo-950/30 px-4 py-2.5 text-sm text-indigo-200 flex items-center gap-2.5">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 shrink-0" />
            <span className="text-indigo-400 font-semibold">Understood topic:</span>
            <span className="font-medium">{topic}</span>
          </div>
        )}

        {/* ── Error ── */}
        {error && (
          <div className="mb-5 rounded-xl border border-red-500/25 bg-red-950/20 px-4 py-3 text-sm text-red-300 flex items-center gap-2">
            <span className="text-red-400">⚠</span>
            {error}
          </div>
        )}

        {/* ── macOS-style terminal log ── */}
        {log.length > 0 && (
          <section className="mb-8">
            {/* Terminal chrome bar */}
            <div className="flex items-center justify-between px-4 py-2.5 bg-slate-800/80 border border-slate-700/60 border-b-0 rounded-t-2xl backdrop-blur-sm">
              {/* macOS dots */}
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-full bg-red-500/80" />
                <span className="w-3 h-3 rounded-full bg-yellow-500/80" />
                <span className="w-3 h-3 rounded-full bg-green-500/80" />
              </div>
              {/* Title */}
              <span className="text-xs font-mono text-slate-500 tracking-wide">
                pipeline.log
              </span>
              {/* Live badge */}
              {running ? (
                <span className="flex items-center gap-1.5 text-[10px] font-semibold text-green-400 uppercase tracking-wider">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                  Live
                </span>
              ) : (
                <span className="text-[10px] font-semibold text-slate-600 uppercase tracking-wider">
                  Done
                </span>
              )}
            </div>

            {/* Console body */}
            <div className="log-scroll bg-slate-950 border border-slate-700/60 border-t-0 rounded-b-2xl p-4 h-56 overflow-y-auto shadow-inner">
              {log.map((entry, i) => (
                <div
                  key={i}
                  className="flex gap-3 items-start py-0.5 text-xs font-mono"
                >
                  <span className="text-slate-600 shrink-0 tabular-nums select-none">
                    {entry.timestamp}
                  </span>
                  <span
                    className={
                      entry.step === "done"
                        ? "text-green-400 font-bold"
                        : entry.step.endsWith("_done")
                        ? "text-indigo-400"
                        : "text-slate-400"
                    }
                  >
                    <span className="mr-1.5 select-none">
                      {entry.step === "done"
                        ? "✔"
                        : entry.step.endsWith("_done")
                        ? "✔"
                        : "›"}
                    </span>
                    {entry.message}
                  </span>
                </div>
              ))}
              <div ref={logBottomRef} />
            </div>
          </section>
        )}

        {/* ── Dashboard ── */}
        {report && (
          <section className="space-y-8">

            {/* Section divider */}
            <div className="flex items-center gap-4">
              <div className="flex-1 h-px bg-white/8" />
              <span className="text-[11px] font-bold text-slate-500 uppercase tracking-[0.2em]">
                Run Report
              </span>
              <div className="flex-1 h-px bg-white/8" />
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <StatCard
                label="Raw Entries"
                value={String(totalRaw)}
                sub={`${report.counts.raw_text} text · ${report.counts.raw_image} img`}
                icon="📥"
              />
              <StatCard
                label="Clean Entries"
                value={String(totalFinal)}
                sub={`${report.counts.final_text} text · ${report.counts.final_image} img`}
                icon="✨"
                accent
              />
              <StatCard
                label="Data Reduction"
                value={`${fmt(report.savings.data_reduction_pct, 1)}%`}
                sub={`${report.savings.duplicates_removed} duplicates removed`}
                icon="🔁"
              />
              <StatCard
                label="GPU-Hrs Saved"
                value={fmt(report.savings.gpu_hours_saved)}
                sub={`of ${report.savings.baseline_gpu_hours}h baseline`}
                icon="⚡"
              />
            </div>

            {/* Savings panel */}
            <div className="rounded-2xl border border-emerald-500/20 bg-emerald-950/10 p-6">
              <div className="flex items-center gap-2 mb-5">
                <span className="text-[11px] font-bold text-emerald-400 uppercase tracking-[0.18em]">
                  Estimated Savings vs Training on Raw Data
                </span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <SavingsCard
                  value={fmt(report.savings.energy_saved_kwh)}
                  unit="kWh"
                  label="Energy saved"
                  icon="⚡"
                />
                <SavingsCard
                  value={`$${fmt(report.savings.cost_saved_usd)}`}
                  unit=""
                  label="Compute cost saved"
                  icon="💰"
                />
                <SavingsCard
                  value={fmt(report.savings.carbon_saved_kg_co2)}
                  unit="kg CO₂"
                  label="Carbon avoided"
                  icon="🌿"
                />
                <SavingsCard
                  value={fmt(report.savings.water_saved_litres)}
                  unit="L"
                  label="Water saved"
                  icon="💧"
                />
              </div>
              <p className="text-[10px] text-slate-600 mt-4 leading-relaxed border-t border-white/5 pt-4">
                Methodology: NVIDIA A100 SXM4 TDP 400 W · Training time scales
                linearly with data volume · Lambda Labs A100 @ $2.49/hr ·
                US EPA eGRID 2022 386 g CO₂/kWh · Patterson et al. 2021 1.8
                L/kWh
              </p>
            </div>

            {/* Duplicate proof */}
            {(report.text_dup_pairs.length > 0 ||
              report.image_dup_pairs.length > 0) && (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-bold text-slate-500 uppercase tracking-[0.18em]">
                    Duplicate-Pair Proof
                  </span>
                  <span className="text-[10px] text-slate-600 font-normal normal-case tracking-normal">
                    — live-computed from this run
                  </span>
                </div>
                {report.text_dup_pairs.slice(0, 1).map((pair, i) => (
                  <TextDupProof key={i} pair={pair} />
                ))}
                {report.image_dup_pairs.slice(0, 1).map((pair, i) => (
                  <ImageDupProof key={i} pair={pair} />
                ))}
              </div>
            )}

            {/* Alpaca training records */}
            {report.sample_texts.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-bold text-slate-500 uppercase tracking-[0.18em]">
                    Alpaca Training Records
                  </span>
                  <span className="text-[10px] text-slate-600 normal-case tracking-normal font-normal">
                    — sample output from this run
                  </span>
                </div>
                <div className="space-y-2">
                  {report.sample_texts.map((t, i) => (
                    <AlpacaCard key={i} record={t} index={i} />
                  ))}
                </div>
              </div>
            )}

            {/* Sample images */}
            {report.sample_images.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-bold text-slate-500 uppercase tracking-[0.18em]">
                    Sample Images
                  </span>
                  <span className="text-[10px] text-slate-600 font-normal normal-case">
                    — {report.sample_images.length} shown
                  </span>
                </div>
                <div className="grid grid-cols-3 md:grid-cols-4 gap-3 mt-2">
                  {report.sample_images.map((img, i) => (
                    <div key={i} className="group relative rounded-xl overflow-hidden border border-white/8 hover:border-white/20 transition bg-white/3">
                      {img.thumbnail ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={img.thumbnail}
                          alt={img.title || "Image"}
                          className="w-full aspect-video object-cover"
                          onError={(e) => {
                            const el = e.target as HTMLImageElement;
                            el.style.display = "none";
                            const parent = el.parentElement;
                            if (parent) {
                              parent.innerHTML = `<div class="w-full aspect-video bg-white/5 flex items-center justify-center text-slate-600 text-xs">No preview</div>`;
                            }
                          }}
                        />
                      ) : (
                        <div className="w-full aspect-video bg-white/5 flex items-center justify-center text-slate-600 text-xs">
                          No preview
                        </div>
                      )}
                      <div className="p-2">
                        {img.title && (
                          <p className="text-[10px] text-slate-400 truncate leading-snug mb-0.5">
                            {img.title}
                          </p>
                        )}
                        <div className="flex items-center justify-between gap-1">
                          <span className="text-[9px] text-slate-600 uppercase tracking-wider">
                            {img.license || "unknown"}
                          </span>
                          {img.source && (
                            <span className="text-[9px] text-indigo-500/70 uppercase tracking-wider">
                              {img.source === "wikimedia" ? "wiki" : img.source === "ddg_images" ? "ddg" : img.source}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Download */}
            {runId && (
              <div className="pt-2 pb-4">
                <a
                  href={`${API_BASE}/mine/${runId}/download`}
                  download
                  className="inline-flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-semibold px-7 py-3.5 rounded-xl transition-all duration-200 hover:scale-[1.02] text-sm shadow-xl shadow-indigo-900/40 tracking-wide"
                >
                  ↓ Download Dataset (.zip)
                </a>
              </div>
            )}

          </section>
        )}
      </div>
    </main>
  );
}
