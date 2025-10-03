// AskForm.tsx
"use client";

import { useState } from "react";
import type { AskResponse } from "@/lib/pdfSources";

export default function AskForm({
  onResult,
}: {
  onResult: (data: AskResponse) => void;
}) {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function ask() {
    if (!q.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: q.trim(), k: 6 }), // keep k modest
      });
      const text = await res.text();       // read as text first
      let raw: any = null;
      try { raw = JSON.parse(text); } catch { /* not JSON? */ }

      console.log("[/api/ask] status", res.status, "raw:", raw ?? text);

      if (!res.ok) throw new Error(`HTTP ${res.status}: ${text}`);

      const normalized: AskResponse = {
        answer: raw?.result?.answer ?? raw?.answer ?? "",
        hits: Array.isArray(raw?.result?.hits)
          ? raw.result.hits
          : Array.isArray(raw?.hits)
          ? raw.hits
          : [],
      };

      // Ensure we always show something
      if (!normalized.answer) normalized.answer = "(empty answer)";
      onResult(normalized);
    } catch (e: any) {
      setErr(e?.message || "Request failed");
      // also push something up so the page shows a box
      onResult({ answer: `(error) ${e?.message || "Request failed"}`, hits: [] });
      console.error("ask /api/ask error:", e);
    } finally {
      setBusy(false);
    }
  }

  function onKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      void ask();
    }
  }

  return (
    <div className="space-y-3">
      <textarea
        className="input"
        placeholder="e.g., Are sprinklers required on exterior balconies?"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={onKey}
      />
      <div className="flex items-center gap-3">
        <button className="btn" onClick={ask} disabled={busy}>
          {busy ? "Thinking…" : "Ask"}
        </button>
        <span className="text-zinc-400 text-sm">Tip: Cmd/Ctrl + Enter to submit</span>
      </div>
      {err && <div className="text-red-300 text-sm">{err}</div>}
    </div>
  );
}
