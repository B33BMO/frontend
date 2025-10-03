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
        // your route.ts forces sync=true on the backend
        body: JSON.stringify({ q: q.trim(), k: 8 }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const raw = await res.json();

      // Normalize both shapes:
      //  - JobStatus envelope: { result: { answer, hits } }
      //  - Flat legacy:       { answer, hits }
      const normalized: AskResponse = {
        answer: raw?.result?.answer ?? raw?.answer ?? "",
        hits: Array.isArray(raw?.result?.hits)
          ? raw.result.hits
          : Array.isArray(raw?.hits)
          ? raw.hits
          : [],
      };

      onResult(normalized);
    } catch (e: any) {
      setErr(e?.message || "Something exploded 🤷");
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
