// components/AskForm.tsx
"use client";
import React from "react";

export default function AskForm({
  onResult,
  onLoading,
  onError,
}: {
  onResult: (data: any) => void;
  onLoading: (val: boolean) => void;
  onError: (msg: string | null) => void;
}) {
  const [q, setQ] = React.useState("");

  async function submit() {
    onError(null);
    if (!q.trim()) {
      onError("Please enter a question.");
      return;
    }

    onLoading(true);
    try {
      const resp = await fetch("/api/ask", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ q, k: 6, sync: true }),
      });

      const text = await resp.text();
      let data: any = {};
      try { data = JSON.parse(text); } catch { /* leave as empty */ }

      if (!resp.ok) {
        const detail = data?.detail || `HTTP ${resp.status}`;
        throw new Error(detail);
      }

      onResult(data);
    } catch (e: any) {
      onError(e?.message || "Request failed");
    } finally {
      onLoading(false);
    }
  }

  return (
    <div className="flex gap-3">
      <textarea
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Ask a question…"
        className="flex-1 rounded-xl bg-neutral-900/50 border border-neutral-800 p-3 text-neutral-100 min-h-[100px]"
      />
      <button
        onClick={submit}
        className="h-[48px] self-end px-4 rounded-lg bg-neutral-200 text-black font-medium hover:bg-white"
      >
        Ask
      </button>
    </div>
  );
}
