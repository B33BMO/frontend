// components/AnswerBox.tsx
"use client";
import React from "react";

type Hit = { doc: string; page: number; text: string; score: number };

export default function AnswerBox({
  result,
  loading,
  error,
}: {
  result:
    | { answer: string; hits?: Hit[] }
    | { id: string; status: string; result?: { answer: string; hits?: Hit[] } }
    | null;
  loading: boolean;
  error: string | null;
}) {
  if (loading) {
    return (
      <div className="rounded-xl bg-neutral-900/50 p-4 border border-neutral-800">
        <div className="text-neutral-300">Thinking…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl bg-red-900/30 p-4 border border-red-700 text-red-200">
        {error}
      </div>
    );
  }

  if (!result) {
    return (
      <div className="rounded-xl bg-neutral-900/30 p-4 border border-neutral-800 text-neutral-400">
        (no answer yet)
      </div>
    );
  }

  // Normalize both shapes
  const normalized =
    "answer" in result
      ? { answer: result.answer, hits: (result as any).hits ?? [] }
      : {
          answer: result.result?.answer ?? "",
          hits: result.result?.hits ?? [],
        };

  return (
    <div className="rounded-xl bg-neutral-900/50 p-4 border border-neutral-800">
      <div className="whitespace-pre-wrap text-neutral-100">
        {normalized.answer || "(empty answer)"}
      </div>

      <div className="mt-4 text-sm text-neutral-400">Top supporting passages</div>
      {normalized.hits && normalized.hits.length > 0 ? (
        <ul className="mt-2 space-y-2">
          {normalized.hits.slice(0, 5).map((h: Hit, i: number) => (
            <li key={i} className="rounded-lg border border-neutral-800 p-3">
              <div className="text-xs text-neutral-400 mb-1">
                {h.doc} p.{h.page} • score {h.score}
              </div>
              <div className="text-neutral-200">{h.text}</div>
            </li>
          ))}
        </ul>
      ) : (
        <div className="mt-2 text-neutral-500 italic">No hits yet.</div>
      )}
    </div>
  );
}
