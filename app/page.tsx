// app/page.tsx
"use client";
import React from "react";
import AskForm from "@/components/AskForm";
import AnswerBox from "@/components/AnswerBox";

export default function Page() {
  const [result, setResult] = React.useState<any>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  return (
    <main className="max-w-3xl mx-auto p-6 space-y-6">
      <h1 className="text-3xl font-semibold">NFPA 13 / 13R — Q&amp;A</h1>
      <p className="text-neutral-400">
        Answers strictly from the two PDFs. Citations open the exact page.
      </p>

      <AskForm
        onResult={(d) => setResult(d)}
        onLoading={setLoading}
        onError={setError}
      />

      <AnswerBox result={result} loading={loading} error={error} />

      <div className="text-center text-neutral-500 text-sm">
        Sources: NFPA 13-2022 &amp; PCI NFPA 13R • Served via FastAPI proxy
      </div>
    </main>
  );
}
