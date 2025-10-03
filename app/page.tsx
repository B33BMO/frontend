// page.tsx
"use client";

import { useState } from "react";
import AskForm from "@/components/AskForm";
import AnswerBox from "@/components/AnswerBox";
import HitCard from "@/components/HitCard";
import type { AskResponse, Hit as HitType } from "@/lib/pdfSources";

export default function Page() {
  const [answer, setAnswer] = useState<string>("");   // keep string
  const [hits, setHits] = useState<HitType[]>([]);

  function handleResult(data: AskResponse) {
    setAnswer(data.answer ?? "");
    setHits(Array.isArray(data.hits) ? data.hits : []);
  }

  return (
    <main className="space-y-4">
      <div className="card">
        <h1 className="text-[22px] font-semibold tracking-wide">
          NFPA 13 / 13R — Q&amp;A
        </h1>
        <p className="text-zinc-400 text-sm mt-1">
          Answers strictly from the two PDFs. Citations open the exact page.
        </p>

        <div className="mt-4">
          <AskForm onResult={handleResult} />
        </div>

        {/* Always show an answer box so you see errors/empty text too */}
        <div className="mt-4">
          <AnswerBox answer={answer} />
        </div>

        <div className="mt-4">
          <h3 className="text-zinc-400 text-sm font-semibold mb-2">
            Top supporting passages
          </h3>
          {hits.length === 0 ? (
            <div className="text-zinc-500 text-sm italic">No hits yet.</div>
          ) : (
            hits.map((h, i) => <HitCard key={i} hit={h} />)
          )}
        </div>
      </div>

      <div className="text-center text-zinc-500 text-xs">
        Sources: NFPA 13-2022 &amp; PCI NFPA 13R • Served via FastAPI proxy
      </div>
    </main>
  );
}
