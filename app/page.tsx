"use client";

import { useState } from "react";
import AskForm from "@/components/AskForm";
import AnswerBox from "@/components/AnswerBox";
import HitCard from "@/components/HitCard";
import type { AskResponse, Hit as HitType } from "@/lib/pdfSources";

export default function Page() {
  const [answer, setAnswer] = useState<string>("");
  const [hits, setHits] = useState<HitType[]>([]);

  function handleResult(data: AskResponse) {
    setAnswer(data.answer || "");
    setHits(data.hits || []);
  }

  return (
    <main className="space-y-4">
      <div className="card">
        <h1 className="text-[22px] font-semibold tracking-wide">
          NFPA 13 / 13R — Q&A
        </h1>
        <p className="text-zinc-400 text-sm mt-1">
          Answers strictly from the two PDFs. Citations open the exact page.
        </p>

        <div className="mt-4">
          <AskForm onResult={handleResult} />
        </div>

        {answer && (
          <div className="mt-4">
            <AnswerBox answer={answer} />
          </div>
        )}

        {!!hits.length && (
          <div className="mt-4">
            <h3 className="text-zinc-400 text-sm font-semibold mb-2">
              Top supporting passages
            </h3>
            {hits.map((h, i) => (
              <HitCard key={i} hit={h} />
            ))}
          </div>
        )}
      </div>

      <div className="text-center text-zinc-500 text-xs">
        Sources: NFPA 13-2022 &amp; PCI NFPA 13R • Served via FastAPI proxy
      </div>
    </main>
  );
}
