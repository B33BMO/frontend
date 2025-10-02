// components/HitCard.tsx
import { sourceUrl, type Hit as HitType } from "@/lib/pdfSources";

export default function HitCard({ hit }: { hit: HitType }) {
  return (
    <div className="border border-white/20 rounded-xl p-3 mt-2 bg-white/5">
      <div className="text-zinc-400 text-sm mb-1">
        <a
          className="text-blue-300 hover:underline"
          href={sourceUrl(hit.doc, hit.page)}
          target="_blank"
          rel="noopener noreferrer"
        >
          {hit.doc} — p.{hit.page}
        </a>
        <span className="ml-2 opacity-60">score {hit.score.toFixed(3)}</span>
      </div>
      <div className="text-sm">{hit.text}</div>
    </div>
  );
}
