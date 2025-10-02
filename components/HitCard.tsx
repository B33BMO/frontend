import { sourceUrl, type Hit as HitType } from "@/lib/pdfSources";

type Props = { hit?: HitType | null };

export default function HitCard({ hit }: Props) {
  // Runtime guard: if something weird comes in, render nothing
  if (!hit || !hit.doc || typeof hit.page !== "number") return null;

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
        {Number.isFinite(hit.score) && (
          <span className="ml-2 opacity-60">score {hit.score.toFixed(3)}</span>
        )}
      </div>
      <div className="text-sm">{hit.text}</div>
    </div>
  );
}
