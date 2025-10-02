export default function AnswerBox({ answer }: { answer: string }) {
  return (
    <div className="card">
      <pre className="whitespace-pre-wrap font-mono text-[14px]">{answer || "No answer."}</pre>
    </div>
  );
}
