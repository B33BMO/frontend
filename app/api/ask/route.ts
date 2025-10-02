import { NextResponse } from "next/server";

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}));
  const q = (body?.q || "").toString();
  const k = Number.isFinite(body?.k) ? Number(body.k) : 8;

  if (!q || q.length < 3) {
    return NextResponse.json({ error: "Query too short" }, { status: 400 });
  }

  const fastapi = process.env.FASTAPI_URL || "http://127.0.0.1:8000";
  const url = new URL("/ask", fastapi);
  url.searchParams.set("q", q);
  url.searchParams.set("k", String(k));

  const resp = await fetch(url.toString(), { method: "GET" });
  if (!resp.ok) {
    return NextResponse.json({ error: `Upstream HTTP ${resp.status}` }, { status: 502 });
  }
  const data = await resp.json();
  return NextResponse.json(data, { status: 200 });
}
