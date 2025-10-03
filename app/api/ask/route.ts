// app/api/ask/route.ts
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const BACKEND_ORIGIN =
  process.env.BACKEND_ORIGIN ||
  process.env.NEXT_PUBLIC_BACKEND_ORIGIN ||
  "http://127.0.0.1:8500";

const ASK_PATH = "/api/ask";

async function fetchWithTimeout(resource: string, options: RequestInit & { timeout?: number } = {}) {
  const { timeout = 60000, ...opts } = options;
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    return await fetch(resource, { ...opts, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({}));
    const payload = {
      q: body?.q ?? "",
      k: Number.isFinite(body?.k) ? body.k : 6,
      sync: true,      // ← force sync so UI gets the full answer in one call
    };

    if (!payload.q) {
      return new Response(JSON.stringify({ detail: "Missing body field: q" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }

    const backendURL = new URL(ASK_PATH, BACKEND_ORIGIN);
    const resp = await fetchWithTimeout(backendURL.toString(), {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify(payload),
      timeout: 60000,
    });

    const text = await resp.text();
    return new Response(text, {
      status: resp.status,
      headers: {
        "content-type": resp.headers.get("content-type") || "application/json",
        "cache-control": "no-store",
      },
    });
  } catch (err: any) {
    const msg = err?.name === "AbortError" ? "Upstream timeout" : String(err);
    return new Response(JSON.stringify({ detail: `Proxy POST failed: ${msg}` }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }
}
