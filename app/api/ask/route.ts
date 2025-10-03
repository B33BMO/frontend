// app/api/ask/route.ts
// Proxies the frontend to your FastAPI backend (default http://127.0.0.1:8500).
// Works for both GET and POST. Add to your Next.js App Router.

export const dynamic = "force-dynamic"; // avoid caching
export const runtime = "nodejs";        // not 'edge' because we use custom fetch timeouts

const BACKEND_ORIGIN =
  process.env.BACKEND_ORIGIN || process.env.NEXT_PUBLIC_BACKEND_ORIGIN || "http://127.0.0.1:8500";

// If your FastAPI exposes /api/ask (recommended)
const ASK_PATH = "/api/ask";

// Simple fetch with timeout helper
async function fetchWithTimeout(resource: string, options: RequestInit & { timeout?: number } = {}) {
  const { timeout = 300000, ...opts } = options;
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    return await fetch(resource, { ...opts, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

export async function GET(req: Request) {
  try {
    const url = new URL(req.url);
    const q = url.searchParams.get("q");
    const k = url.searchParams.get("k") ?? "6";
    const sync = url.searchParams.get("sync") ?? "false";

    if (!q) {
      return new Response(JSON.stringify({ detail: "Missing query param: q" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }

    const backendURL = new URL(ASK_PATH, BACKEND_ORIGIN);
    backendURL.searchParams.set("q", q);
    backendURL.searchParams.set("k", k);
    backendURL.searchParams.set("sync", sync);

    const resp = await fetchWithTimeout(backendURL.toString(), {
      method: "GET",
      headers: { "accept": "application/json" },
      timeout: 30000,
    });

    const text = await resp.text();
    return new Response(text, {
      status: resp.status,
      headers: { "content-type": resp.headers.get("content-type") || "application/json" },
    });
  } catch (err: any) {
    const msg = err?.name === "AbortError" ? "Upstream timeout" : String(err);
    return new Response(JSON.stringify({ detail: `Proxy GET failed: ${msg}` }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }
}

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({}));
    const payload = {
      q: body.q ?? "",
      k: Number.isFinite(body.k) ? body.k : 6,
      sync: Boolean(body.sync ?? false),
    };

    if (!payload.q) {
      return new Response(JSON.stringify({ detail: "Missing body field: q" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }

    const backendURL = new URL(ASK_PATH, BACKEND_ORIGIN);

    payload.sync = true;                 // force synchronous answer
    payload.k = Number(payload.k) || 6;  // default k
    const resp = await fetchWithTimeout(backendURL.toString(), {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify(payload),     // <— no undefined vars now
      timeout: payload.sync ? 60000 : 30000,
    });


    const text = await resp.text();
    return new Response(text, {
      status: resp.status,
      headers: { "content-type": resp.headers.get("content-type") || "application/json" },
    });
  } catch (err: any) {
    const msg = err?.name === "AbortError" ? "Upstream timeout" : String(err);
    return new Response(JSON.stringify({ detail: `Proxy POST failed: ${msg}` }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }
}
