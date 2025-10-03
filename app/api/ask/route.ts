// app/api/ask/route.ts
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

// Use your public domain by default so this works in prod behind nginx/Cloudflare.
// You can override locally via NEXT_PUBLIC_BACKEND_ORIGIN or BACKEND_ORIGIN.
const BACKEND_ORIGIN =
  process.env.BACKEND_ORIGIN ||
  process.env.NEXT_PUBLIC_BACKEND_ORIGIN ||
  "https://nfpa.bmo.guru";

const ASK_PATH = "/api/ask";

// Fetch with abort timeout
async function fetchWithTimeout(
  resource: string,
  options: RequestInit & { timeout?: number } = {}
) {
  const { timeout = 75000, ...opts } = options; // 75s >= backend ASK_TIMEOUT (70s)
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    return await fetch(resource, { ...opts, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

// Always POST to backend (even for GET requests to this proxy) so server.py works.
export async function GET(req: Request) {
  try {
    const url = new URL(req.url);
    const q = (url.searchParams.get("q") || "").trim();
    const k = Number(url.searchParams.get("k") || "6");

    if (!q) {
      return new Response(JSON.stringify({ detail: "Missing query param: q" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }

    const backendURL = new URL(ASK_PATH, BACKEND_ORIGIN);
    const resp = await fetchWithTimeout(backendURL.toString(), {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify({ q, k, sync: true }), // force sync for single-call UX
      timeout: 75000,
    });

    const text = await resp.text();
    return new Response(text, {
      status: resp.status,
      headers: {
        "content-type": resp.headers.get("content-type") || "application/json",
      },
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
    const body = await req.json().catch(() => ({} as any));
    const q = (body?.q || "").trim();
    const k = Number.isFinite(body?.k) ? Number(body.k) : 6;

    if (!q) {
      return new Response(JSON.stringify({ detail: "Missing body field: q" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }

    const backendURL = new URL(ASK_PATH, BACKEND_ORIGIN);
    const resp = await fetchWithTimeout(backendURL.toString(), {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify({ q, k, sync: true }), // force sync
      timeout: 75000,
    });

    const text = await resp.text();
    return new Response(text, {
      status: resp.status,
      headers: {
        "content-type": resp.headers.get("content-type") || "application/json",
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
