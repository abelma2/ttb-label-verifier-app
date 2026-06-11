import type { NextConfig } from "next";

/**
 * /api/py/* is the Python (FastAPI) surface:
 *  - dev:  proxy to the local uvicorn server, so the browser stays same-origin
 *          (no CORS) and `next dev` + `npm run dev:api` compose cleanly.
 *  - prod: rewrite to /api/ — on Vercel this falls through Next to the
 *          api/index.py serverless function, which receives the ORIGINAL
 *          /api/py/... path and routes it inside FastAPI. (The same pattern as
 *          Vercel's Next.js + FastAPI hybrid template.)
 */
const nextConfig: NextConfig = {
  rewrites: async () => [
    {
      source: "/api/py/:path*",
      destination:
        process.env.NODE_ENV === "development"
          ? "http://127.0.0.1:8000/api/py/:path*"
          : "/api/",
    },
  ],
};

export default nextConfig;
