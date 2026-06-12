/**
 * Batch screening runner. Where the retired Streamlit prototype (dev-archive
 * branch) fanned a batch out over a server-side ThreadPoolExecutor, here
 * each product is one API request (= one serverless invocation) issued from
 * the browser with a small concurrency cap. Results are keyed by index, not
 * label — distinct products can share a stem — and one failed product becomes
 * an error item, never a sunk batch.
 */
// Explicit .ts extensions: these modules also run under plain Node for the
// unit tests (node --experimental-strip-types), whose ESM loader does not
// infer extensions the way the Next bundler does.
import { verifyImages, VerifyError } from "./api.ts";
import type { AppRow } from "./applications.ts";
import type { Product } from "./stem.ts";
import type { VerifyResponse } from "./types.ts";

/** Concurrent in-flight requests — 8, matching the engine's
 *  config.BATCH_MAX_WORKERS (a 10-product batch is 2 waves, not 3). The account's
 *  rate limits leave ~50x headroom at this rate, and the engine's 429 backoff
 *  remains as a safety net for smaller accounts. In local dev the browser's
 *  ~6-connection HTTP/1.1 cap queues the extras harmlessly; on Vercel (HTTP/2)
 *  all 8 run in parallel as separate function invocations. */
export const BATCH_CONCURRENCY = 8;

/** Short error labels for the results table (ported from the prototype,
 *  extended with the client-side kinds). Full messages show in the detail view. */
export const ERROR_SHORT: Record<string, string> = {
  auth: "service not set up",
  quota: "service out of credits",
  rate_limit: "service busy — try again",
  timeout: "service timeout — try again",
  connection: "no connection to service",
  network: "no connection to service",
  bad_response: "bad service reply — try again",
  payload_too_large: "images too large",
  too_many_images: "too many files for one product",
  // API-side upload-validation kinds — never blame the photo READ for an
  // upload-size/format problem (the project's error-attribution principle)
  file_too_large: "an image is too large",
  unsupported_type: "not a supported image",
  empty_file: "an image file is empty",
  invalid_application: "bad application data",
  invalid_request: "bad request — try again",
  unknown: "service error — try again",
};

export function errorShort(kind: string | null): string {
  // Object.hasOwn, not `kind in`/ERROR_SHORT[kind]: a server- or platform-
  // supplied kind like "constructor" would otherwise resolve to a Function on
  // the prototype chain and crash the table (rendered as a React child). A
  // neutral fallback — never "could not read image", which would blame the
  // photo for a server/transport failure (the error-attribution principle).
  if (kind && Object.hasOwn(ERROR_SHORT, kind)) return ERROR_SHORT[kind];
  return "verification failed — try again";
}

export interface BatchItem {
  label: string;
  fileNames: string[];
  /** null when this product errored */
  result: VerifyResponse | null;
  seconds: number | null;
  /** true when a non-blank application row was matched (verify() ran) */
  matched: boolean;
  errorKind: string | null;
  errorMessage: string | null;
}

export interface BatchProgress {
  done: number;
  total: number;
  current: string;
}

/** Run every product through the verify API with bounded concurrency.
 *  Aborting the signal stops scheduling and rejects with VerifyError("cancelled").
 *  `verifyFn` is injectable for tests (mirroring how the API tests monkeypatch
 *  extract_fields); production always uses the real verifyImages. */
export async function runBatch(
  products: Product<File>[],
  applicationFor: (label: string) => AppRow | null,
  onProgress: (p: BatchProgress) => void,
  signal?: AbortSignal,
  verifyFn: typeof verifyImages = verifyImages,
): Promise<BatchItem[]> {
  const items: BatchItem[] = new Array(products.length);
  let next = 0;
  let done = 0;

  async function worker(): Promise<void> {
    while (next < products.length) {
      if (signal?.aborted) throw new VerifyError("cancelled", "Screening was cancelled.");
      const idx = next++;
      const { label, files } = products[idx];
      const base = { label, fileNames: files.map((f) => f.name) };
      const application = applicationFor(label);
      const t0 = performance.now();
      try {
        const result = await verifyFn(files, application, signal);
        items[idx] = {
          ...base,
          result,
          seconds: Math.round((performance.now() - t0) / 100) / 10,
          matched: application !== null,
          errorKind: null,
          errorMessage: null,
        };
      } catch (err) {
        if (err instanceof VerifyError && err.kind === "cancelled") throw err;
        // one bad product must not sink the batch
        items[idx] = {
          ...base,
          result: null,
          seconds: null,
          matched: application !== null,
          errorKind: err instanceof VerifyError ? err.kind : "unknown",
          errorMessage:
            err instanceof VerifyError
              ? err.message
              : "Verification failed unexpectedly. Try again.",
        };
      }
      done += 1;
      onProgress({ done, total: products.length, current: label });
    }
  }

  const workers = Array.from(
    { length: Math.min(BATCH_CONCURRENCY, products.length) },
    () => worker(),
  );
  await Promise.all(workers);
  return items;
}
