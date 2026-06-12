// Explicit .ts extensions: see batch.ts — these modules also run under plain
// Node for the unit tests.
import type { ApplicationData, VerifyResponse } from "./types.ts";
import { APPLICATION_KEYS } from "./types.ts";
import { prepareImage } from "./image.ts";

/** Mirrors the API's hard total-body cap so oversize uploads fail client-side
 *  with a friendly message instead of a platform-level 413. */
const MAX_TOTAL_BYTES = 4.3 * 1024 * 1024;

export class VerifyError extends Error {
  /** Machine-readable kind: the API's ErrorBody.kind, or a client-side kind
   *  ("network" | "payload_too_large"). */
  readonly kind: string;
  readonly status: number | null;

  constructor(kind: string, message: string, status: number | null = null) {
    super(message);
    this.kind = kind;
    this.status = status;
  }
}

/** Trimmed, non-empty application values; null when the form is entirely blank
 *  (=> the API runs rules-only screening — never auto-filled). */
export function cleanApplication(values: ApplicationData): ApplicationData | null {
  const out: ApplicationData = {};
  let any = false;
  for (const key of APPLICATION_KEYS) {
    const v = values[key]?.trim();
    if (v) {
      out[key] = v;
      any = true;
    }
  }
  return any ? out : null;
}

/** Client-side deadline, slightly above the server's ceilings (engine request
 *  timeout 30 s, function maxDuration 60 s) so it can never mask a real 504. */
const FETCH_TIMEOUT_MS = 75_000;

function deadlineSignal(external?: AbortSignal): AbortSignal | undefined {
  if (typeof AbortSignal.timeout === "function" && typeof AbortSignal.any === "function") {
    const timeout = AbortSignal.timeout(FETCH_TIMEOUT_MS);
    return external ? AbortSignal.any([external, timeout]) : timeout;
  }
  // Fallback for browsers without AbortSignal.any (Safari <17.4 / Firefox <124):
  // a manual controller so the deadline is ALWAYS enforced — never silently
  // dropped just because we also have an external signal to honor. (A benign
  // timer survives a completed request; it just aborts an already-settled one.)
  const controller = new AbortController();
  setTimeout(
    () => controller.abort(new DOMException("The operation timed out.", "TimeoutError")),
    FETCH_TIMEOUT_MS,
  );
  if (external) {
    if (external.aborted) controller.abort(external.reason);
    else external.addEventListener("abort", () => controller.abort(external.reason), { once: true });
  }
  return controller.signal;
}

/** Mirrors the API's MAX_IMAGES so an oversized stem group fails client-side
 *  with a friendly message instead of a 400. */
export const MAX_IMAGES_PER_PRODUCT = 4;

export async function verifyLabel(
  front: File,
  back: File | null,
  application: ApplicationData | null,
  signal?: AbortSignal,
): Promise<VerifyResponse> {
  return verifyImages(back ? [front, back] : [front], application, signal);
}

/** Verify ONE product from 1–4 label images (front/back/neck/strip), read
 *  together as one label. Used directly by batch mode, where each product is
 *  its own request (= its own serverless invocation). */
export async function verifyImages(
  files: File[],
  application: ApplicationData | null,
  signal?: AbortSignal,
): Promise<VerifyResponse> {
  if (files.length === 0) {
    throw new VerifyError("no_images", "Upload at least one label image.");
  }
  if (files.length > MAX_IMAGES_PER_PRODUCT) {
    throw new VerifyError(
      "too_many_images",
      `This product has ${files.length} files; the limit is ${MAX_IMAGES_PER_PRODUCT} images ` +
        "per product (front, back/other, neck/strip).",
    );
  }
  const images = await Promise.all(files.map(prepareImage));

  const total = images.reduce((sum, img) => sum + img.blob.size, 0);
  if (total > MAX_TOTAL_BYTES) {
    throw new VerifyError(
      "payload_too_large",
      "The images are too large to upload even after compression. " +
        "Please use images around 2000 px on the long side.",
    );
  }

  const form = new FormData();
  for (const img of images) form.append("images", img.blob, img.name);
  if (application) form.append("application", JSON.stringify(application));

  /** The signal also aborts body streaming, so both the fetch AND the body
   *  read must map DOMExceptions to typed VerifyErrors — callers' control flow
   *  depends on every rejection being a VerifyError. */
  function classifyDomException(err: unknown): VerifyError | null {
    if (err instanceof DOMException && err.name === "TimeoutError") {
      return new VerifyError(
        "timeout",
        "Reading the label took too long and was cancelled. Try again, or upload a smaller image.",
      );
    }
    if (err instanceof DOMException && err.name === "AbortError") {
      return new VerifyError("cancelled", "Verification was cancelled.");
    }
    return null;
  }

  let res: Response;
  try {
    res = await fetch("/api/py/verify", {
      method: "POST",
      body: form,
      signal: deadlineSignal(signal),
    });
  } catch (err) {
    throw (
      classifyDomException(err) ??
      new VerifyError(
        "network",
        "Could not reach the verification service. Check your connection and try again.",
      )
    );
  }

  if (!res.ok) {
    let kind = "unknown";
    let message = `Verification failed (HTTP ${res.status}).`;
    try {
      const body = await res.json();
      if (body?.error?.message) {
        kind = body.error.kind ?? kind;
        message = body.error.message;
      }
    } catch {
      // Non-JSON error: a platform-level page (Vercel 413/502/503/504), not our
      // JSON envelope. Map the status to the right kind so the batch table shows
      // an accurate cause instead of blaming the photo.
      if (res.status === 413) {
        kind = "payload_too_large";
        message = "The upload was rejected as too large. Please use smaller images.";
      } else if (res.status === 504) {
        kind = "timeout";
        message = "Reading the label took too long and was cancelled. Try again, or use a smaller image.";
      } else if (res.status === 502 || res.status === 503) {
        kind = "connection";
        message = "The verification service is temporarily unavailable. Wait a moment and try again.";
      }
    }
    throw new VerifyError(kind, message, res.status);
  }

  try {
    return (await res.json()) as VerifyResponse;
  } catch (err) {
    throw (
      classifyDomException(err) ??
      new VerifyError("bad_response", "The verification service returned an unreadable reply.")
    );
  }
}
