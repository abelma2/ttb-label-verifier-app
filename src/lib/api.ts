import type { ApplicationData, VerifyResponse } from "./types";
import { APPLICATION_KEYS } from "./types";
import { prepareImage } from "./image";

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

export async function verifyLabel(
  front: File,
  back: File | null,
  application: ApplicationData | null,
): Promise<VerifyResponse> {
  const images = [await prepareImage(front)];
  if (back) images.push(await prepareImage(back));

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

  let res: Response;
  try {
    res = await fetch("/api/py/verify", { method: "POST", body: form });
  } catch {
    throw new VerifyError(
      "network",
      "Could not reach the verification service. Check your connection and try again.",
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
      // Non-JSON error (e.g. the platform's own 413) — keep the generic message.
      if (res.status === 413) {
        kind = "payload_too_large";
        message = "The upload was rejected as too large. Please use smaller images.";
      }
    }
    throw new VerifyError(kind, message, res.status);
  }

  return (await res.json()) as VerifyResponse;
}
