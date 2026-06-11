/**
 * Client-side image preparation for the 4.5 MB Vercel request-body limit.
 *
 * Anything we send beyond ~2048 px is wasted anyway: the vision pipeline's
 * "high detail" mode downscales images to fit 2048 px before reading them, so
 * resizing client-side loses nothing the model would have seen. Large files
 * are drawn to a canvas at <= 2048 px on the long side and re-encoded as JPEG.
 *
 * This is best-effort: on any decode/encode failure we fall back to the
 * original file and let the API's hard limits produce a clear error message.
 */

export const ACCEPTED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"];

/** Above this size we downscale/re-encode. Sized so the MAX of 4 images per
 *  product, each passed through untouched at this limit, still fits under the
 *  ~4.3 MB total body cap (4 x 1.0 MB = 4.0 MB) — a 3-4 image product whose
 *  files are individually small must not blow the total uncompressed. */
const COMPRESS_THRESHOLD_BYTES = 1_000_000;
/** Matches the model's high-detail input cap. */
const MAX_DIMENSION = 2048;
const JPEG_QUALITY = 0.9;

export interface PreparedImage {
  blob: Blob;
  /** Original filename, with the extension fixed up if we re-encoded. */
  name: string;
}

export async function prepareImage(file: File): Promise<PreparedImage> {
  if (file.size <= COMPRESS_THRESHOLD_BYTES) {
    return { blob: file, name: file.name };
  }
  try {
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(1, MAX_DIMENSION / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("canvas 2d context unavailable");
    // JPEG has no alpha channel and a canvas is transparent by default, so a
    // transparent label render (common for artwork proofs) would composite onto
    // BLACK and could blank out dark text. Paint white first — matching how
    // labels are physically printed.
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY),
    );
    if (!blob || blob.size >= file.size) {
      return { blob: file, name: file.name };
    }
    return { blob, name: file.name.replace(/\.\w+$/, "") + ".jpg" };
  } catch {
    return { blob: file, name: file.name };
  }
}
