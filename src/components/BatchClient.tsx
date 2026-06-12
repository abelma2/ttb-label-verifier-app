"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { MAX_IMAGES_PER_PRODUCT, VerifyError } from "@/lib/api";
import {
  APP_FIELDS,
  appRowFor,
  parseApplicationsFile,
  type ParsedApplications,
} from "@/lib/applications";
import AppFileControls from "./AppFileControls";
import { errorShort, runBatch, type BatchItem, type BatchProgress } from "@/lib/batch";
import { ACCEPTED_IMAGE_TYPES } from "@/lib/image";
import { groupUploads } from "@/lib/stem";
import type { Status } from "@/lib/types";
import { ProductReport } from "./ResultsView";

type Phase = "idle" | "running" | "done";

type DetailFilter = "all" | Status | "error";

const STATUS_LABEL: Record<Status, string> = {
  pass: "Pass",
  needs_review: "Needs review",
  fail: "Fail",
};

const FILTER_ACTIVE: Record<DetailFilter, string> = {
  all: "border-blue-700 bg-blue-700 text-white",
  pass: "border-emerald-200 bg-pass-soft text-emerald-900",
  needs_review: "border-amber-200 bg-review-soft text-amber-900",
  fail: "border-red-200 bg-fail-soft text-red-900",
  error: "border-slate-300 bg-slate-100 text-slate-900",
};

/** Worst-first ordering for the results table. */
const RANK_ORDER: Record<string, number> = { fail: 0, error: 1, needs_review: 2, pass: 3 };

const PAGE_SIZE = 10;

function itemKey(item: BatchItem): string {
  return item.errorKind !== null || item.result === null ? "error" : item.result.overall;
}

function fileKey(f: File): string {
  return `${f.name}|${f.size}|${f.lastModified}`;
}

function DetailReport({
  item,
  product,
}: {
  item: BatchItem;
  product?: { label: string; files: File[] };
}) {
  const [images, setImages] = useState<{ url: string; alt: string }[]>([]);
  useEffect(() => {
    if (!product) {
      setImages([]);
      return;
    }
    const made = product.files.map((f) => ({
      url: URL.createObjectURL(f),
      alt: `Uploaded image ${f.name}`,
    }));
    setImages(made);
    return () => made.forEach((m) => URL.revokeObjectURL(m.url));
  }, [product]);
  return <ProductReport result={item.result!} elapsed={item.seconds} images={images} />;
}

export default function BatchClient() {
  const [files, setFiles] = useState<File[]>([]);
  const [groupPairs, setGroupPairs] = useState(true);
  const [apps, setApps] = useState<ParsedApplications | null>(null);
  const [appFileName, setAppFileName] = useState<string | null>(null);
  const [appFileReading, setAppFileReading] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState<BatchProgress | null>(null);
  const [items, setItems] = useState<BatchItem[] | null>(null);
  // snapshot of the products the run was computed from — detail thumbnails must
  // not drift when the upload list changes after a run (the staleness cue case)
  const [runProducts, setRunProducts] = useState<{ label: string; files: File[] }[]>([]);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [resultsSig, setResultsSig] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [detailFilter, setDetailFilter] = useState<DetailFilter>("all");
  const [announcement, setAnnouncement] = useState("");
  const [dragging, setDragging] = useState(false);
  const [rejectedNote, setRejectedNote] = useState<string | null>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const appInputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const appFileSeq = useRef(0);
  const resultsRef = useRef<HTMLDivElement>(null);

  const running = phase === "running";
  const mapping = apps?.error ? {} : (apps?.mapping ?? {});
  const products = useMemo(() => groupUploads(files, groupPairs), [files, groupPairs]);
  const rowFor = (label: string) => appRowFor(mapping, label);

  // staleness signature: results carry the inputs they were computed from.
  // Application ROW VALUES are part of it — a corrected file with the same
  // product stems still changes verdicts, so it must trip the banner.
  const currentSig = JSON.stringify({
    files: files.map((f) => fileKey(f)).sort(),
    group: groupPairs,
    apps: Object.entries(mapping)
      .map(([k, row]) => [k, ...APP_FIELDS.map((f) => row[f])])
      .sort(),
  });

  const stems = new Set(products.map((p) => p.label.toLowerCase()));
  const unusedAppRows = Object.keys(mapping).filter((p) => !stems.has(p)).sort();
  const oversizeProducts = products.filter((p) => p.files.length > MAX_IMAGES_PER_PRODUCT);

  function addFiles(list: FileList | File[] | null) {
    if (!list) return;
    const all = [...list];
    const incoming = all.filter((f) => ACCEPTED_IMAGE_TYPES.includes(f.type));
    const rejected = all.filter((f) => !ACCEPTED_IMAGE_TYPES.includes(f.type));
    // Surface silently-dropped files (HEIC, unknown type) — a vanished label is
    // a silent screening gap, so the operator must see it, not just an off count.
    setRejectedNote(
      rejected.length > 0
        ? `${rejected.length} file(s) skipped — not a PNG, JPEG, or WebP image ` +
            `(${rejected.slice(0, 4).map((f) => f.name).join(", ")}${rejected.length > 4 ? "…" : ""}). ` +
            "iPhone HEIC photos must be exported as JPEG first."
        : null,
    );
    setFiles((prev) => {
      const seen = new Set(prev.map(fileKey));
      return [...prev, ...incoming.filter((f) => !seen.has(fileKey(f)))];
    });
  }

  async function onAppFile(file: File | null) {
    if (!file) return;
    // last-write-wins: a slow parse of a since-replaced file must not overwrite
    // the newer selection's mapping (the banner would then credit the wrong file).
    const seq = ++appFileSeq.current;
    setAppFileReading(file.name);
    const parsed = await parseApplicationsFile(file);
    if (seq === appFileSeq.current) {
      // name and mapping commit together — the banner must never credit the
      // new file's name with the previous file's data (or error)
      setApps(parsed);
      setAppFileName(file.name);
      setAppFileReading(null);
    }
  }

  async function handleScreen() {
    if (products.length === 0 || running) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setPhase("running");
    // Do NOT clear prior results here: cancelling a just-started run must leave
    // the previous batch's report intact (it cost real API spend). Results are
    // hidden while running and replaced atomically only on success below.
    setProgress({ done: 0, total: products.length, current: "starting" });
    setAnnouncement(`Screening ${products.length} products.`);
    const t0 = performance.now();
    try {
      const results = await runBatch(products, rowFor, setProgress, controller.signal);
      // a late completion from a cancelled/superseded run must never clobber
      // the current run's state (Cancel -> re-Screen within the same window)
      if (abortRef.current !== controller) return;
      const secs = Math.round((performance.now() - t0) / 100) / 10;
      // snapshot the run's products + signature together with the results, so the
      // detail thumbnails and the staleness banner always match what was screened
      setRunProducts(products);
      setItems(results);
      setElapsed(secs);
      setResultsSig(currentSig);
      setPage(1);
      setDetailFilter("all");
      setPhase("done");
      const counts = { fail: 0, error: 0, needs_review: 0, pass: 0 };
      results.forEach((it) => (counts[itemKey(it) as keyof typeof counts] += 1));
      setAnnouncement(
        `Screening complete: ${counts.fail} fail, ${counts.needs_review} need review, ` +
          `${counts.pass} pass, ${counts.error} errors.`,
      );
      requestAnimationFrame(() => {
        // skip when this tab panel is hidden (display:none -> no layout box)
        if (resultsRef.current?.offsetParent != null) {
          resultsRef.current.scrollIntoView({ block: "start" });
          document.getElementById("batch-results-heading")?.focus({ preventScroll: true });
        }
      });
    } catch (err) {
      if (abortRef.current !== controller) return;
      if (err instanceof VerifyError && err.kind === "cancelled") {
        setPhase("idle");
        setAnnouncement("Screening cancelled.");
        return;
      }
      setPhase("idle");
      setAnnouncement("Screening failed unexpectedly. Please try again.");
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setProgress(null);
      }
    }
  }

  /** Stop the in-flight run only — leave every input and the previous results
   *  untouched (the button says "Cancel", not "Clear"). The run's catch sets
   *  phase back to idle. */
  function handleCancel() {
    abortRef.current?.abort();
  }

  function handleClear() {
    abortRef.current?.abort();
    abortRef.current = null; // close the same-tick window where a completing run resurrects state
    appFileSeq.current++; // an in-flight parse must not resurrect the cleared mapping
    setFiles([]);
    setApps(null);
    setAppFileName(null);
    setAppFileReading(null);
    setRejectedNote(null);
    setItems(null);
    setRunProducts([]);
    setElapsed(null);
    setResultsSig(null);
    setPage(1);
    setDetailFilter("all");
    setPhase("idle");
    if (imageInputRef.current) imageInputRef.current.value = "";
    if (appInputRef.current) appInputRef.current.value = "";
  }

  const ranked = items
    ? [...items.keys()].sort((a, b) => RANK_ORDER[itemKey(items[a])] - RANK_ORDER[itemKey(items[b])])
    : [];
  const counts = { fail: 0, error: 0, needs_review: 0, pass: 0 };
  items?.forEach((it) => (counts[itemKey(it) as keyof typeof counts] += 1));
  const overallTone =
    counts.fail || counts.error
      ? "border-red-200 bg-fail-soft text-red-900"
      : counts.needs_review
        ? "border-amber-200 bg-review-soft text-amber-900"
        : "border-emerald-200 bg-pass-soft text-emerald-900";
  const detailRanked =
    detailFilter === "all" ? ranked : ranked.filter((i) => itemKey(items![i]) === detailFilter);
  const nPages = Math.max(1, Math.ceil(detailRanked.length / PAGE_SIZE));
  const filterOptions: { key: DetailFilter; label: string; count: number }[] = items
    ? [
        { key: "all", label: "All", count: items.length },
        { key: "pass", label: STATUS_LABEL.pass, count: counts.pass },
        { key: "needs_review", label: STATUS_LABEL.needs_review, count: counts.needs_review },
        { key: "fail", label: STATUS_LABEL.fail, count: counts.fail },
        // Error is not a verdict — only offer the filter when a product actually errored.
        ...(counts.error > 0 ? [{ key: "error" as const, label: "Error", count: counts.error }] : []),
      ]
    : [];

  return (
    <div className="space-y-6">
      {!items && !running && (
        <div className="rounded-xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-900">
          <span className="font-semibold">How it works</span> — 1. Upload all the label
          photos. 2. Add the application data file if you have one. 3. Click{" "}
          <span className="font-semibold">Screen products</span>. Products without
          application data are screened against the fixed rules only.
        </div>
      )}

      <section
        aria-labelledby="batch-upload-heading"
        className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
      >
        <h2 id="batch-upload-heading" className="text-base font-semibold text-slate-900">
          Upload label photos
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          Name each product&apos;s photos with the same beginning plus _front / _back —
          e.g. <span className="font-medium">oldtom_front.jpg</span> and{" "}
          <span className="font-medium">oldtom_back.jpg</span> are read together as one
          product. Files named like IMG_1234.jpg are each treated as a separate product.
        </p>

        <button
          type="button"
          disabled={running}
          onClick={() => imageInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            if (!running) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            if (running) return;
            addFiles(e.dataTransfer.files);
          }}
          className={`mt-4 flex w-full flex-col items-center justify-center gap-1 rounded-xl border-2 border-dashed px-4 py-7 text-center transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-600 disabled:opacity-50 ${
            dragging
              ? "border-blue-500 bg-blue-50"
              : "border-slate-300 bg-white hover:border-blue-400 hover:bg-slate-50"
          }`}
        >
          <span className="text-sm font-medium text-slate-700">
            Drop label images here or <span className="text-blue-700">browse</span>
          </span>
          <span className="text-xs text-slate-500">
            PNG, JPEG, or WebP — add as many products as you like
          </span>
        </button>
        <input
          ref={imageInputRef}
          type="file"
          accept={ACCEPTED_IMAGE_TYPES.join(",")}
          multiple
          className="sr-only"
          aria-hidden="true"
          tabIndex={-1}
          onChange={(e) => {
            addFiles(e.target.files);
            e.target.value = "";
          }}
        />

        {rejectedNote && (
          <p role="alert" className="mt-2 rounded-lg bg-review-soft px-3 py-2 text-sm text-amber-900">
            {rejectedNote}
          </p>
        )}

        {files.length > 0 && (
          <ul className="mt-3 flex flex-wrap gap-2">
            {files.map((f, i) => (
              <li
                key={fileKey(f)}
                className="flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 py-1 pl-3 pr-1 text-xs text-slate-700"
              >
                {f.name}
                <button
                  type="button"
                  aria-label={`Remove ${f.name}`}
                  disabled={running}
                  onClick={() => setFiles((prev) => prev.filter((_, j) => j !== i))}
                  className="rounded-full px-1.5 py-0.5 font-semibold text-slate-500 hover:bg-slate-200 hover:text-red-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-600"
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}

        <label className="mt-4 flex items-center gap-2 text-sm text-slate-700">
          <input
            type="checkbox"
            checked={groupPairs}
            disabled={running}
            onChange={(e) => setGroupPairs(e.target.checked)}
            className="h-4 w-4 rounded border-slate-300 text-blue-700 focus:ring-blue-600"
          />
          Group front/back images of one product by filename
        </label>

        <div className="mt-5 border-t border-slate-100 pt-4">
          <p className="text-sm font-semibold text-slate-800">
            Application data <span className="font-normal text-slate-400">(optional)</span>
          </p>
          <p className="mt-0.5 text-xs text-slate-500">
            One small spreadsheet checks every label against its application values — one
            row per product.
          </p>
          <AppFileControls
            inputId="batch-app-file"
            inputRef={appInputRef}
            disabled={running}
            onFile={onAppFile}
          >
            <li>
              In the <span className="font-semibold">product</span> column, write the
              photos&apos; shared name without the ending — a row named{" "}
              <span className="font-semibold">oldtom</span> matches oldtom.jpg as well as
              oldtom_front.jpg + oldtom_back.jpg (capitals don&apos;t matter).
            </li>
            <li>
              The other columns are {APP_FIELDS.join(", ")} — fill in what you have; blank
              cells simply aren&apos;t checked.
            </li>
            <li>
              Products without a matching row are screened against the federal rules only.
            </li>
          </AppFileControls>
          {appFileReading && (
            <p className="mt-2.5 rounded-lg bg-slate-100 px-3 py-2 text-sm text-slate-600">
              Reading {appFileReading}…
            </p>
          )}
          {apps?.error && (
            <p role="alert" className="mt-2.5 rounded-lg bg-review-soft px-3 py-2 text-sm text-amber-900">
              Couldn&apos;t use the spreadsheet ({apps.error}) — all products will be
              screened against the federal rules only.
            </p>
          )}
          {apps && !apps.error && (
            <div className="mt-2.5 space-y-1.5">
              {apps.warnings.map((w, i) => (
                <p key={i} className="rounded-lg bg-review-soft px-3 py-2 text-sm text-amber-900">
                  Spreadsheet: {w}
                </p>
              ))}
              <p className="rounded-lg bg-pass-soft px-3 py-2 text-sm text-emerald-900">
                ✓ Loaded application data for {Object.keys(apps.mapping).length} product(s)
                {appFileName && <> from {appFileName}</>}.{" "}
                <button
                  type="button"
                  disabled={running}
                  onClick={() => {
                    appFileSeq.current++; // an in-flight parse must not resurrect the removed file
                    setApps(null);
                    setAppFileName(null);
                    setAppFileReading(null);
                    if (appInputRef.current) appInputRef.current.value = "";
                  }}
                  className="font-semibold underline underline-offset-2 hover:text-emerald-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-600"
                >
                  Remove file
                </button>
              </p>
            </div>
          )}
        </div>

        {products.length > 0 && (
          <div className="mt-4">
            <p className="text-sm text-slate-600">
              {products.length} product(s) from {files.length} file(s):
            </p>
            <div className="mt-2 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                    <th className="py-1.5 pr-4 font-semibold">Product</th>
                    <th className="py-1.5 pr-4 font-semibold">Files</th>
                    <th className="py-1.5 font-semibold">Application data</th>
                  </tr>
                </thead>
                <tbody>
                  {products.map((p, i) => (
                    <tr key={i} className="border-b border-slate-100 text-slate-700">
                      <td className="py-1.5 pr-4 font-medium">{p.label}</td>
                      <td className="py-1.5 pr-4">{p.files.map((f) => f.name).join(", ")}</td>
                      <td className="py-1.5">
                        {rowFor(p.label)
                          ? "matched"
                          : Object.hasOwn(mapping, p.label.toLowerCase())
                            ? "row found but empty"
                            : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {unusedAppRows.length > 0 && (
              <p className="mt-2 rounded-lg bg-review-soft px-3 py-2 text-sm text-amber-900">
                {unusedAppRows.length} application row(s) match no uploaded product:{" "}
                {unusedAppRows.slice(0, 8).join(", ")}
                {unusedAppRows.length > 8 && "…"} — check the &apos;product&apos; values if
                this is unexpected.
              </p>
            )}
            {oversizeProducts.length > 0 && (
              <p className="mt-2 rounded-lg bg-review-soft px-3 py-2 text-sm text-amber-900">
                {oversizeProducts.length} product(s) have more than {MAX_IMAGES_PER_PRODUCT}{" "}
                files and will fail to screen:{" "}
                {oversizeProducts.map((p) => `${p.label} (${p.files.length})`).join(", ")} —
                remove extra files or rename them to separate products.
              </p>
            )}
          </div>
        )}
      </section>

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={handleScreen}
          disabled={products.length === 0 || running}
          className="inline-flex items-center gap-2 rounded-xl bg-blue-700 px-6 py-3 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {running && (
            <svg viewBox="0 0 24 24" aria-hidden="true" className="h-4 w-4 animate-spin">
              <circle cx="12" cy="12" r="10" className="stroke-white/30" strokeWidth="4" fill="none" />
              <path d="M22 12a10 10 0 0 0-10-10" className="stroke-white" strokeWidth="4" fill="none" strokeLinecap="round" />
            </svg>
          )}
          {running
            ? "Screening…"
            : products.length > 0
              ? `Screen ${products.length} product(s)`
              : "Screen products"}
        </button>
        <button
          type="button"
          onClick={running ? handleCancel : handleClear}
          className="rounded-xl px-4 py-3 text-sm font-medium text-slate-600 hover:bg-slate-100 hover:text-slate-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-700"
        >
          {running ? "Cancel" : "Clear all"}
        </button>
      </div>

      <p aria-live="polite" role="status" className="sr-only">
        {announcement}
      </p>

      {running && progress && (
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={progress.total}
            aria-valuenow={progress.done}
            aria-label="Screening progress"
            className="h-2 w-full overflow-hidden rounded-full bg-slate-100"
          >
            <div
              className="h-full rounded-full bg-blue-600 transition-all"
              style={{ width: `${(progress.done / progress.total) * 100}%` }}
            />
          </div>
          <p className="mt-2 text-sm text-slate-600">
            Processed {progress.done}/{progress.total}
            {progress.done > 0 && <> — {progress.current}</>}
          </p>
        </div>
      )}

      <div ref={resultsRef} className="scroll-mt-6">
        {!running && items && (
          <section aria-label="Screening results" className="space-y-4">
            {resultsSig !== currentSig && (
              <p className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
                <span className="font-semibold">Inputs changed:</span> the results below are
                from a previous screening and may not match the files, grouping, or
                application data currently set above — click Screen to refresh, or Clear to
                start over.
              </p>
            )}
            <div className={`rounded-xl border p-5 ${overallTone}`}>
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h2 id="batch-results-heading" tabIndex={-1} className="text-lg font-bold outline-none">
                  Screened {items.length} product(s)
                </h2>
                <p className="text-sm font-medium opacity-80">
                  {counts.fail} fail · {counts.needs_review} needs review · {counts.pass} pass
                  · {counts.error} error{elapsed != null && <> · {elapsed}s total</>}
                </p>
              </div>
            </div>

            <div className="overflow-x-auto rounded-2xl border border-slate-200 bg-white shadow-sm">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                    <th className="px-4 py-2.5 font-semibold">Product</th>
                    <th className="px-4 py-2.5 font-semibold">Files</th>
                    <th className="px-4 py-2.5 font-semibold">Result</th>
                    <th className="px-4 py-2.5 font-semibold">Application data</th>
                    <th className="px-4 py-2.5 font-semibold">Flagged fields</th>
                  </tr>
                </thead>
                <tbody>
                  {ranked.map((i) => {
                    const item = items[i];
                    const isError = itemKey(item) === "error";
                    const flags =
                      item.result?.fields
                        .filter((f) => f.status !== "pass")
                        .map((f) => f.field.replace(/_/g, " ")) ?? [];
                    return (
                      <tr key={i} className="border-b border-slate-100 align-top text-slate-700">
                        <td className="px-4 py-2 font-medium">{item.label}</td>
                        <td className="px-4 py-2">{item.fileNames.join(", ")}</td>
                        <td className="px-4 py-2">
                          {isError ? "Error" : STATUS_LABEL[item.result!.overall]}
                        </td>
                        <td className="px-4 py-2">{item.matched ? "matched" : "—"}</td>
                        <td className="px-4 py-2">
                          {isError
                            ? errorShort(item.errorKind)
                            : flags.length > 0
                              ? flags.join(", ")
                              : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div>
              <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
                <h3 className="text-base font-semibold text-slate-900">Per-product detail</h3>
                <div className="flex flex-wrap items-center gap-3">
                  <div role="group" aria-label="Filter detail by result" className="flex flex-wrap gap-1.5">
                    {filterOptions.map((opt) => (
                      <button
                        key={opt.key}
                        type="button"
                        aria-pressed={detailFilter === opt.key}
                        disabled={opt.count === 0}
                        onClick={() => {
                          if (opt.key === detailFilter) return; // re-click must not yank pagination
                          setDetailFilter(opt.key);
                          setPage(1);
                          const noun = opt.count === 1 ? "product" : "products";
                          setAnnouncement(
                            opt.key === "all"
                              ? `Showing all ${opt.count} ${noun}.`
                              : `Showing ${opt.count} ${opt.label.toLowerCase()} ${noun}.`,
                          );
                        }}
                        className={`inline-flex items-center gap-1 rounded-full border px-3 py-1 text-xs font-semibold transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 disabled:cursor-not-allowed disabled:opacity-40 ${
                          detailFilter === opt.key
                            ? FILTER_ACTIVE[opt.key]
                            : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                        }`}
                      >
                        {detailFilter === opt.key && (
                          <svg viewBox="0 0 16 16" aria-hidden="true" className="h-3 w-3 fill-current">
                            <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-6.5 6.5a.75.75 0 0 1-1.06 0l-3-3a.75.75 0 1 1 1.06-1.06l2.47 2.47 5.97-5.97a.75.75 0 0 1 1.06 0Z" />
                          </svg>
                        )}
                        {opt.label} ({opt.count})
                      </button>
                    ))}
                  </div>
                  {nPages > 1 && (
                    <label className="flex items-center gap-2 text-sm text-slate-600">
                      Detail page
                      <select
                        value={page}
                        onChange={(e) => setPage(Number(e.target.value))}
                        className="rounded-lg border border-slate-300 px-2 py-1 text-sm"
                      >
                        {Array.from({ length: nPages }, (_, p) => (
                          <option key={p + 1} value={p + 1}>
                            Page {p + 1} of {nPages}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
                </div>
              </div>
              <div className="mt-3 space-y-3">
                {detailRanked.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE).map((i) => {
                  const item = items[i];
                  const isError = itemKey(item) === "error";
                  const status = isError ? "Error" : STATUS_LABEL[item.result!.overall];
                  const product = runProducts[i];
                  return (
                    <details
                      key={i}
                      className="group rounded-2xl border border-slate-200 bg-white shadow-sm"
                    >
                      <summary className="cursor-pointer select-none rounded-2xl px-4 py-3 text-sm font-medium text-slate-800 hover:bg-slate-50">
                        {item.label} — {status} ({item.fileNames.join(", ")})
                      </summary>
                      <div className="border-t border-slate-100 px-4 py-4">
                        {isError ? (
                          <p role="alert" className="rounded-lg bg-fail-soft px-3 py-2 text-sm text-red-900">
                            {item.errorMessage}
                          </p>
                        ) : (
                          <DetailReport item={item} product={product} />
                        )}
                      </div>
                    </details>
                  );
                })}
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
