"use client";

import { useEffect, useRef, useState } from "react";
import type { ApplicationData, VerifyResponse } from "@/lib/types";
import { cleanApplication, verifyLabel, VerifyError } from "@/lib/api";
import { parseApplications, pickApplicationRow, type ParsedApplications } from "@/lib/applications";
import { stem } from "@/lib/stem";
import ApplicationForm from "./ApplicationForm";
import ResultsView, { RESULTS_HEADING_ID } from "./ResultsView";
import UploadSlot from "./UploadSlot";

type Phase = "idle" | "verifying" | "done" | "error";

const OVERALL_ANNOUNCEMENT: Record<string, string> = {
  pass: "no issues found",
  needs_review: "needs human review",
  fail: "failed verification",
};

export default function VerifierClient() {
  const [front, setFront] = useState<File | null>(null);
  const [back, setBack] = useState<File | null>(null);
  const [application, setApplication] = useState<ApplicationData>({});
  const [phase, setPhase] = useState<Phase>("idle");
  const [result, setResult] = useState<VerifyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [announcement, setAnnouncement] = useState("");
  const [prefill, setPrefill] = useState<{ name: string; parsed: ParsedApplications } | null>(null);
  const [prefillMessage, setPrefillMessage] = useState<string | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const prefillInputRef = useRef<HTMLInputElement>(null);

  // Re-run the prefill match whenever the parsed file OR the front image
  // changes (mirrors app.py's `file_id|stem` gate — fixed there after a review
  // finding: matching only at file-selection time left the form permanently
  // unfilled when the file arrived before the image, and silently kept product
  // A's values after the image was swapped to product B).
  useEffect(() => {
    if (!prefill) return;
    const { mapping, error, warnings } = prefill.parsed;
    if (error) {
      setPrefillMessage(`Could not use the application file (${error}).`);
      return;
    }
    const { row, message } = pickApplicationRow(mapping, front ? stem(front.name) : null);
    if (row) {
      setApplication((prev) => ({ ...prev, ...row }));
    }
    const warningText = warnings.length > 0 ? ` ${warnings.join("; ")}.` : "";
    setPrefillMessage(`Application file: ${message}.${warningText}`);
  }, [prefill, front]);

  const verifying = phase === "verifying";
  const appValues = cleanApplication(application);

  async function handleVerify() {
    if (!front || verifying) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setPhase("verifying");
    setError(null);
    setResult(null);
    setAnnouncement("Reading the label, this typically takes five to ten seconds.");
    try {
      const response = await verifyLabel(front, back, appValues, controller.signal);
      // a late completion after Start over / a newer verify must not clobber state
      if (abortRef.current !== controller) return;
      setResult(response);
      setPhase("done");
      setAnnouncement(
        `Verification complete: ${OVERALL_ANNOUNCEMENT[response.overall] ?? response.overall}. ` +
          `${response.fields.length} fields checked.`,
      );
      requestAnimationFrame(() => {
        resultsRef.current?.scrollIntoView({ block: "start" });
        document.getElementById(RESULTS_HEADING_ID)?.focus({ preventScroll: true });
      });
    } catch (err) {
      if (abortRef.current !== controller) return;
      if (err instanceof VerifyError && err.kind === "cancelled") {
        setPhase("idle");
        setAnnouncement("Verification cancelled.");
        return;
      }
      const message =
        err instanceof VerifyError
          ? err.message
          : "Something went wrong while verifying the label. Please try again.";
      setError(message);
      setPhase("error");
      setAnnouncement(`Verification failed: ${message}`);
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
    }
  }

  function handleReset() {
    abortRef.current?.abort();
    abortRef.current = null;
    setFront(null);
    setBack(null);
    setApplication({});
    setResult(null);
    setError(null);
    setPrefill(null);
    setPrefillMessage(null);
    setPhase("idle");
    if (prefillInputRef.current) prefillInputRef.current.value = "";
  }

  /** Parse an application file (same format as the batch file) and keep it in
   *  state — the match itself runs in the effect above so it re-attempts when
   *  the front image changes. Values stay editable after prefill — the file is
   *  the applicant's submission, so this remains an independent comparison. */
  async function handlePrefillFile(file: File | null) {
    if (!file) return;
    setPrefill({ name: file.name, parsed: parseApplications(await file.text(), file.name) });
  }

  return (
    <div className="space-y-6">
      <section
        aria-labelledby="images-heading"
        className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
      >
        <h2 id="images-heading" className="text-base font-semibold text-slate-900">
          1. Label images
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          One product per check. Add the back label too when the government warning or net
          contents live there — both images are read together as one label.
        </p>
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <UploadSlot
            id="front"
            label="Front label (required)"
            hint="PNG, JPEG, or WebP — large photos are resized in your browser"
            file={front}
            disabled={verifying}
            onSelect={setFront}
            onClear={() => setFront(null)}
          />
          <UploadSlot
            id="back"
            label="Back / other label (optional)"
            hint="Usually carries the government warning"
            file={back}
            disabled={verifying}
            onSelect={setBack}
            onClear={() => setBack(null)}
          />
        </div>
      </section>

      <section
        aria-labelledby="application-heading"
        className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id="application-heading" className="text-base font-semibold text-slate-900">
            2. Application values <span className="font-normal text-slate-400">(optional)</span>
          </h2>
          <span
            className={`rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset ${
              appValues
                ? "bg-blue-50 text-blue-800 ring-blue-600/20"
                : "bg-slate-100 text-slate-600 ring-slate-500/20"
            }`}
          >
            {appValues ? "Will match label vs. application" : "Will screen rules-only"}
          </span>
        </div>
        <p className="mt-1 text-sm text-slate-500">
          Type the values from the application to verify the label against them. Leave everything
          blank to screen against the fixed federal rules only — the form is never auto-filled
          from the label, so it stays an independent check.
        </p>
        <div className="mt-3">
          <label htmlFor="single-app-file" className="block text-sm font-medium text-slate-700">
            Prefill from application file{" "}
            <span className="font-normal text-slate-400">(optional, CSV or JSON)</span>
          </label>
          <p className="mt-0.5 text-xs text-slate-500">
            Same format as the batch application file; the row whose &apos;product&apos; value
            matches the front image&apos;s filename stem is used.
          </p>
          <input
            id="single-app-file"
            ref={prefillInputRef}
            type="file"
            accept=".csv,.json"
            disabled={verifying}
            onChange={(e) => {
              handlePrefillFile(e.target.files?.[0] ?? null);
              e.target.value = ""; // allow re-selecting the same (edited) file
            }}
            className="mt-1.5 block w-full text-sm text-slate-600 file:mr-3 file:rounded-lg file:border-0 file:bg-blue-50 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-blue-700 hover:file:bg-blue-100"
          />
          {prefillMessage && <p className="mt-1.5 text-xs text-slate-600">{prefillMessage}</p>}
        </div>
        <div className="mt-4">
          <ApplicationForm values={application} disabled={verifying} onChange={setApplication} />
        </div>
      </section>

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={handleVerify}
          disabled={!front || verifying}
          className="inline-flex items-center gap-2 rounded-xl bg-blue-700 px-6 py-3 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {verifying && (
            <svg viewBox="0 0 24 24" aria-hidden="true" className="h-4 w-4 animate-spin">
              <circle cx="12" cy="12" r="10" className="stroke-white/30" strokeWidth="4" fill="none" />
              <path d="M22 12a10 10 0 0 0-10-10" className="stroke-white" strokeWidth="4" fill="none" strokeLinecap="round" />
            </svg>
          )}
          {verifying ? "Reading label…" : "Verify label"}
        </button>
        <button
          type="button"
          onClick={handleReset}
          className="rounded-xl px-4 py-3 text-sm font-medium text-slate-600 hover:bg-slate-100 hover:text-slate-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-700"
        >
          {verifying ? "Cancel" : "Start over"}
        </button>
        {!front && <p className="text-sm text-slate-500">Add a front label image to begin.</p>}
      </div>

      {/* Small dedicated live region: announces one-line progress/outcome only.
          The full results render OUTSIDE it so screen readers aren't read the
          entire report as one announcement. */}
      <p aria-live="polite" role="status" className="sr-only">
        {announcement}
      </p>

      <div ref={resultsRef} className="scroll-mt-6">
        {verifying && (
          <div className="rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm">
            <p className="text-sm font-medium text-slate-700">
              Reading the label with the vision model…
            </p>
            <p className="mt-1 text-xs text-slate-500">
              Typically 5–10 seconds. The model transcribes the label; deterministic rules then
              judge each field.
            </p>
          </div>
        )}

        {phase === "error" && error && (
          <div role="alert" className="rounded-2xl border border-red-200 bg-fail-soft p-5">
            <h2 className="text-sm font-bold text-red-900">Verification failed</h2>
            <p className="mt-1 text-sm text-red-800">{error}</p>
          </div>
        )}

        {phase === "done" && result && <ResultsView result={result} />}
      </div>
    </div>
  );
}
