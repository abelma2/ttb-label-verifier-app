"use client";

import { useRef, useState } from "react";
import type { ApplicationData, VerifyResponse } from "@/lib/types";
import { cleanApplication, verifyLabel, VerifyError } from "@/lib/api";
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
  const resultsRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

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
      abortRef.current = null;
    }
  }

  function handleReset() {
    abortRef.current?.abort();
    setFront(null);
    setBack(null);
    setApplication({});
    setResult(null);
    setError(null);
    setPhase("idle");
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
