import type { FieldVerdict, Status, VerifyResponse } from "@/lib/types";
import { fieldLabel } from "@/lib/types";
import { JsonReadout, OtherLabelDetails, WarningEvidence } from "./EvidencePanels";
import StatusPill from "./StatusPill";

const BANNER: Record<Status, { title: string; className: string }> = {
  pass: {
    title: "No issues found",
    className: "border-emerald-200 bg-pass-soft text-emerald-900",
  },
  needs_review: {
    title: "Needs human review",
    className: "border-amber-200 bg-review-soft text-amber-900",
  },
  fail: {
    title: "Failed verification",
    className: "border-red-200 bg-fail-soft text-red-900",
  },
};

/** Focus target after results render (keyboard/screen-reader users land here). */
export const RESULTS_HEADING_ID = "results-heading";

const LONG_TEXT_THRESHOLD = 160;

function ValueBlock({ heading, value }: { heading: string; value: string }) {
  if (!value) {
    return (
      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">{heading}</p>
        <p className="mt-0.5 text-sm text-slate-400">—</p>
      </div>
    );
  }
  const long = value.length > LONG_TEXT_THRESHOLD;
  return (
    <div className="min-w-0">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">{heading}</p>
      {long ? (
        <details>
          <summary className="mt-0.5 cursor-pointer text-sm text-slate-700 hover:text-blue-700">
            {value.slice(0, LONG_TEXT_THRESHOLD)}…{" "}
            <span className="text-xs font-medium text-blue-700">show all</span>
          </summary>
          <p className="mt-1 whitespace-pre-wrap break-words text-sm text-slate-700">{value}</p>
        </details>
      ) : (
        <p className="mt-0.5 break-words text-sm text-slate-700">{value}</p>
      )}
    </div>
  );
}

function FieldCard({ verdict, showExpected }: { verdict: FieldVerdict; showExpected: boolean }) {
  return (
    <li className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-900">{fieldLabel(verdict.field)}</h3>
        <StatusPill status={verdict.status} />
      </div>
      <p className="mt-2 text-sm leading-5 text-slate-600">{verdict.reason}</p>
      <div className={`mt-3 grid gap-3 ${showExpected ? "sm:grid-cols-2" : ""}`}>
        <ValueBlock heading="Read from the label" value={verdict.extracted} />
        {showExpected && <ValueBlock heading="Expected" value={verdict.expected} />}
      </div>
    </li>
  );
}

export interface ProductReportProps {
  result: VerifyResponse;
  /** id for the focusable banner heading (single mode's focus target) */
  headingId?: string;
  /** preview object-URLs for the product's uploaded images (batch detail) */
  images?: { url: string; alt: string }[];
  /** per-product read time in seconds (batch detail) */
  elapsed?: number | null;
  /** include the "checked against …" mode sentence (single mode) */
  showModeSentence?: boolean;
}

/** One product's full verification report: verdict banner, field cards, photo
 *  note, warning-absence hint, and the evidence expanders — the web port of
 *  app.py's _render_product. */
export function ProductReport({
  result,
  headingId,
  images,
  elapsed,
  showModeSentence = false,
}: ProductReportProps) {
  const banner = BANNER[result.overall];
  const counts = result.fields.reduce(
    (acc, f) => ({ ...acc, [f.status]: (acc[f.status] ?? 0) + 1 }),
    {} as Partial<Record<Status, number>>,
  );
  const summary = [
    counts.fail ? `${counts.fail} fail` : null,
    counts.needs_review ? `${counts.needs_review} need${counts.needs_review === 1 ? "s" : ""} review` : null,
    counts.pass ? `${counts.pass} pass` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  // a missing warning is usually a back label that wasn't uploaded; branch on
  // the machine-readable cause, never the display reason (which may be reworded)
  const warning = result.fields.find((f) => f.field === "government_warning");
  const warningAbsent = warning?.status === "fail" && warning.cause === "absence";

  return (
    <section aria-label="Verification results">
      <div className={`rounded-xl border p-5 ${banner.className}`}>
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id={headingId} tabIndex={headingId ? -1 : undefined} className="text-lg font-bold outline-none">
            {banner.title}
          </h2>
          <p className="text-sm font-medium opacity-80">
            {result.fields.length} checks · {summary}
            {elapsed != null && <> · {elapsed.toFixed(1)}s</>}
          </p>
        </div>
        <p className="mt-1 text-sm opacity-80">
          {showModeSentence &&
            (result.mode === "application_match"
              ? "Checked against the federal labeling rules and the submitted application values."
              : "Rules-only screening — no application data was provided, so only the fixed federal rules and mandatory-field presence were checked.")}
          {result.beverage_type !== "unknown" && (
            <> Read as a <span className="font-semibold">{result.beverage_type}</span> label.</>
          )}
        </p>
      </div>

      {result.image_quality_notes && (
        <p className="mt-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
          <span className="font-semibold">Photo note:</span> {result.image_quality_notes}
        </p>
      )}
      {warningAbsent && (
        <p className="mt-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
          No government warning was found in the image(s). It is usually on the back/other
          label — include that image too if you have it.
        </p>
      )}

      {images && images.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {images.map((img) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={img.url}
              src={img.url}
              alt={img.alt}
              className="h-28 rounded-lg border border-slate-200 object-contain"
            />
          ))}
        </div>
      )}

      <ul className="mt-4 space-y-3">
        {result.fields.map((f) => (
          <FieldCard
            key={f.field}
            verdict={f}
            showExpected={result.mode === "application_match" || f.field === "government_warning"}
          />
        ))}
      </ul>

      <div className="mt-4 space-y-3">
        <WarningEvidence extracted={result.extracted} />
        <OtherLabelDetails
          extracted={result.extracted}
          additionalStatements={result.additional_statements}
          imageQualityNotes={result.image_quality_notes}
        />
        <JsonReadout extracted={result.extracted} />
      </div>
    </section>
  );
}

/** Single-label mode's results view. */
export default function ResultsView({ result }: { result: VerifyResponse }) {
  return <ProductReport result={result} headingId={RESULTS_HEADING_ID} showModeSentence />;
}
