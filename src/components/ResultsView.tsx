import type { FieldVerdict, Status, VerifyResponse } from "@/lib/types";
import { fieldLabel } from "@/lib/types";
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

export default function ResultsView({ result }: { result: VerifyResponse }) {
  const banner = BANNER[result.overall];
  const counts = result.fields.reduce(
    (acc, f) => ({ ...acc, [f.status]: (acc[f.status] ?? 0) + 1 }),
    {} as Partial<Record<Status, number>>,
  );
  const summary = [
    counts.pass ? `${counts.pass} pass` : null,
    counts.needs_review ? `${counts.needs_review} need${counts.needs_review === 1 ? "s" : ""} review` : null,
    counts.fail ? `${counts.fail} fail` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <section aria-label="Verification results">
      <div className={`rounded-xl border p-5 ${banner.className}`}>
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id={RESULTS_HEADING_ID} tabIndex={-1} className="text-lg font-bold outline-none">
            {banner.title}
          </h2>
          <p className="text-sm font-medium opacity-80">
            {result.fields.length} checks · {summary}
          </p>
        </div>
        <p className="mt-1 text-sm opacity-80">
          {result.mode === "application_match"
            ? "Checked against the federal labeling rules and the submitted application values."
            : "Rules-only screening — no application data was provided, so only the fixed federal rules and mandatory-field presence were checked."}
          {result.beverage_type !== "unknown" && (
            <> Read as a <span className="font-semibold">{result.beverage_type}</span> label.</>
          )}
        </p>
      </div>

      <ul className="mt-4 space-y-3">
        {result.fields.map((f) => (
          <FieldCard
            key={f.field}
            verdict={f}
            showExpected={result.mode === "application_match" || f.field === "government_warning"}
          />
        ))}
      </ul>

      {result.additional_statements.length > 0 && (
        <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
          <h3 className="text-sm font-semibold text-slate-900">
            Additional statements on the label
          </h3>
          <p className="mt-1 text-xs text-slate-500">
            Conditional disclosures transcribed for the reviewer (no automated pass/fail —
            their triggers aren&apos;t visible on the label).
          </p>
          <ul className="mt-2 space-y-1.5">
            {result.additional_statements.map((s, i) => (
              <li key={i} className="text-sm text-slate-700">
                “{s.value}”
                {s.kind && <span className="ml-2 text-xs text-slate-500">({s.kind})</span>}
              </li>
            ))}
          </ul>
        </div>
      )}

      {result.image_quality_notes && (
        <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
          <h3 className="text-sm font-semibold text-slate-900">Image quality</h3>
          <p className="mt-1 text-sm text-slate-600">{result.image_quality_notes}</p>
        </div>
      )}
    </section>
  );
}
