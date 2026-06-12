"use client";

import { useState, type ReactNode, type RefObject } from "react";
import { downloadApplicationsTemplate, EXCEL_ACCEPT } from "@/lib/applications";

/**
 * The two actions for the optional application spreadsheet — download the
 * blank template, load the filled-in file — plus the naming rules collapsed
 * behind a "How the spreadsheet works" toggle. Shared by single mode (prefill)
 * and batch mode so the flow reads identically in both: two obvious buttons,
 * no naked file input, no wall of instructions.
 */
export default function AppFileControls({
  inputId,
  inputRef,
  disabled,
  onFile,
  children,
}: {
  inputId: string;
  inputRef: RefObject<HTMLInputElement | null>;
  disabled: boolean;
  onFile: (file: File | null) => void;
  /** The "How the spreadsheet works" bullet list — pass <li> items. */
  children: ReactNode;
}) {
  const [templateNote, setTemplateNote] = useState<string | null>(null);

  const buttonClass =
    "inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 " +
    "py-2 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:border-blue-400 " +
    "hover:bg-blue-50 hover:text-blue-800 focus-visible:outline focus-visible:outline-2 " +
    "focus-visible:outline-offset-1 focus-visible:outline-blue-600 disabled:cursor-not-allowed " +
    "disabled:opacity-50";

  return (
    <div>
      <div className="mt-3 flex flex-wrap gap-2">
        <button
          type="button"
          disabled={disabled}
          onClick={() => {
            setTemplateNote(null);
            downloadApplicationsTemplate().catch(() =>
              setTemplateNote(
                "Could not generate the template — check your connection and try again.",
              ),
            );
          }}
          className={buttonClass}
        >
          <svg viewBox="0 0 20 20" aria-hidden="true" className="h-4 w-4 fill-current">
            <path d="M10.75 2.75a.75.75 0 0 0-1.5 0v8.614L6.295 8.235a.75.75 0 1 0-1.09 1.03l4.25 4.5a.75.75 0 0 0 1.09 0l4.25-4.5a.75.75 0 0 0-1.09-1.03l-2.955 3.129V2.75Z" />
            <path d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z" />
          </svg>
          Download blank template
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
          className={buttonClass}
        >
          <svg viewBox="0 0 20 20" aria-hidden="true" className="h-4 w-4 fill-current">
            <path d="M9.25 13.25a.75.75 0 0 0 1.5 0V4.636l2.955 3.129a.75.75 0 0 0 1.09-1.03l-4.25-4.5a.75.75 0 0 0-1.09 0l-4.25 4.5a.75.75 0 1 0 1.09 1.03L9.25 4.636v8.614Z" />
            <path d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z" />
          </svg>
          Load filled-in spreadsheet
        </button>
      </div>
      <input
        id={inputId}
        ref={inputRef}
        type="file"
        accept={EXCEL_ACCEPT}
        className="sr-only"
        aria-hidden="true"
        tabIndex={-1}
        onChange={(e) => {
          onFile(e.target.files?.[0] ?? null);
          e.target.value = ""; // allow re-selecting the same (edited) file
        }}
      />
      {templateNote && (
        <p role="alert" className="mt-2 rounded-lg bg-review-soft px-3 py-2 text-sm text-amber-900">
          {templateNote}
        </p>
      )}
      <details className="group mt-2.5">
        <summary className="cursor-pointer select-none text-xs font-medium text-slate-500 hover:text-slate-700">
          <span className="underline decoration-dotted underline-offset-2">
            How the spreadsheet works
          </span>
        </summary>
        <ul className="mt-1.5 list-disc space-y-1 pl-5 text-xs leading-relaxed text-slate-600">
          {children}
        </ul>
      </details>
    </div>
  );
}
