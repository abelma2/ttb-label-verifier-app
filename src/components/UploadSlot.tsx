"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ACCEPTED_IMAGE_TYPES } from "@/lib/image";

interface UploadSlotProps {
  id: string;
  label: string;
  hint: string;
  file: File | null;
  disabled: boolean;
  onSelect: (file: File) => void;
  onClear: () => void;
}

/** One drag-and-drop / click-to-browse image slot with a preview thumbnail. */
export default function UploadSlot({
  id,
  label,
  hint,
  file,
  disabled,
  onSelect,
  onClear,
}: UploadSlotProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [dragging, setDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [rejected, setRejected] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const accept = useCallback(
    (candidate: File | undefined) => {
      setRejected(null);
      if (!candidate) return;
      if (!ACCEPTED_IMAGE_TYPES.includes(candidate.type)) {
        setRejected("Please choose a PNG, JPEG, or WebP image.");
        return;
      }
      onSelect(candidate);
    },
    [onSelect],
  );

  /** A multi-file drop on a one-image slot: load the first usable image and SAY
   *  SO — which file is "first" depends on OS selection order, so silently
   *  discarding the rest could put the back label in the Front slot with the
   *  real front photo gone (the warning check then fails as "absent"). */
  const acceptDrop = useCallback(
    (list: FileList) => {
      const files = [...list];
      const first = files.find((f) => ACCEPTED_IMAGE_TYPES.includes(f.type)) ?? files[0];
      accept(first);
      if (files.length > 1 && first && ACCEPTED_IMAGE_TYPES.includes(first.type)) {
        const rest =
          files.length === 2
            ? "Drop the other file on its own slot"
            : "Drop the others on their own slots";
        setRejected(
          `${files.length} files were dropped, but this slot holds one image — only ` +
            `${first.name} was loaded. ${rest}, or use the Multiple labels tab for ` +
            "many products.",
        );
      }
    },
    [accept],
  );

  return (
    <div>
      <span className="mb-1.5 block text-sm font-medium text-slate-700" id={`${id}-label`}>
        {label}
      </span>
      {file ? (
        <div
          onDragOver={(e) => {
            // accept replacement drops on the filled card too — without this the
            // drop lands on an unclaimed element and only the window-level
            // DropGuard saves the session (the gesture would silently no-op)
            e.preventDefault();
            if (!disabled && !dialogRef.current?.open) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            // the open lightbox (and its backdrop, whose events are delivered
            // to the dialog element) is a child of this card, so a drop there
            // bubbles here — viewing is read-only and must not swap the file
            if (disabled || dialogRef.current?.open) return;
            if (e.dataTransfer.files?.length) acceptDrop(e.dataTransfer.files);
          }}
          className={`rounded-xl border bg-white p-3 transition-colors ${
            dragging ? "border-blue-500 bg-blue-50" : "border-slate-200"
          }`}
        >
          <div className="flex items-center gap-3">
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-slate-800">{file.name}</p>
              <p className="text-xs text-slate-500">
                {(file.size / 1024 / 1024).toFixed(1)} MB · click image for full screen
              </p>
            </div>
            <button
              type="button"
              onClick={() => {
                // a multi-drop notice ("only X was loaded") must not outlive
                // the file it describes — Remove empties the slot, so a
                // lingering alert would be asserting a load that no longer exists
                setRejected(null);
                onClear();
              }}
              disabled={disabled}
              className="rounded-lg px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-100 hover:text-red-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-600 disabled:opacity-50"
            >
              Remove
            </button>
          </div>
          {/* The preview renders large inline so the label is readable without
              clicking; the click is only for the full-screen view. Read-only, so
              it stays available even while a verify is running (unlike Remove,
              which would swap the inputs). */}
          <button
            type="button"
            onClick={() => dialogRef.current?.showModal()}
            aria-haspopup="dialog"
            aria-label={`View ${label.toLowerCase()} full screen: ${file.name}`}
            title="Click for full screen"
            className="group relative mt-2.5 block w-full cursor-zoom-in rounded-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-600"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl ?? undefined}
              alt=""
              className="max-h-[32rem] w-full rounded-lg border border-slate-100 bg-slate-50 object-contain"
            />
            <span
              aria-hidden="true"
              className="absolute right-2 top-2 flex items-center gap-1 rounded-md bg-slate-900/55 px-2 py-1 text-xs font-medium text-white opacity-0 transition group-hover:opacity-100 group-focus-visible:opacity-100"
            >
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 fill-none stroke-white" strokeWidth="2">
                <circle cx="11" cy="11" r="7" />
                <path strokeLinecap="round" d="m21 21-4.35-4.35M11 8v6M8 11h6" />
              </svg>
              Full screen
            </span>
          </button>
          {/* Native <dialog> for the lightbox: showModal() gives focus trapping,
              Escape-to-close, and focus restoration to the thumbnail for free. */}
          {previewUrl && (
            <dialog
              ref={dialogRef}
              aria-label={`Large preview of ${label.toLowerCase()}: ${file.name}`}
              onClick={(e) => {
                // a click on the ::backdrop is delivered to the dialog element
                // itself; clicks on the content hit children
                if (e.target === e.currentTarget) e.currentTarget.close();
              }}
              className="m-auto max-w-[94vw] rounded-2xl p-0 shadow-2xl backdrop:bg-slate-900/80"
            >
              <div className="flex flex-col">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={previewUrl}
                  alt={`Preview of ${label.toLowerCase()}: ${file.name}`}
                  className="max-h-[80vh] max-w-[92vw] bg-slate-50 object-contain"
                />
                <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 border-t border-slate-200 bg-white px-4 py-2.5">
                  <p className="min-w-0 flex-1 truncate text-sm font-medium text-slate-800">
                    {file.name}
                  </p>
                  <div className="flex items-center gap-2">
                    <a
                      href={previewUrl}
                      target="_blank"
                      rel="noopener"
                      className="rounded-lg px-2.5 py-1.5 text-xs font-medium text-blue-700 hover:bg-blue-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-600"
                    >
                      Open full size in new tab
                    </a>
                    <button
                      type="button"
                      onClick={() => dialogRef.current?.close()}
                      className="rounded-lg bg-slate-100 px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-600"
                    >
                      Close
                    </button>
                  </div>
                </div>
              </div>
            </dialog>
          )}
        </div>
      ) : (
        <button
          type="button"
          aria-labelledby={`${id}-label`}
          aria-describedby={`${id}-hint`}
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            // disabled suppresses click but NOT drag events — guard explicitly
            // so a drop mid-verify can't swap the image out from under a
            // result computed from the old one.
            e.preventDefault();
            if (!disabled) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            if (disabled) return;
            if (e.dataTransfer.files?.length) acceptDrop(e.dataTransfer.files);
          }}
          className={`flex w-full flex-col items-center justify-center gap-1 rounded-xl border-2 border-dashed px-4 py-8 text-center transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-600 disabled:opacity-50 ${
            dragging
              ? "border-blue-500 bg-blue-50"
              : "border-slate-300 bg-white hover:border-blue-400 hover:bg-slate-50"
          }`}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true" className="h-7 w-7 fill-none stroke-slate-400" strokeWidth="1.5">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
          </svg>
          <span className="text-sm font-medium text-slate-700">
            Drop an image here or <span className="text-blue-700">browse</span>
          </span>
          <span id={`${id}-hint`} className="text-xs text-slate-500">
            {hint}
          </span>
        </button>
      )}
      {/* Purely programmatic (the labeled dropzone button is the interaction
          point): aria-hidden keeps screen readers from finding an unnamed
          duplicate file control; tabIndex=-1 already removes it from the tab
          order, so hiding it is safe. */}
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_IMAGE_TYPES.join(",")}
        className="sr-only"
        aria-hidden="true"
        tabIndex={-1}
        onChange={(e) => {
          accept(e.target.files?.[0]);
          e.target.value = ""; // allow re-selecting the same file
        }}
      />
      {rejected && (
        <p role="alert" className="mt-1.5 text-xs font-medium text-red-700">
          {rejected}
        </p>
      )}
    </div>
  );
}
