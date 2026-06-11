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

  return (
    <div>
      <span className="mb-1.5 block text-sm font-medium text-slate-700" id={`${id}-label`}>
        {label}
      </span>
      {file ? (
        <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white p-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={previewUrl ?? undefined}
            alt={`Preview of ${label.toLowerCase()}: ${file.name}`}
            className="h-20 w-20 shrink-0 rounded-lg border border-slate-100 object-cover"
          />
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-slate-800">{file.name}</p>
            <p className="text-xs text-slate-500">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
          </div>
          <button
            type="button"
            onClick={onClear}
            disabled={disabled}
            className="rounded-lg px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-100 hover:text-red-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-600 disabled:opacity-50"
          >
            Remove
          </button>
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
            accept(e.dataTransfer.files?.[0]);
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
