import { useCallback, useRef, useState } from "react";
import { useUploadVideo } from "../../hooks/useVideo";

interface Props {
  onUploaded: (videoId: string) => void;
}

const ACCEPTED = ".mp4,.avi,.mov,.mkv,.webm";

export function VideoUploader({ onUploaded }: Props) {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const upload = useUploadVideo();

  const handleFile = useCallback(
    (file: File) => {
      upload.mutate(file, {
        onSuccess: (video) => onUploaded(video.id),
      });
    },
    [upload, onUploaded]
  );

  return (
    <div
      className={`panel border-dashed p-10 text-center transition-all duration-200 ${
        dragActive
          ? "border-accent bg-accent/5 shadow-glow"
          : "border-surface-700 hover:border-accent/40"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragActive(true);
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragActive(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
      }}
    >
      <svg
        viewBox="0 0 48 48"
        className="mx-auto h-10 w-10 mb-4 text-accent"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <path d="M24 32V12m0 0-7 7m7-7 7 7" />
        <path d="M8 30v4a4 4 0 0 0 4 4h24a4 4 0 0 0 4-4v-4" />
      </svg>

      <p className="text-slate-200 mb-1">Drag a video here, or</p>
      <button
        type="button"
        className="btn-ghost"
        onClick={() => inputRef.current?.click()}
        disabled={upload.isPending}
      >
        {upload.isPending ? "Uploading..." : "Select a file"}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
      <p className="font-mono text-[11px] text-slate-600 mt-4 tracking-wide">
        MP4 · AVI · MOV · MKV · WEBM
      </p>

      {upload.isError && (
        <p className="text-verdict-fake text-sm mt-4" role="alert">
          Upload failed: {axiosErrorMessage(upload.error) ?? "Please try again."}
        </p>
      )}
    </div>
  );
}

function axiosErrorMessage(error: unknown): string | null {
  if (typeof error !== "object" || error === null || !("response" in error)) return null;
  const response = (error as { response?: { data?: { error?: string } } }).response;
  return response?.data?.error ?? null;
}
