interface Props {
  src: string;
  overlay?: React.ReactNode;
}

export function VideoPlayer({ src, overlay }: Props) {
  return (
    <div className="relative panel overflow-hidden">
      <video src={src} controls className="w-full h-auto block" />
      {overlay && <div className="absolute inset-0 pointer-events-none">{overlay}</div>}
    </div>
  );
}
