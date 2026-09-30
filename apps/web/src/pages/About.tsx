import { Layout } from "../components/layout/Layout";

export default function About() {
  return (
    <Layout>
      <h1 className="page-title mb-6">About</h1>
      <div className="panel p-6 space-y-4 text-sm text-slate-300 leading-relaxed">
        <p>
          This tool analyses remote photoplethysmography (rPPG) signals extracted from facial
          video as an additional, explainable cue in deepfake detection research. It measures
          whether a video exhibits physiologically plausible, spatially consistent blood-flow
          signal across the face -- something many (but not all) synthesis pipelines fail to
          reproduce faithfully.
        </p>
        <p className="text-slate-400">
          <strong className="text-slate-200">This is not a standalone deepfake detector.</strong>{" "}
          A REAL verdict here means the physiological signal was plausible and consistent, not
          that the video is confirmed authentic. A FAKE or UNCERTAIN verdict is evidence to weigh
          alongside other forensic methods, not a final determination. Signal quality is always
          reported alongside any verdict, and low-quality signals are explicitly flagged as
          unreliable rather than silently producing a confident-looking answer.
        </p>
        <p className="text-slate-400">
          See <code className="text-accent">docs/limitations.md</code> in the project repository
          for a full discussion of known limitations, and{" "}
          <code className="text-accent">docs/methodology.md</code> for the underlying methods
          (Green channel, CHROM, POS) and their literature sources.
        </p>
      </div>
    </Layout>
  );
}
