import { Link } from "react-router-dom";
import { Layout } from "../components/layout/Layout";

export default function NotFound() {
  return (
    <Layout>
      <div className="max-w-md">
        <div className="font-mono text-accent text-sm tracking-widest mb-3">ERROR 404</div>
        <h1 className="page-title mb-3">This page doesn't exist</h1>
        <p className="text-sm text-slate-400 mb-6 leading-relaxed">
          The address may be mistyped, or the analysis it points to was deleted.
        </p>
        <Link to="/dashboard" className="btn-primary">
          Back to dashboard
        </Link>
      </div>
    </Layout>
  );
}
