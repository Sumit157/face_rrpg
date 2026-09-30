import { createBrowserRouter, Navigate } from "react-router-dom";
import Dashboard from "../pages/Dashboard";
import NewAnalysis from "../pages/NewAnalysis";
import Analysis from "../pages/Analysis";
import SignalExplorer from "../pages/SignalExplorer";
import ModelEvaluation from "../pages/ModelEvaluation";
import Reports from "../pages/Reports";
import LiveAnalysis from "../pages/LiveAnalysis";
import About from "../pages/About";
import NotFound from "../pages/NotFound";

export const router = createBrowserRouter([
  { path: "/", element: <Navigate to="/dashboard" replace /> },
  { path: "/dashboard", element: <Dashboard /> },
  { path: "/analysis/new", element: <NewAnalysis /> },
  { path: "/analysis/:id", element: <Analysis /> },
  { path: "/signal-explorer", element: <SignalExplorer /> },
  { path: "/models", element: <ModelEvaluation /> },
  { path: "/reports", element: <Reports /> },
  { path: "/live", element: <LiveAnalysis /> },
  { path: "/about", element: <About /> },
  { path: "*", element: <NotFound /> },
]);
