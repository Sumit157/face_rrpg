#!/usr/bin/env node
/**
 * Cross-platform runner for the Python ML service's dev/test commands.
 *
 * Resolves the service's virtualenv (venv/ on Windows, venv/ on POSIX) and
 * runs the requested command from services/ml so relative paths inside the
 * service (models_assets, storage, ...) resolve correctly.
 *
 *   node scripts/ml.mjs dev     # uvicorn --reload on port 8000
 *   node scripts/ml.mjs test    # pytest tests/ -v
 */
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const mlDir = path.join(repoRoot, "services", "ml");

function resolveVenvPython() {
  const candidates = [
    path.join(mlDir, "venv", "Scripts", "python.exe"),
    path.join(mlDir, "venv", "bin", "python3"),
    path.join(mlDir, "venv", "bin", "python"),
  ];
  return candidates.find((p) => fs.existsSync(p));
}

const mode = process.argv[2] ?? "dev";
const python = resolveVenvPython();

if (!python) {
  console.error(
    "No virtualenv found at services/ml/venv.\n" +
      "Create one first:\n" +
      "  cd services/ml && python -m venv venv\n" +
      "  venv\\Scripts\\pip install -r requirements.txt   (Windows)\n" +
      "  venv/bin/pip install -r requirements.txt         (macOS/Linux)"
  );
  process.exit(1);
}

const commands = {
  dev: [python, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"],
  test: [python, "-m", "pytest", "tests/", "-v"],
};

const cmd = commands[mode];
if (!cmd) {
  console.error(`Unknown mode "${mode}". Use: ${Object.keys(commands).join(", ")}`);
  process.exit(1);
}

const [bin, ...args] = cmd;
const result = spawnSync(bin, args, { cwd: mlDir, stdio: "inherit", env: process.env });
process.exit(result.status ?? 1);
