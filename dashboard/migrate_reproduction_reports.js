/* Add explicit reproduction provenance to reports exported before schema v2. */
"use strict";

const fs = require("fs");
const path = require("path");

const projectRoot = path.resolve(__dirname, "..");
const experimentRoot = path.join(__dirname, "experiments", "geometry_lab");
const batchAssumption = "The paper does not specify batch size. This value is a declared dashboard reproduction assumption and must be included in sensitivity analysis.";
const datasetProtocol = {
  factor_domain: "[0, 1]^2",
  stretch_factor: 2,
  stretched_factor: 1,
  embedding: "R^2 -> R^3 with an appended zero coordinate",
  rotation_degrees: 45,
  rotation_axis: [1, -1, 1],
  transformation_order: ["stretch", "embed", "rotate"],
  paper_location: "Supplement B.5",
};

function readJson(file, fallback = {}) {
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : fallback;
}

function updateReport(file) {
  let report = fs.readFileSync(file, "utf8");
  const originalReport = report;
  const match = report.match(/Experiment ID: `([^`]+)`/);
  if (!match) return false;
  const experimentId = match[1];
  const directory = path.join(experimentRoot, experimentId);
  const config = readJson(path.join(directory, "config.json"));
  const training = readJson(path.join(directory, "training_config.json"));
  const summary = readJson(path.join(directory, "summary.json"));
  const counts = summary.split_counts || config.split_counts || {};
  const paperLinear = config.generator_type === "paper_linear" || config.dataset_source === "paper_linear";
  if (!paperLinear) return false;

  if (!report.includes("## Dataset construction protocol")) {
    report = report.replace(
      /\| Train \/ validation \/ test \| ([^\r\n]+) \|/,
      "| Nominal train / validation / test fractions | $1 |\n" +
        `| Exact train / validation / test samples | ${counts.train ?? "not available"} / ${counts.validation ?? "not available"} / ${counts.test ?? "not available"} |`,
    );

    const warning = counts.test === 5000
      ? ""
      : `\nLegacy split warning: this persisted dataset predates exact integer allocation. Its paper metrics used ${counts.test ?? "an unknown number of"} test samples; regenerate, retrain, and reanalyze it before treating it as a corrected reproduction.\n`;
    const construction = [
      warning,
      "## Dataset construction protocol",
      "",
      "| Field | Effective value |",
      "|---|---|",
      "| Factor domain | `[0, 1]^2` |",
      "| Stretch | factor 1 multiplied by 2 |",
      "| Embedding | `R^2 -> R^3 with an appended zero coordinate` |",
      "| Rotation | 45 degrees about axis `[1, -1, 1]` |",
      "| Transformation order | `stretch -> embed -> rotate` |",
      "| Paper location | `Supplement B.5` |",
      "",
    ].join("\n");
    report = report.replace("## Effective model and training configuration", construction + "\n## Effective model and training configuration");

    report = report.replace(
      /(\| Batch size \| `[^`]+` \|)/,
      `$1\n| Batch size provenance | \`declared_reproduction_assumption\` |\n| Layer biases enabled | \`${training.bias ?? true}\` |`,
    );
    report = report.replace(
      /(\| Training seed \| `[^`]+` \|)/,
      "$1\n| Training seed provenance | `effective_training_configuration` |",
    );
    report = report.replace(
      "\n## Final training results",
      `\n\nBatch-size assumption: ${batchAssumption}\nThe selected value is an experimental assumption, not a value attributed to the paper.\n\n## Final training results`,
    );
  }

  const protocol = {...(config.paper_protocol || {})};
  const reportedTrainingSeed = report.match(/\| Training seed \| `(-?\d+)` \|/);
  const effective = {
    // A copied report is an immutable result snapshot and may predate a later
    // retraining of the same experiment directory. Its own effective-value
    // table is therefore authoritative for the report's result provenance.
    seed: reportedTrainingSeed ? Number(reportedTrainingSeed[1]) : Number(training.seed),
    seed_source: "effective_training_configuration",
    batch_size: Number(training.batch_size),
    batch_size_source: "declared_reproduction_assumption",
    batch_size_assumption: batchAssumption,
    bias: training.bias ?? true,
  };
  Object.assign(protocol, effective);
  if (protocol.training && typeof protocol.training === "object") {
    protocol.training = {...protocol.training, ...effective};
    delete protocol.training.bias;
  }
  if (protocol.model && typeof protocol.model === "object") protocol.model = {...protocol.model, bias: effective.bias};
  const protocolSection = [
    "## Resolved paper protocol",
    "",
    "This protocol is resolved against the effective training configuration, so its seed, batch size, and bias fields describe the run that produced the checkpoint.",
    "",
    "```json",
    JSON.stringify(protocol, null, 2),
    "```",
    "",
  ].join("\n");
  report = report.replace(/## (?:Declared|Resolved) paper protocol\s+(?:This protocol[^\r\n]*\s+)?```json[\s\S]*?```\s+(?=## Interpretation note)/, protocolSection);
  if (report === originalReport) return false;
  fs.writeFileSync(file, report, "utf8");
  return true;
}

const reportFiles = [];
const reportsDirectory = path.join(projectRoot, "reports");
if (fs.existsSync(reportsDirectory)) {
  for (const name of fs.readdirSync(reportsDirectory)) {
    if (name.endsWith("_reproduction_report.md")) reportFiles.push(path.join(reportsDirectory, name));
  }
}
if (fs.existsSync(experimentRoot)) {
  for (const name of fs.readdirSync(experimentRoot)) {
    const candidate = path.join(experimentRoot, name, "reproduction_report.md");
    if (fs.existsSync(candidate)) reportFiles.push(candidate);
  }
}

let updated = 0;
for (const file of reportFiles) updated += updateReport(file) ? 1 : 0;
process.stdout.write(`Updated ${updated} legacy reproduction reports.\n`);
