import { readdir, readFile, rm } from "node:fs/promises";
import { extname, join, resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { runEvolvedAgent } from "./evolved-agent-runner.js";
import {
	analyzeEvolution,
	collectPendingInstincts,
	findPromotionCandidates,
	generateEvolvedOutputs,
	importInstincts,
	loadMergedInstincts,
	loadProjectOnlyInstincts,
	parseInstinctExport,
	pendingExpiryThresholdDays,
	pendingTtlDays,
	prunePendingInstincts,
	renderInstinctExport,
	serializeInstinct,
} from "./instincts.js";
import { applyLearnEvalResult, evaluateSessionLearning } from "./learn-eval.js";
import { resolveActiveOrDefaultModel } from "./model-selection.js";
import type { ObserverRuntimeState } from "./observer.js";
import { createSkillFromRepository } from "./skill-create.js";
import {
	countObservationLines,
	getStorageLayout,
	loadConfig,
	loadObserverState,
	loadProjectRegistry,
	writeTextFile,
} from "./storage.js";
import type {
	AgentRunMessageDetails,
	LearnEvalMessageDetails,
	ProjectInfo,
	SkillCreateMessageDetails,
	StorageLayout,
} from "./types.js";

interface ParsedArgs {
	flags: Map<string, string | true>;
	positionals: string[];
}

function parseArgs(input: string): ParsedArgs {
	const tokens: string[] = [];
	let current = "";
	let quote: '"' | "'" | null = null;
	for (const char of input.trim()) {
		if (quote) {
			if (char === quote) {
				quote = null;
			} else {
				current += char;
			}
			continue;
		}
		if (char === '"' || char === "'") {
			quote = char;
			continue;
		}
		if (/\s/u.test(char)) {
			if (current.length > 0) {
				tokens.push(current);
				current = "";
			}
			continue;
		}
		current += char;
	}
	if (current.length > 0) {
		tokens.push(current);
	}

	const flags = new Map<string, string | true>();
	const positionals: string[] = [];
	for (let index = 0; index < tokens.length; index++) {
		const token = tokens[index];
		if (!token.startsWith("--")) {
			positionals.push(token);
			continue;
		}
		const [name, inlineValue] = token.slice(2).split("=", 2);
		if (inlineValue !== undefined) {
			flags.set(name, inlineValue);
			continue;
		}
		const next = tokens[index + 1];
		if (next && !next.startsWith("--")) {
			flags.set(name, next);
			index++;
		} else {
			flags.set(name, true);
		}
	}
	return { flags, positionals };
}

function formatConfidenceBar(confidence: number): string {
	const filled = Math.max(0, Math.min(10, Math.round(confidence * 10)));
	return `${"█".repeat(filled)}${"░".repeat(10 - filled)}`;
}

function extractAction(content: string): string {
	const match = content.match(/## Action\s+([\s\S]*?)(?:\n## |\n*$)/u);
	const action = match?.[1]?.trim().split("\n")[0];
	return action && action.length > 0 ? action : "No action recorded";
}

function emitReport(pi: ExtensionAPI, customType: string, content: string): void {
	pi.sendMessage(
		{
			customType,
			content,
			display: true,
		},
		{ triggerTurn: false },
	);
}

function currentProjectLabel(project: ProjectInfo): string {
	return `${project.name} (${project.id})`;
}

function buildLearnEvalTarget(details: {
	verdict: "save" | "improve-then-save" | "absorb" | "drop";
	absorbTarget?: string;
	targetPath: string | null;
}): string {
	return details.verdict === "absorb" ? (details.absorbTarget ?? "existing skill") : (details.targetPath ?? "(none)");
}

function buildLearnEvalSummary(details: LearnEvalMessageDetails): string {
	const status = details.awaitingConfirmation ? "awaiting-confirmation" : details.applied ? "applied" : "not-applied";
	return [
		`LEARN EVAL - ${details.projectLabel}`,
		`Verdict: ${details.verdict}`,
		`Scope: ${details.scope}`,
		`Target: ${details.target}`,
		`Status: ${status}`,
		`Rationale: ${details.rationale}`,
	].join("\n");
}

function emitLearnEvalReport(pi: ExtensionAPI, details: LearnEvalMessageDetails): void {
	pi.sendMessage({
		customType: "continuous-learning-learn-eval",
		content: buildLearnEvalSummary(details),
		display: true,
		details,
	});
}

function buildLearnEvalConfirmTitle(verdict: LearnEvalMessageDetails["verdict"]): string {
	if (verdict === "absorb") {
		return "Absorb learned pattern?";
	}
	if (verdict === "improve-then-save") {
		return "Save revised learned pattern?";
	}
	return "Save learned pattern?";
}

function buildLearnEvalConfirmBody(details: LearnEvalMessageDetails): string {
	return [
		`Target: ${details.target}`,
		`Scope: ${details.scope}`,
		"",
		...details.checklist.map((item) => `- ${item}`),
		"",
		details.rationale,
	].join("\n");
}

function formatRelativeMs(ms: number): string {
	const seconds = Math.max(0, Math.round(ms / 1000));
	if (seconds < 60) {
		return `${seconds}s`;
	}
	const minutes = Math.floor(seconds / 60);
	const remainingSeconds = seconds % 60;
	return `${minutes}m ${remainingSeconds}s`;
}

function formatOptionalTimestamp(value: string | undefined): string {
	return value ?? "(never)";
}

const INSTINCT_FILE_EXTENSIONS = new Set([".md", ".yaml", ".yml"]);

async function countInstinctFiles(dirPath: string): Promise<number> {
	try {
		const entries = await readdir(dirPath, { withFileTypes: true });
		return entries.filter(
			(entry) => entry.isFile() && INSTINCT_FILE_EXTENSIONS.has(extname(entry.name).toLowerCase()),
		).length;
	} catch {
		return 0;
	}
}

async function loadImportSource(source: string, cwd: string): Promise<string> {
	if (source.startsWith("http://") || source.startsWith("https://")) {
		const response = await fetch(source);
		if (!response.ok) {
			throw new Error(`Failed to fetch ${source}: ${response.status}`);
		}
		return response.text();
	}
	return readFile(resolve(cwd, source), "utf-8");
}

function appendInstinctsByDomain(
	lines: string[],
	instincts: Array<{
		id: string;
		domain: string;
		confidence: number;
		scopeLabel: string;
		trigger: string;
		content: string;
	}>,
): void {
	const byDomain = new Map<string, typeof instincts>();
	for (const instinct of instincts) {
		const domain = instinct.domain.trim().length > 0 ? instinct.domain : "general";
		const group = byDomain.get(domain) ?? [];
		group.push(instinct);
		byDomain.set(domain, group);
	}

	for (const domain of Array.from(byDomain.keys()).sort()) {
		const group = byDomain
			.get(domain)
			?.slice()
			.sort((left, right) => right.confidence - left.confidence);
		if (!group || group.length === 0) {
			continue;
		}
		lines.push(`### ${domain.toUpperCase()} (${group.length})`);
		for (const instinct of group) {
			lines.push(
				`${formatConfidenceBar(instinct.confidence)} ${Math.round(instinct.confidence * 100)}% ${instinct.id} [${instinct.scopeLabel}]`,
			);
			lines.push(`trigger: ${instinct.trigger}`);
			lines.push(`action: ${extractAction(instinct.content)}`);
			lines.push("");
		}
	}
}

export function registerContinuousLearningCommands(
	pi: ExtensionAPI,
	getState: () => {
		project: ProjectInfo | null;
		layout: StorageLayout | null;
		observer: ObserverRuntimeState;
	},
): void {
	pi.registerCommand("instinct-status", {
		description: "Show learned instincts for the current project and global scope",
		handler: async (_args, _ctx) => {
			const { project, layout } = getState();
			if (!project || !layout) {
				return;
			}
			const instincts = await loadMergedInstincts(layout);
			const [pending, observationCount] = await Promise.all([
				collectPendingInstincts(layout, { includeAllProjects: true }),
				countObservationLines(layout),
			]);
			const projectInstincts = instincts.filter((instinct) => instinct.scopeLabel === "project");
			const globalInstincts = instincts.filter((instinct) => instinct.scopeLabel === "global");
			const lines =
				instincts.length === 0
					? [
							"No instincts found.",
							"",
							`Project: ${currentProjectLabel(project)}`,
							`Project instincts: ${layout.projectPersonalDir}`,
							`Global instincts: ${layout.globalPersonalDir}`,
						]
					: [
							`INSTINCT STATUS - ${instincts.length} total`,
							"",
							`Project: ${currentProjectLabel(project)}`,
							`Project instincts: ${projectInstincts.length}`,
							`Global instincts: ${globalInstincts.length}`,
							"",
						];

			if (instincts.length > 0) {
				for (const [label, group] of [
					[`PROJECT-SCOPED (${project.name})`, projectInstincts],
					["GLOBAL", globalInstincts],
				] as const) {
					if (group.length === 0) {
						continue;
					}
					lines.push(`## ${label}`);
					appendInstinctsByDomain(lines, group);
				}
			}
			if (observationCount > 0) {
				lines.push("---");
				lines.push(`Observations: ${observationCount} events logged`);
			}
			if (pending.length > 0) {
				if (observationCount === 0) {
					lines.push("---");
				}
				lines.push(`Pending instincts: ${pending.length} awaiting review`);
				if (pending.length >= 5) {
					lines.push(
						`${pending.length} pending instincts awaiting review. Unreviewed instincts auto-delete after ${pendingTtlDays()} days.`,
					);
				}
				const expiringSoon = pending.filter((item) => item.ageDays >= pendingExpiryThresholdDays());
				if (expiringSoon.length > 0) {
					lines.push(`Expiring within 7 days:`);
					for (const item of expiringSoon.slice(0, 10)) {
						lines.push(`- ${item.name} (${Math.max(0, 30 - item.ageDays)}d remaining)`);
					}
				}
			}
			emitReport(pi, "continuous-learning-status", lines.join("\n"));
		},
	});

	pi.registerCommand("observer-status", {
		description: "Show observer runtime status, scheduling state, and recent learning activity",
		handler: async (_args, _ctx) => {
			const { project, layout, observer } = getState();
			if (!project || !layout) {
				return;
			}
			const [config, observerState, observationCount, pending] = await Promise.all([
				loadConfig(layout),
				loadObserverState(layout),
				countObservationLines(layout),
				collectPendingInstincts(layout),
			]);
			const pendingObservationCount = Math.max(0, observationCount - observerState.lastAnalyzedIndex);
			const scheduledInMs =
				observer.scheduledAnalysisAt !== undefined
					? Math.max(0, observer.scheduledAnalysisAt - Date.now())
					: undefined;
			const lines = [
				`OBSERVER STATUS - ${currentProjectLabel(project)}`,
				"",
				`Enabled: ${config.observer.enabled ? "yes" : "no"}`,
				`Running: ${observer.running ? "yes" : "no"}`,
				`Scheduled: ${observer.scheduledAnalysis ? `yes (${formatRelativeMs(scheduledInMs ?? 0)})` : "no"}`,
				`Observations: ${observationCount}`,
				`Pending observations: ${pendingObservationCount}`,
				`Pending instincts: ${pending.length}`,
				"",
				`Last analyzed index: ${observerState.lastAnalyzedIndex}`,
				`Last analyzed at: ${formatOptionalTimestamp(observerState.lastAnalyzedAt)}`,
				`Last attempted at: ${formatOptionalTimestamp(observer.lastAttemptedAt)}`,
				`Last completed at: ${formatOptionalTimestamp(observer.lastCompletedAt)}`,
				`Last result: ${observer.lastResult ? JSON.stringify(observer.lastResult) : "(none)"}`,
				`Last error: ${observer.lastError ?? "(none)"}`,
				"",
				`Config interval: ${config.observer.runIntervalMinutes} min`,
				`Min observations: ${config.observer.minObservationsToAnalyze}`,
				`Max recent observations: ${config.observer.maxRecentObservations}`,
				`Configured observer model: ${config.observer.model ?? "(active/settings fallback)"}`,
			];
			emitReport(pi, "continuous-learning-observer-status", lines.join("\n"));
		},
	});

	pi.registerCommand("instinct-export", {
		description: "Export instincts to stdout-like report or a file",
		handler: async (args, _ctx) => {
			const { project, layout } = getState();
			if (!project || !layout) {
				return;
			}
			const parsed = parseArgs(args);
			const scope = typeof parsed.flags.get("scope") === "string" ? String(parsed.flags.get("scope")) : "all";
			const domain = typeof parsed.flags.get("domain") === "string" ? String(parsed.flags.get("domain")) : undefined;
			const minConfidenceRaw = parsed.flags.get("min-confidence");
			const minConfidence = typeof minConfidenceRaw === "string" ? Number.parseFloat(minConfidenceRaw) : undefined;
			const output = typeof parsed.flags.get("output") === "string" ? String(parsed.flags.get("output")) : undefined;

			const instincts =
				scope === "project" ? await loadProjectOnlyInstincts(layout) : await loadMergedInstincts(layout);
			if (instincts.length === 0) {
				emitReport(pi, "continuous-learning-export", "No instincts to export.");
				return;
			}
			const filtered = instincts.filter((instinct) => {
				if (scope === "global" && instinct.scopeLabel !== "global") {
					return false;
				}
				if (domain && instinct.domain !== domain) {
					return false;
				}
				if (minConfidence !== undefined && instinct.confidence < minConfidence) {
					return false;
				}
				return true;
			});
			if (filtered.length === 0) {
				emitReport(pi, "continuous-learning-export", "No instincts match the criteria.");
				return;
			}

			const rendered = renderInstinctExport(filtered, { scope: scope as "project" | "global" | "all", project });
			if (output) {
				const outputPath = resolve(_ctx.cwd, output);
				await writeTextFile(outputPath, rendered);
				emitReport(pi, "continuous-learning-export", `Exported ${filtered.length} instincts to ${outputPath}`);
				return;
			}
			emitReport(pi, "continuous-learning-export", rendered);
		},
	});

	pi.registerCommand("instinct-import", {
		description: "Import instincts from a file or URL",
		handler: async (args, ctx) => {
			const { project, layout } = getState();
			if (!project || !layout) {
				return;
			}
			const parsed = parseArgs(args);
			const source = parsed.positionals[0];
			if (!source) {
				ctx.ui.notify("Usage: /instinct-import <file-or-url> [--scope project|global] [--force]", "warning");
				return;
			}
			const scope = parsed.flags.get("scope") === "global" ? "global" : "project";
			const effectiveScope = project.id === "global" && scope === "project" ? "global" : scope;
			const dryRun = parsed.flags.has("dry-run");
			const force = parsed.flags.has("force");
			const minConfidenceRaw = parsed.flags.get("min-confidence");
			const minConfidence = typeof minConfidenceRaw === "string" ? Number.parseFloat(minConfidenceRaw) : undefined;

			const raw = await loadImportSource(source, ctx.cwd);
			const incoming = parseInstinctExport(raw);
			if (incoming.length === 0) {
				emitReport(pi, "continuous-learning-import", "No valid instincts found in source.");
				return;
			}
			const summary = await importInstincts(layout, project, source, incoming, scope, minConfidence, true);
			if (!force && ctx.hasUI) {
				const confirmed = await ctx.ui.confirm(
					"Import instincts?",
					`Add ${summary.added.length}, update ${summary.updated.length}, skip ${summary.skipped.length}`,
				);
				if (!confirmed) {
					ctx.ui.notify("Import cancelled", "info");
					return;
				}
			}
			const applied = await importInstincts(layout, project, source, incoming, scope, minConfidence, dryRun);
			emitReport(
				pi,
				"continuous-learning-import",
				`Import complete for ${currentProjectLabel(project)}\nScope: ${effectiveScope}\nAdded: ${applied.added.length}\nUpdated: ${applied.updated.length}\nSkipped: ${applied.skipped.length}${dryRun ? "\n[DRY RUN]" : ""}`,
			);
		},
	});

	pi.registerCommand("promote", {
		description: "Promote project instincts to global scope",
		handler: async (args, ctx) => {
			const { project, layout } = getState();
			if (!project || !layout) {
				return;
			}
			const parsed = parseArgs(args);
			const instinctId = parsed.positionals[0];
			const dryRun = parsed.flags.has("dry-run");
			const force = parsed.flags.has("force");
			const globalInstincts = await loadProjectOnlyInstincts(
				getStorageLayout({
					id: "global",
					name: "global",
					root: project.root,
				}),
			);
			const globalIds = new Set(globalInstincts.map((instinct) => instinct.id));
			if (instinctId && globalIds.has(instinctId)) {
				ctx.ui.notify(`Instinct '${instinctId}' already exists in global scope`, "info");
				return;
			}
			let targetCandidates = await findPromotionCandidates(layout);
			targetCandidates = targetCandidates.filter((candidate) => !globalIds.has(candidate.id));
			if (instinctId) {
				const projectInstincts = await loadProjectOnlyInstincts(layout);
				const specific = projectInstincts.find((instinct) => instinct.id === instinctId);
				if (!specific) {
					ctx.ui.notify(`Instinct '${instinctId}' not found in project ${project.name}`, "info");
					return;
				}
				targetCandidates = specific
					? [
							{
								id: specific.id,
								entries: [specific],
								averageConfidence: specific.confidence,
							},
						]
					: [];
			}

			if (targetCandidates.length === 0) {
				ctx.ui.notify(
					instinctId
						? "No promotion candidates found"
						: "No instincts qualify for auto-promotion.\nCriteria: appears in 2+ projects, avg confidence >= 80%",
					"info",
				);
				return;
			}
			if (!force && ctx.hasUI) {
				const confirmed = await ctx.ui.confirm(
					"Promote instincts?",
					targetCandidates
						.map((candidate) => `${candidate.id} (${Math.round(candidate.averageConfidence * 100)}%)`)
						.join("\n"),
				);
				if (!confirmed) {
					ctx.ui.notify("Promotion cancelled", "info");
					return;
				}
			}

			if (!dryRun) {
				for (const candidate of targetCandidates) {
					const best = [...candidate.entries].sort((left, right) => right.confidence - left.confidence)[0];
					await writeTextFile(
						join(layout.globalPersonalDir, `${candidate.id}.md`),
						serializeInstinct({
							...best,
							scope: "global",
							projectId: undefined,
							projectName: undefined,
							promotedFrom: best.scopeLabel === "project" ? best.projectId : undefined,
						}),
					);
				}
			}

			emitReport(
				pi,
				"continuous-learning-promote",
				`Promotion candidates: ${targetCandidates.length}${dryRun ? "\n[DRY RUN]" : ""}\n${targetCandidates.map((candidate) => `- ${candidate.id} (${Math.round(candidate.averageConfidence * 100)}%)`).join("\n")}`,
			);
		},
	});

	pi.registerCommand("projects", {
		description: "List known projects and instinct statistics",
		handler: async (_args, _ctx) => {
			const { layout } = getState();
			if (!layout) {
				return;
			}
			const registry = await loadProjectRegistry(layout);
			const lines = ["KNOWN PROJECTS", ""];
			for (const entry of Object.values(registry).sort((left, right) =>
				right.lastSeen.localeCompare(left.lastSeen),
			)) {
				const projectLayout = getStorageLayout({
					id: entry.id,
					name: entry.name,
					root: entry.root,
					remote: entry.remote,
				});
				const [personalCount, inheritedCount, observationCount] = await Promise.all([
					countInstinctFiles(projectLayout.projectPersonalDir),
					countInstinctFiles(projectLayout.projectInheritedDir),
					countObservationLines(projectLayout),
				]);
				lines.push(`${entry.name} [${entry.id}]`);
				lines.push(`root: ${entry.root}`);
				if (entry.remote) {
					lines.push(`remote: ${entry.remote}`);
				}
				lines.push(`personal instincts: ${personalCount}`);
				lines.push(`inherited instincts: ${inheritedCount}`);
				lines.push(`observations: ${observationCount}`);
				lines.push(`last seen: ${entry.lastSeen}`);
				lines.push("");
			}
			const [globalPersonalCount, globalInheritedCount] = await Promise.all([
				countInstinctFiles(layout.globalPersonalDir),
				countInstinctFiles(layout.globalInheritedDir),
			]);
			lines.push("GLOBAL TOTALS");
			lines.push(`global personal instincts: ${globalPersonalCount}`);
			lines.push(`global inherited instincts: ${globalInheritedCount}`);
			emitReport(pi, "continuous-learning-projects", lines.join("\n"));
		},
	});

	pi.registerCommand("prune", {
		description: "Delete pending instincts older than the TTL threshold",
		handler: async (args, _ctx) => {
			const { layout } = getState();
			if (!layout) {
				return;
			}
			const parsed = parseArgs(args);
			const maxAgeRaw = parsed.flags.get("max-age");
			const maxAge = typeof maxAgeRaw === "string" ? Math.max(1, Number.parseInt(maxAgeRaw, 10) || 30) : 30;
			const dryRun = parsed.flags.has("dry-run");
			const summary = await prunePendingInstincts(layout, maxAge, dryRun);
			const lines = [
				`${dryRun ? "[DRY RUN] " : ""}Prune pending instincts older than ${maxAge} days`,
				`Pruned: ${summary.pruned.length}`,
				`Remaining: ${summary.remaining.length}`,
			];
			if (summary.pruned.length > 0) {
				lines.push("", "## Pruned");
				for (const item of summary.pruned) {
					lines.push(`- ${item.name} (${item.ageDays}d)`);
				}
			}
			emitReport(pi, "continuous-learning-prune", lines.join("\n"));
		},
	});

	pi.registerCommand("evolve", {
		description: "Analyze instincts and generate evolved skills, prompts, and agent specs",
		handler: async (args, ctx) => {
			const { project, layout } = getState();
			if (!project || !layout) {
				return;
			}
			const parsed = parseArgs(args);
			const generate = parsed.flags.has("generate");
			const instincts = await loadMergedInstincts(layout);
			const analysis = analyzeEvolution(instincts);
			let generated: string[] = [];
			if (generate) {
				generated = await generateEvolvedOutputs(layout, analysis);
				if (ctx.hasUI) {
					await ctx.reload();
				}
			}
			const lines = [
				`EVOLVE ANALYSIS - ${instincts.length} instincts`,
				`Project: ${currentProjectLabel(project)}`,
				"",
				`Skill candidates: ${analysis.skillCandidates.length}`,
				`Prompt candidates: ${analysis.promptCandidates.length}`,
				`Agent candidates: ${analysis.agentCandidates.length}`,
			];
			if (analysis.skillCandidates.length > 0) {
				lines.push("", "## Skill candidates");
				for (const candidate of analysis.skillCandidates.slice(0, 5)) {
					lines.push(
						`- ${candidate.key}: ${candidate.instincts.length} instincts, avg ${Math.round(candidate.averageConfidence * 100)}%`,
					);
				}
			}
			if (generate) {
				lines.push("", `Generated files: ${generated.length}`);
				for (const filePath of generated) {
					lines.push(`- ${filePath}`);
				}
			}
			emitReport(pi, "continuous-learning-evolve", lines.join("\n"));
		},
	});

	pi.registerCommand("agent-run", {
		description: "Run an evolved agent artifact manually against an explicit task",
		handler: async (args, ctx) => {
			const { project, layout } = getState();
			if (!project || !layout) {
				return;
			}
			const parsed = parseArgs(args);
			const agentRef = parsed.positionals[0];
			const task = parsed.positionals.slice(1).join(" ").trim();
			const modelOverride =
				typeof parsed.flags.get("model") === "string" ? String(parsed.flags.get("model")) : undefined;
			if (!agentRef || task.length === 0) {
				ctx.ui.notify("Usage: /agent-run <agent-name-or-path> <task...> [--model provider/id]", "warning");
				return;
			}

			try {
				const result = await runEvolvedAgent({
					ctx,
					project,
					layout,
					agentRef,
					task,
					modelOverride,
				});
				const details: AgentRunMessageDetails = {
					agentName: result.agent.name,
					agentPath: result.agent.filePath,
					executionMode: result.agent.executionMode,
					modelLabel: result.modelLabel,
					task: result.task,
					output: result.output,
					sessionId: result.sessionId,
				};
				pi.sendMessage({
					customType: "continuous-learning-agent-run",
					content: result.output,
					display: true,
					details,
				});
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				ctx.ui.notify(`agent-run 失败: ${message}`, "error");
			}
		},
	});

	pi.registerCommand("skill-create", {
		description:
			"Analyze git history and generate a repository skill; use --instincts to also write repo-analysis instincts",
		handler: async (args, ctx) => {
			const { project, layout } = getState();
			if (!project || !layout) {
				return;
			}

			const parsed = parseArgs(args);
			const commitsRaw = parsed.flags.get("commits");
			const commits = typeof commitsRaw === "string" ? Math.max(1, Number.parseInt(commitsRaw, 10) || 200) : 200;
			const output = typeof parsed.flags.get("output") === "string" ? String(parsed.flags.get("output")) : undefined;
			const includeInstincts = parsed.flags.has("instincts");
			const resolvedModel = await resolveActiveOrDefaultModel(ctx.model, ctx.modelRegistry);
			const model = resolvedModel.model;

			let llm:
				| {
						model: NonNullable<typeof model>;
						apiKey: string;
						headers?: Record<string, string>;
						modelRegistry: typeof ctx.modelRegistry;
				  }
				| undefined;
			if (model) {
				const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
				if (auth.ok && auth.apiKey) {
					llm = {
						model,
						apiKey: auth.apiKey,
						headers: auth.headers,
						modelRegistry: ctx.modelRegistry,
					};
				}
			}

			try {
				const result = await createSkillFromRepository({
					cwd: ctx.cwd,
					project,
					layout,
					commits,
					output,
					includeInstincts,
					llm,
				});

				if (ctx.hasUI && (!output || result.skillPath.startsWith(layout.projectEvolvedSkillsDir))) {
					await ctx.reload();
				}

				const details: SkillCreateMessageDetails = {
					repoName: project.name,
					commitCount: result.commitCount,
					generationMode: result.generationMode,
					llmStatus: result.llmStatus,
					modelLabel: llm ? `${llm.model.provider}/${llm.model.id}` : undefined,
					modelSource: resolvedModel.source,
					skillPath: result.skillPath,
					instinctPaths: result.instinctPaths,
					prefixes: result.prefixes,
					representativeFiles: result.representativeFiles,
					quality: result.quality,
				};

				pi.sendMessage({
					customType: "continuous-learning-skill-create",
					content: `${result.summary}${llm ? `\n使用模型: ${llm.model.provider}/${llm.model.id}` : "\n使用模型: none"}${includeInstincts ? `\n--instincts 已启用` : ""}`,
					display: true,
					details: ctx.hasUI ? details : undefined,
				});
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				ctx.ui.notify(`skill-create 失败: ${message}`, "error");
			}
		},
	});

	pi.registerCommand("learn-eval", {
		description: "Extract one reusable session pattern, evaluate quality, and optionally save it as a learned skill",
		handler: async (args, ctx) => {
			const { project } = getState();
			if (!project) {
				return;
			}

			const parsed = parseArgs(args);
			const applyRequested = parsed.flags.has("apply") || parsed.flags.has("force");
			const resolvedModel = await resolveActiveOrDefaultModel(ctx.model, ctx.modelRegistry);
			const model = resolvedModel.model;
			if (!model) {
				ctx.ui.notify("没有可用模型，无法执行 learn-eval", "warning");
				return;
			}
			const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
			if (!auth.ok || !auth.apiKey) {
				ctx.ui.notify(auth.ok ? "缺少模型 API key" : auth.error, "warning");
				return;
			}

			try {
				const result = await evaluateSessionLearning({
					project,
					sessionManager: ctx.sessionManager,
					llm: {
						model,
						apiKey: auth.apiKey,
						headers: auth.headers,
					},
				});

				const target = buildLearnEvalTarget({
					verdict: result.quality.verdict,
					absorbTarget: result.quality.absorbTarget,
					targetPath: result.targetPath,
				});
				const baseDetails: LearnEvalMessageDetails = {
					projectLabel: currentProjectLabel(project),
					verdict: result.quality.verdict,
					scope: result.quality.scope,
					target,
					targetPath: result.targetPath,
					applied: false,
					awaitingConfirmation: false,
					rationale: result.quality.rationale,
					checklist: result.quality.checklist,
					improvements: result.quality.improvements,
					absorbTarget: result.quality.absorbTarget,
					absorbContent: result.quality.absorbContent,
					skillMarkdown: result.skillMarkdown,
				};
				const needsConfirmation = ctx.hasUI && !applyRequested && result.quality.verdict !== "drop";
				if (needsConfirmation) {
					emitLearnEvalReport(pi, {
						...baseDetails,
						awaitingConfirmation: true,
					});
				}

				let applied = false;
				if (result.quality.verdict !== "drop") {
					if (needsConfirmation) {
						const confirmed = await ctx.ui.confirm(
							buildLearnEvalConfirmTitle(result.quality.verdict),
							buildLearnEvalConfirmBody(baseDetails),
						);
						if (confirmed) {
							await applyLearnEvalResult(result);
							applied = true;
						}
					} else if (applyRequested) {
						await applyLearnEvalResult(result);
						applied = true;
					}
				}

				if (applied && ctx.hasUI) {
					await ctx.reload();
				}
				emitLearnEvalReport(pi, {
					...baseDetails,
					applied,
				});
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				ctx.ui.notify(`learn-eval 失败: ${message}`, "error");
			}
		},
	});

	pi.registerCommand("instinct-prune", {
		description: "Prune obvious duplicate or superseded active project instincts generated by repo analysis",
		handler: async (_args, ctx) => {
			const { layout } = getState();
			if (!layout) {
				return;
			}

			const instincts = await loadProjectOnlyInstincts(layout);
			const byId = new Map(instincts.map((instinct) => [instinct.id, instinct]));
			const pruneTargets: string[] = [];

			if (byId.has("detsql-commit-convention") && byId.has("conventional-commit-scopes")) {
				pruneTargets.push("detsql-commit-convention");
			}

			if (pruneTargets.length === 0) {
				ctx.ui.notify("没有发现可安全裁剪的重复 instinct", "info");
				return;
			}

			if (ctx.hasUI) {
				const confirmed = await ctx.ui.confirm("Prune instincts?", pruneTargets.map((id) => `- ${id}`).join("\n"));
				if (!confirmed) {
					ctx.ui.notify("裁剪已取消", "info");
					return;
				}
			}

			const removed: string[] = [];
			for (const instinctId of pruneTargets) {
				const instinct = byId.get(instinctId);
				if (!instinct) {
					continue;
				}
				await rm(instinct.filePath, { force: true });
				removed.push(instinctId);
			}

			emitReport(
				pi,
				"continuous-learning-instinct-prune",
				`已裁剪 ${removed.length} 个 project instinct\n${removed.map((id) => `- ${id}`).join("\n")}`,
			);
		},
	});
}
