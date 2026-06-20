import type { TextContent } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext, ToolResultEvent } from "@earendil-works/pi-coding-agent";
import { isReadToolResult } from "@earendil-works/pi-coding-agent";
import { Box, Text } from "@earendil-works/pi-tui";
import { registerContinuousLearningCommands } from "./lib/commands.js";
import { loadMergedInstincts } from "./lib/instincts.js";
import { maybeAnalyzeObservations, type ObserverRuntimeState } from "./lib/observer.js";
import { detectProject } from "./lib/project.js";
import {
	appendObservation,
	countObservationLines,
	ensureStorage,
	getStorageLayout,
	loadConfig,
	loadObserverState,
} from "./lib/storage.js";
import type {
	AgentRunMessageDetails,
	ContinuousLearningConfig,
	LearnEvalMessageDetails,
	ObservationEntry,
	ProjectInfo,
	SkillCreateMessageDetails,
	StorageLayout,
} from "./lib/types.js";

function scrubSecrets(text: string): string {
	return text.replace(
		/(api[_-]?key|token|secret|password|authorization|credentials?|auth)(["'\s:=]+)([A-Za-z]+\s+)?([A-Za-z0-9_\-/.+=]{8,})/giu,
		(_match, key: string, separator: string, prefix: string | undefined) =>
			`${key}${separator}${prefix ?? ""}[REDACTED]`,
	);
}

function flattenTextContent(content: unknown): string {
	if (typeof content === "string") {
		return content;
	}
	if (!Array.isArray(content)) {
		return "";
	}
	return content
		.filter(
			(item): item is TextContent =>
				Boolean(item) && typeof item === "object" && (item as { type?: string }).type === "text",
		)
		.map((item) => item.text)
		.join("\n");
}

interface RuntimeState {
	project: ProjectInfo | null;
	layout: StorageLayout | null;
	config: ContinuousLearningConfig | null;
	observer: ObserverRuntimeState;
}

export default function continuousLearningV2(pi: ExtensionAPI) {
	pi.registerMessageRenderer("continuous-learning-skill-create", (message, { expanded }, theme) => {
		const details = message.details as SkillCreateMessageDetails | undefined;
		const lines: string[] = [];

		lines.push(theme.fg("accent", theme.bold("╔════════════════════════════════════════════════════════════╗")));
		lines.push(theme.fg("accent", theme.bold("║ ECC Skill Creator                                         ║")));
		lines.push(
			theme.fg(
				"accent",
				theme.bold(
					`║ Repo: ${details?.repoName ?? "unknown"}${" ".repeat(Math.max(0, 49 - (details?.repoName?.length ?? 7)))}║`,
				),
			),
		);
		lines.push(theme.fg("accent", theme.bold("╚════════════════════════════════════════════════════════════╝")));
		lines.push("");

		if (details) {
			lines.push(`${theme.fg("muted", "Generation:")} ${theme.fg("success", details.generationMode)}`);
			lines.push(`${theme.fg("muted", "LLM:")} ${details.llmStatus}`);
			lines.push(
				`${theme.fg("muted", "Model:")} ${details.modelLabel ?? "none"} ${theme.fg("dim", `[${details.modelSource}]`)}`,
			);
			lines.push(`${theme.fg("muted", "Commits:")} ${String(details.commitCount)}`);
			lines.push(`${theme.fg("muted", "Verdict:")} ${details.quality.verdict}`);
			lines.push("");
			lines.push(theme.bold("Analysis"));
			lines.push(`${theme.fg("muted", "Prefixes:")} ${details.prefixes.join(", ") || "none"}`);
			lines.push(
				`${theme.fg("muted", "Representative files:")} ${details.representativeFiles.join(", ") || "none"}`,
			);
			lines.push("");
			lines.push(theme.bold("Output"));
			lines.push(`${theme.fg("muted", "Skill:")} ${details.skillPath}`);
			if (details.instinctPaths.length > 0) {
				lines.push(`${theme.fg("muted", "Instincts:")} ${details.instinctPaths.length}`);
			}
			if (expanded) {
				lines.push("");
				lines.push(theme.bold("Quality Gate"));
				for (const item of details.quality.checklist) {
					lines.push(`- ${item}`);
				}
				if (details.quality.overlapSkills.length > 0) {
					lines.push(theme.fg("warning", "Overlapping skills:"));
					for (const path of details.quality.overlapSkills) {
						lines.push(`- ${path}`);
					}
				}
				if (details.quality.absorbTarget) {
					lines.push(theme.fg("warning", `Absorb target: ${details.quality.absorbTarget}`));
				}
				if (details.quality.droppedInstinctIds.length > 0) {
					lines.push(theme.fg("warning", "Dropped instincts:"));
					for (const instinctId of details.quality.droppedInstinctIds) {
						lines.push(`- ${instinctId}`);
					}
				}
				if (details.quality.improvements && details.quality.improvements.length > 0) {
					lines.push(theme.fg("warning", "Improvements:"));
					for (const item of details.quality.improvements) {
						lines.push(`- ${item}`);
					}
				}
				if (details.quality.revised) {
					lines.push(theme.fg("success", "Revised once via improve-then-save"));
				}
				if (details.quality.absorbContent) {
					lines.push(theme.fg("warning", "Absorb content:"));
					lines.push(details.quality.absorbContent);
				}
				lines.push(`${theme.fg("muted", "Rationale:")} ${details.quality.rationale}`);
			}
		} else {
			lines.push(String(message.content));
		}

		const box = new Box(1, 1, (text) => theme.bg("customMessageBg", text));
		box.addChild(new Text(lines.join("\n"), 0, 0));
		return box;
	});

	pi.registerMessageRenderer("continuous-learning-agent-run", (message, { expanded }, theme) => {
		const details = message.details as AgentRunMessageDetails | undefined;
		const lines: string[] = [];
		lines.push(theme.fg("accent", theme.bold("╔════════════════════════════════════════════════════════════╗")));
		lines.push(theme.fg("accent", theme.bold("║ Evolved Agent Runner                                      ║")));
		lines.push(theme.fg("accent", theme.bold("╚════════════════════════════════════════════════════════════╝")));
		lines.push("");

		if (details) {
			lines.push(`${theme.fg("muted", "Agent:")} ${details.agentName}`);
			lines.push(`${theme.fg("muted", "Model:")} ${details.modelLabel}`);
			lines.push(`${theme.fg("muted", "Mode:")} ${details.executionMode ?? "manual"}`);
			lines.push(`${theme.fg("muted", "Session:")} ${details.sessionId}`);
			lines.push("");
			lines.push(theme.bold("Task"));
			lines.push(details.task);
			if (expanded) {
				lines.push("");
				lines.push(theme.bold("Agent Path"));
				lines.push(details.agentPath);
			}
			lines.push("");
			lines.push(theme.bold("Output"));
			lines.push(details.output);
		} else {
			lines.push(String(message.content));
		}

		const box = new Box(1, 1, (text) => theme.bg("customMessageBg", text));
		box.addChild(new Text(lines.join("\n"), 0, 0));
		return box;
	});

	pi.registerMessageRenderer("continuous-learning-learn-eval", (message, { expanded }, theme) => {
		const details = message.details as LearnEvalMessageDetails | undefined;
		const lines: string[] = [];

		lines.push(theme.fg("accent", theme.bold("╔════════════════════════════════════════════════════════════╗")));
		lines.push(theme.fg("accent", theme.bold("║ ECC Learn Eval                                            ║")));
		lines.push(
			theme.fg(
				"accent",
				theme.bold(
					`║ Project: ${details?.projectLabel ?? "unknown"}${" ".repeat(Math.max(0, 46 - (details?.projectLabel?.length ?? 7)))}║`,
				),
			),
		);
		lines.push(theme.fg("accent", theme.bold("╚════════════════════════════════════════════════════════════╝")));
		lines.push("");

		if (details) {
			const rawStatus = details.awaitingConfirmation
				? "awaiting-confirmation"
				: (details.applyStatus ?? (details.applied ? "applied" : "not-applied"));
			const status = details.awaitingConfirmation
				? theme.fg("warning", rawStatus)
				: details.applied
					? theme.fg("success", rawStatus)
					: rawStatus.startsWith("skipped")
						? theme.fg("warning", rawStatus)
						: theme.fg("muted", rawStatus);
			lines.push(`${theme.fg("muted", "Verdict:")} ${details.verdict}`);
			lines.push(`${theme.fg("muted", "Scope:")} ${details.scope}`);
			lines.push(`${theme.fg("muted", "Target:")} ${details.target}`);
			lines.push(`${theme.fg("muted", "Status:")} ${status}`);
			if (details.applyMessage) {
				lines.push(`${theme.fg("muted", "Apply:")} ${details.applyMessage}`);
			}
			lines.push("");
			lines.push(theme.bold("Checklist"));
			for (const item of details.checklist) {
				lines.push(`- ${item}`);
			}
			lines.push("");
			lines.push(`${theme.fg("muted", "Rationale:")} ${details.rationale}`);
			if (expanded) {
				if (details.improvements && details.improvements.length > 0) {
					lines.push("");
					lines.push(theme.bold("Improvements"));
					for (const item of details.improvements) {
						lines.push(`- ${item}`);
					}
				}
				if (details.absorbTarget) {
					lines.push("");
					lines.push(theme.bold("Absorb Target"));
					lines.push(details.absorbTarget);
				}
				if (details.absorbContent) {
					lines.push("");
					lines.push(theme.bold("Absorb Content"));
					lines.push(details.absorbContent);
				}
				if (details.skillMarkdown && (details.verdict === "save" || details.verdict === "improve-then-save")) {
					lines.push("");
					lines.push(theme.bold("Draft Skill"));
					lines.push(details.skillMarkdown);
				}
			}
		} else {
			lines.push(String(message.content));
		}

		const box = new Box(1, 1, (text) => theme.bg("customMessageBg", text));
		box.addChild(new Text(lines.join("\n"), 0, 0));
		return box;
	});

	const state: RuntimeState = {
		project: null,
		layout: null,
		config: null,
		observer: {
			running: false,
			timer: null,
			scheduledAnalysis: null,
		},
	};

	const clearObserverTimer = () => {
		if (state.observer.timer) {
			clearInterval(state.observer.timer);
			state.observer.timer = null;
		}
		if (state.observer.scheduledAnalysis) {
			clearTimeout(state.observer.scheduledAnalysis);
			state.observer.scheduledAnalysis = null;
			state.observer.scheduledAnalysisAt = undefined;
		}
	};

	const setupProject = async (ctx: ExtensionContext) => {
		state.project = await detectProject(ctx.cwd);
		state.layout = getStorageLayout(state.project);
		await ensureStorage(state.project, state.layout);
		state.config = await loadConfig(state.layout);
	};

	const scheduleObserverAnalysis = async (ctx: ExtensionContext, delayMs: number) => {
		if (!state.project || !state.layout) {
			return;
		}
		state.config = state.config ?? (await loadConfig(state.layout));
		if (!state.config.observer.enabled) {
			return;
		}
		const scheduledAt = Date.now() + Math.max(0, delayMs);
		if (
			state.observer.scheduledAnalysis &&
			state.observer.scheduledAnalysisAt !== undefined &&
			state.observer.scheduledAnalysisAt <= scheduledAt
		) {
			return;
		}
		if (state.observer.scheduledAnalysis) {
			clearTimeout(state.observer.scheduledAnalysis);
		}
		state.observer.scheduledAnalysisAt = scheduledAt;
		state.observer.scheduledAnalysis = setTimeout(
			() => {
				state.observer.scheduledAnalysis = null;
				state.observer.scheduledAnalysisAt = undefined;
				if (!state.project || !state.layout) {
					return;
				}
				void maybeAnalyzeObservations(ctx, state.project, state.layout, state.observer)
					.then((result) => {
						if (result.retryAfterMs && state.config?.observer.enabled) {
							void scheduleObserverAnalysis(ctx, result.retryAfterMs);
						}
					})
					.catch(() => {});
			},
			Math.max(0, delayMs),
		);
	};

	const scheduleObserver = async (ctx: ExtensionContext) => {
		clearObserverTimer();
		if (!state.layout || !state.project) {
			return;
		}
		state.config = await loadConfig(state.layout);
		if (!state.config.observer.enabled) {
			return;
		}
		const intervalMs = Math.max(1, state.config.observer.runIntervalMinutes) * 60_000;
		state.observer.timer = setInterval(() => {
			if (!state.project || !state.layout) {
				return;
			}
			void scheduleObserverAnalysis(ctx, 0);
		}, intervalMs);
	};

	const maybeTriggerObserver = async (ctx: ExtensionContext) => {
		if (!state.project || !state.layout) {
			return;
		}
		state.config = state.config ?? (await loadConfig(state.layout));
		if (!state.config.observer.enabled) {
			return;
		}
		const observationCount = await countObservationLines(state.layout);
		const observerState = await loadObserverState(state.layout);
		const pending = observationCount - observerState.lastAnalyzedIndex;
		if (pending >= state.config.observer.minObservationsToAnalyze) {
			await scheduleObserverAnalysis(ctx, 15_000);
		}
	};

	const append = async (
		ctx: ExtensionContext,
		entry: Omit<ObservationEntry, "timestamp" | "projectId" | "projectName" | "cwd">,
	) => {
		if (!state.project || !state.layout) {
			await setupProject(ctx);
		}
		if (!state.project || !state.layout) {
			return;
		}
		await appendObservation(state.layout, {
			timestamp: new Date().toISOString(),
			projectId: state.project.id,
			projectName: state.project.name,
			cwd: ctx.cwd,
			...entry,
		});
		await maybeTriggerObserver(ctx);
	};

	registerContinuousLearningCommands(pi, () => ({
		project: state.project,
		layout: state.layout,
		observer: state.observer,
	}));

	pi.on("resources_discover", async (event) => {
		const project = await detectProject(event.cwd);
		const layout = getStorageLayout(project);
		await ensureStorage(project, layout);
		await loadMergedInstincts(layout);
		return {
			skillPaths: [layout.projectEvolvedSkillsDir, layout.globalEvolvedSkillsDir],
			promptPaths: [layout.projectEvolvedPromptsDir, layout.globalEvolvedPromptsDir],
		};
	});

	pi.on("session_start", async (_event, ctx) => {
		await setupProject(ctx);
		await scheduleObserver(ctx);
		if (state.project && state.config?.observer.enabled && ctx.hasUI) {
			ctx.ui.notify(`Continuous Learning active for ${state.project.name}`, "info");
		}
		await maybeTriggerObserver(ctx);
	});

	pi.on("session_shutdown", async () => {
		clearObserverTimer();
	});

	pi.on("input", async (event, ctx) => {
		if (!event.text.trim()) {
			return;
		}
		await append(ctx, {
			event: "input",
			inputText: scrubSecrets(event.text).slice(0, 4000),
		});
	});

	pi.on("tool_call", async (event, ctx) => {
		await append(ctx, {
			event: "tool_call",
			toolName: event.toolName,
			toolInput: scrubSecrets(JSON.stringify(event.input)).slice(0, 4000),
		});
	});

	pi.on("tool_result", async (event, ctx) => {
		await append(ctx, buildToolResultObservation(event));
	});

	pi.on("turn_end", async (event, ctx) => {
		const assistantText = "content" in event.message ? flattenTextContent(event.message.content) : "";
		if (!assistantText.trim()) {
			return;
		}
		await append(ctx, {
			event: "turn_end",
			assistantText: scrubSecrets(assistantText).slice(0, 4000),
		});
	});
}

function buildToolResultObservation(
	event: ToolResultEvent,
): Omit<ObservationEntry, "timestamp" | "projectId" | "projectName" | "cwd"> {
	const text = isReadToolResult(event)
		? flattenTextContent(event.content)
		: event.content
				.filter((item): item is TextContent => item.type === "text")
				.map((item) => item.text)
				.join("\n");
	return {
		event: "tool_result",
		toolName: event.toolName,
		toolOutput: scrubSecrets(text).slice(0, 4000),
		isError: event.isError,
	};
}
