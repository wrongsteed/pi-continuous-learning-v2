import { complete, type UserMessage } from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { dedupeComparableInstincts, extractInstinctAction } from "./instinct-quality.js";
import {
	loadMergedInstincts,
	loadPendingInstincts,
	pendingTtlDays,
	prunePendingInstincts,
	removePendingInstincts,
	stagePendingDrafts,
	upsertDrafts,
} from "./instincts.js";
import { resolveActivePreferredOrDefaultModel } from "./model-selection.js";
import {
	archiveProcessedObservations,
	loadConfig,
	loadObserverState,
	readObservations,
	saveObserverState,
} from "./storage.js";
import type { InstinctDraft, ProjectInfo, StorageLayout } from "./types.js";

const OBSERVER_SYSTEM_PROMPT = `You analyze coding-agent observations and extract reusable instincts.

Return strict JSON only with this shape:
{
  "instincts": [
    {
      "id": "kebab-case-id",
      "title": "Short title",
      "trigger": "when ...",
      "confidence": 0.65,
      "domain": "workflow",
      "scope": "project",
      "action": "One clear sentence",
      "evidence": ["short bullet", "short bullet"]
    }
  ]
}

Rules:
- Create at most 5 instincts
- Ignore weak or one-off patterns
- Use confidence between 0.3 and 0.9
- Use scope "global" only for universal patterns
- Never include raw code snippets
- Prefer atomic, clusterable instincts over broad summary rules
- It is acceptable for multiple instincts to share the same trigger when they capture distinct steps or checks in the same workflow
- Do not emit paraphrased duplicates for the same concept. Choose one wording and one instinct id per concept
- Reuse existing instinct IDs only when the trigger and action are materially the same
- Avoid generic instincts like "follow repo conventions" or "write better tests"
- If evidence is tentative, still return the instinct with lower confidence rather than inventing a broad confident rule
- If nothing is worth learning, return {"instincts":[]}`;

export interface ObserverRuntimeState {
	running: boolean;
	timer: NodeJS.Timeout | null;
	scheduledAnalysis: NodeJS.Timeout | null;
	scheduledAnalysisAt?: number;
	lastAttemptedAt?: string;
	lastCompletedAt?: string;
	lastResult?: ObserverAnalysisResult;
	lastError?: string;
}

export interface ObserverAnalysisResult {
	learned: number;
	skippedReason?: string;
	retryAfterMs?: number;
}

function scrubText(text: string | undefined, maxLength: number): string | undefined {
	if (!text) {
		return undefined;
	}
	return text.replace(/\s+/gu, " ").trim().slice(0, maxLength);
}

function summarizeExistingInstincts(
	instincts: Array<{ id: string; trigger: string; content: string; confidence: number }>,
): string {
	if (instincts.length === 0) {
		return "(none)";
	}
	return instincts
		.slice(0, 12)
		.map((instinct) => {
			const action = extractInstinctAction(instinct.content, "no action");
			return `- ${instinct.id} (${Math.round(instinct.confidence * 100)}%): ${instinct.trigger} -> ${action}`;
		})
		.join("\n");
}

function buildObservationPrompt(
	project: ProjectInfo,
	recentObservations: ReturnType<typeof scrubText>[],
	existingInstincts: Array<{ id: string; trigger: string; content: string; confidence: number }>,
	pendingInstincts: Array<{ id: string; trigger: string; content: string; confidence: number }>,
): string {
	const lines = [
		`Project: ${project.name} (${project.id})`,
		project.remote ? `Remote: ${project.remote}` : "Remote: none",
		"",
		"Existing active instincts:",
		summarizeExistingInstincts(existingInstincts),
		"",
		"Existing pending instincts:",
		summarizeExistingInstincts(pendingInstincts),
		"",
		"Recent observations:",
	];
	for (const line of recentObservations) {
		if (line) {
			lines.push(`- ${line}`);
		}
	}
	return lines.join("\n");
}

function extractJsonPayload(text: string): string | null {
	const fenced = text.match(/```json\s*([\s\S]*?)```/u)?.[1];
	if (fenced) {
		return fenced.trim();
	}
	const bracesStart = text.indexOf("{");
	const bracesEnd = text.lastIndexOf("}");
	if (bracesStart >= 0 && bracesEnd > bracesStart) {
		return text.slice(bracesStart, bracesEnd + 1);
	}
	return null;
}

function validateDraft(value: unknown): InstinctDraft | null {
	if (!value || typeof value !== "object") {
		return null;
	}
	const record = value as Record<string, unknown>;
	if (typeof record.id !== "string" || record.id.trim().length === 0) {
		return null;
	}
	if (typeof record.trigger !== "string" || record.trigger.trim().length === 0) {
		return null;
	}
	if (typeof record.action !== "string" || record.action.trim().length === 0) {
		return null;
	}
	return {
		id: record.id.trim(),
		title:
			typeof record.title === "string" && record.title.trim().length > 0 ? record.title.trim() : record.id.trim(),
		trigger: record.trigger.trim(),
		confidence:
			typeof record.confidence === "number"
				? record.confidence
				: Number.parseFloat(String(record.confidence ?? "0.5")),
		domain: typeof record.domain === "string" && record.domain.trim().length > 0 ? record.domain.trim() : "general",
		scope: record.scope === "global" ? "global" : "project",
		action: record.action.trim(),
		evidence: Array.isArray(record.evidence)
			? record.evidence
					.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
					.slice(0, 5)
			: [],
	};
}

function parseDrafts(text: string): InstinctDraft[] {
	const payload = extractJsonPayload(text);
	if (!payload) {
		return [];
	}
	try {
		const parsed = JSON.parse(payload) as { instincts?: unknown[] };
		if (!Array.isArray(parsed.instincts)) {
			return [];
		}
		const drafts = parsed.instincts.map(validateDraft).filter((draft): draft is InstinctDraft => Boolean(draft));
		const deduped = new Map<string, InstinctDraft>();
		for (const draft of drafts) {
			deduped.set(draft.id, draft);
		}
		return Array.from(deduped.values());
	} catch {
		return [];
	}
}

function formatObservationForPrompt(observation: {
	event: string;
	inputText?: string;
	toolName?: string;
	toolInput?: string;
	toolOutput?: string;
	assistantText?: string;
	isError?: boolean;
	timestamp: string;
}): string {
	switch (observation.event) {
		case "input":
			return `${observation.timestamp} input: ${scrubText(observation.inputText, 400) ?? ""}`;
		case "tool_call":
			return `${observation.timestamp} tool_call ${observation.toolName}: ${scrubText(observation.toolInput, 300) ?? ""}`;
		case "tool_result":
			return `${observation.timestamp} tool_result ${observation.toolName}${observation.isError ? " [error]" : ""}: ${scrubText(observation.toolOutput, 400) ?? ""}`;
		case "turn_end":
			return `${observation.timestamp} assistant: ${scrubText(observation.assistantText, 400) ?? ""}`;
		default:
			return `${observation.timestamp} ${observation.event}`;
	}
}

export async function maybeAnalyzeObservations(
	ctx: ExtensionContext,
	project: ProjectInfo,
	layout: StorageLayout,
	runtime: ObserverRuntimeState,
): Promise<ObserverAnalysisResult> {
	runtime.lastAttemptedAt = new Date().toISOString();
	if (runtime.running || !ctx.isIdle()) {
		const result = { learned: 0, skippedReason: "busy", retryAfterMs: 60_000 } satisfies ObserverAnalysisResult;
		runtime.lastResult = result;
		return result;
	}

	const config = await loadConfig(layout);
	if (!config.observer.enabled) {
		const result = { learned: 0, skippedReason: "disabled" } satisfies ObserverAnalysisResult;
		runtime.lastResult = result;
		return result;
	}

	const observerState = await loadObserverState(layout);
	const observations = await readObservations(layout);
	const pendingObservations = observations.slice(observerState.lastAnalyzedIndex);
	if (pendingObservations.length < config.observer.minObservationsToAnalyze) {
		const result = { learned: 0, skippedReason: "insufficient-observations" } satisfies ObserverAnalysisResult;
		runtime.lastResult = result;
		return result;
	}

	const intervalMs = config.observer.runIntervalMinutes * 60_000;
	if (observerState.lastAnalyzedAt) {
		const elapsed = Date.now() - new Date(observerState.lastAnalyzedAt).getTime();
		if (elapsed < intervalMs) {
			const result = {
				learned: 0,
				skippedReason: "cooldown",
				retryAfterMs: Math.max(1_000, intervalMs - elapsed),
			} satisfies ObserverAnalysisResult;
			runtime.lastResult = result;
			return result;
		}
	}

	const resolvedModel = await resolveActivePreferredOrDefaultModel(
		ctx.model,
		config.observer.model,
		ctx.modelRegistry,
	);
	const model = resolvedModel.model;
	if (!model) {
		const result = { learned: 0, skippedReason: "no-model" } satisfies ObserverAnalysisResult;
		runtime.lastResult = result;
		return result;
	}

	const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
	if (!auth.ok || !auth.apiKey) {
		const result = {
			learned: 0,
			skippedReason: auth.ok ? "missing-api-key" : auth.error,
		} satisfies ObserverAnalysisResult;
		runtime.lastResult = result;
		return result;
	}

	await prunePendingInstincts(layout, pendingTtlDays(), false);
	const existingInstincts = await loadMergedInstincts(layout);
	const pendingInstincts = await loadPendingInstincts(layout);
	const sampled = pendingObservations
		.slice(-config.observer.maxRecentObservations)
		.map(formatObservationForPrompt)
		.filter((line) => line.trim().length > 0);

	const userMessage: UserMessage = {
		role: "user",
		content: [
			{
				type: "text",
				text: buildObservationPrompt(
					project,
					sampled,
					existingInstincts.map((instinct) => ({
						id: instinct.id,
						trigger: instinct.trigger,
						content: instinct.content,
						confidence: instinct.confidence,
					})),
					pendingInstincts.map((instinct) => ({
						id: instinct.id,
						trigger: instinct.trigger,
						content: instinct.content,
						confidence: instinct.confidence,
					})),
				),
			},
		],
		timestamp: Date.now(),
	};

	runtime.running = true;
	try {
		const response = await complete(
			model,
			{
				systemPrompt: OBSERVER_SYSTEM_PROMPT,
				messages: [userMessage],
			},
			{
				apiKey: auth.apiKey,
				headers: auth.headers,
				maxTokens: 4096,
			},
		);

		const text = response.content
			.filter((item): item is { type: "text"; text: string } => item.type === "text")
			.map((item) => item.text)
			.join("\n");
		const drafts = dedupeComparableInstincts(
			parseDrafts(text),
			[...existingInstincts, ...pendingInstincts].map((instinct) => ({
				id: instinct.id,
				title: instinct.title,
				trigger: instinct.trigger,
				action: extractInstinctAction(instinct.content),
			})),
		).kept;
		const activeDrafts = drafts.filter((draft) => draft.confidence >= 0.7);
		const pendingDrafts = drafts.filter((draft) => draft.confidence < 0.7);
		if (activeDrafts.length > 0) {
			await upsertDrafts(layout, project, existingInstincts, activeDrafts);
			await removePendingInstincts(
				layout,
				activeDrafts.map((draft) => draft.id),
			);
		}
		if (pendingDrafts.length > 0) {
			await stagePendingDrafts(layout, project, pendingDrafts);
		}
		await archiveProcessedObservations(layout, observations.length);
		await saveObserverState(layout, {
			lastAnalyzedIndex: 0,
			lastAnalyzedAt: new Date().toISOString(),
		});
		const result = { learned: drafts.length } satisfies ObserverAnalysisResult;
		runtime.lastResult = result;
		runtime.lastCompletedAt = new Date().toISOString();
		runtime.lastError = undefined;
		return result;
	} catch (error) {
		runtime.lastError = error instanceof Error ? error.message : String(error);
		const result = {
			learned: 0,
			skippedReason: "analysis-error",
			retryAfterMs: 60_000,
		} satisfies ObserverAnalysisResult;
		runtime.lastResult = result;
		throw error;
	} finally {
		runtime.running = false;
	}
}
