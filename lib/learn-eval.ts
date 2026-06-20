import { readFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { complete, type Model, type UserMessage } from "@earendil-works/pi-ai";
import {
	buildSessionContext,
	convertToLlm,
	getAgentDir,
	loadSkills,
	parseFrontmatter,
	serializeConversation,
} from "@earendil-works/pi-coding-agent";
import { detectOverlappingSkills, type ExistingSkillReference } from "./skill-overlap.js";
import { writeTextFile } from "./storage.js";
import type { InstinctScope, ProjectInfo, SkillCreateQualityReport } from "./types.js";

const LEARN_EVAL_SYSTEM_PROMPT = `You extract one reusable pattern from a coding-agent session and evaluate whether it should be saved.

Return exactly these sections:

<skill_markdown>
...full SKILL.md content including frontmatter...
</skill_markdown>

<quality_json>
{
  "verdict": "save | improve-then-save | absorb | drop",
  "rationale": "1-2 sentence rationale",
  "checklist": [
    "skills overlap: ...",
    "memory overlap: ...",
    "append vs new file: ...",
    "reusability: ..."
  ],
  "absorbTarget": "optional skill path or MEMORY.md",
  "improvements": ["optional improvement"],
  "scope": "project | global"
}
</quality_json>

Rules:
- Extract exactly one pattern
- Prefer the most reusable, highest-value pattern from the session
- Use the session transcript as the primary evidence source
- The pattern must be specific and actionable, not a generic best practice
- Use project scope for repo-specific knowledge and global scope only for patterns reusable across multiple projects
- If the pattern should append to an existing skill or MEMORY, return verdict=absorb
- If the pattern is too weak, trivial, or one-off, return verdict=drop
- The skill format should follow:
  - YAML frontmatter with name, description, user-invocable: false, origin: auto-extracted
  - Title
  - Extracted date
  - Context
  - Problem
  - Solution
  - When to Use`;

type ExistingSkillSummary = ExistingSkillReference;

export interface LearnEvalLlmContext {
	model: Model<any>;
	apiKey: string;
	headers?: Record<string, string>;
}

interface SessionManagerLike {
	getEntries(): Parameters<typeof buildSessionContext>[0];
	getLeafId(): string | null;
}

export interface LearnEvalOptions {
	project: ProjectInfo;
	sessionManager: SessionManagerLike;
	llm: LearnEvalLlmContext;
}

export interface LearnEvalResult {
	skillMarkdown: string | null;
	quality: SkillCreateQualityReport & {
		scope: InstinctScope;
	};
	targetPath: string | null;
	projectRoot: string;
	transcript: string;
}

async function readOptionalText(filePath: string, maxChars: number): Promise<string> {
	try {
		const content = await readFile(filePath, "utf-8");
		return content.slice(0, maxChars);
	} catch {
		return "";
	}
}

async function loadExistingSkills(projectRoot: string): Promise<ExistingSkillSummary[]> {
	const loaded = loadSkills({ cwd: projectRoot }).skills;
	const summaries: ExistingSkillSummary[] = [];
	for (const skill of loaded) {
		const raw = await readOptionalText(skill.filePath, 20000);
		summaries.push({
			name: skill.name,
			description: skill.description,
			filePath: skill.filePath,
			bodyText: raw,
		});
	}
	return summaries;
}

function extractTaggedSection(text: string, tag: string): string | null {
	const match = text.match(new RegExp(`<${tag}>\\s*([\\s\\S]*?)\\s*</${tag}>`, "u"));
	return match?.[1]?.trim() ?? null;
}

function parseQualityVerdict(value: unknown): SkillCreateQualityReport["verdict"] | undefined {
	return value === "save" || value === "improve-then-save" || value === "absorb" || value === "drop"
		? value
		: undefined;
}

function normalizeSkillMarkdown(raw: string, project: ProjectInfo): string {
	const { frontmatter, body } = parseFrontmatter<Record<string, unknown>>(raw);
	const lines = body.trim().length > 0 ? body.trim().split("\n") : [];
	const fallbackName = `${project.name.toLowerCase().replace(/[^a-z0-9-]+/gu, "-")}-learned`;
	const description =
		typeof frontmatter.description === "string" && frontmatter.description.trim().length > 0
			? frontmatter.description.trim()
			: `Learned pattern extracted from ${project.name}`;
	const frontmatterLines = [
		"---",
		`name: ${typeof frontmatter.name === "string" && frontmatter.name.trim().length > 0 ? frontmatter.name.trim() : fallbackName}`,
		`description: ${description}`,
		"user-invocable: false",
		"origin: auto-extracted",
		"---",
		"",
	];
	return [...frontmatterLines, ...lines].join("\n").trimEnd();
}

function validateQualityReport(value: unknown): (SkillCreateQualityReport & { scope: InstinctScope }) | undefined {
	if (!value || typeof value !== "object") {
		return undefined;
	}
	const record = value as Record<string, unknown>;
	const verdict = parseQualityVerdict(record.verdict);
	if (!verdict) {
		return undefined;
	}
	const checklist = Array.isArray(record.checklist)
		? record.checklist.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
		: [];
	return {
		verdict,
		rationale: typeof record.rationale === "string" ? record.rationale.trim() : "",
		checklist,
		overlapSkills: [],
		droppedInstinctIds: [],
		absorbTarget:
			typeof record.absorbTarget === "string" && record.absorbTarget.trim().length > 0
				? record.absorbTarget.trim()
				: undefined,
		improvements: Array.isArray(record.improvements)
			? record.improvements.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
			: undefined,
		scope: record.scope === "global" ? "global" : "project",
	};
}

function buildLearnEvalPrompt(
	project: ProjectInfo,
	transcript: string,
	existingSkills: ExistingSkillSummary[],
	projectMemory: string,
	globalMemory: string,
): string {
	const lines = [
		`Project: ${project.name}`,
		project.remote ? `Remote: ${project.remote}` : "Remote: none",
		"",
		"[EXISTING SKILLS]",
		existingSkills.length > 0
			? existingSkills
					.map(
						(skill) =>
							`- ${skill.name}: ${skill.description} (${skill.filePath})\n${skill.bodyText.slice(0, 1500)}`,
					)
					.join("\n\n")
			: "(none)",
		"",
		"[PROJECT MEMORY]",
		projectMemory || "(missing)",
		"",
		"[GLOBAL MEMORY]",
		globalMemory || "(missing)",
		"",
		"[SESSION TRANSCRIPT]",
		transcript,
		"",
		"[TASK]",
		"Review the session and extract the single most valuable reusable pattern.",
		"Then apply a learn-eval style quality gate, choose scope, and decide save / improve-then-save / absorb / drop.",
	];
	return lines.join("\n");
}

function buildAbsorbContent(skillMarkdown: string, absorbTarget: string | undefined): string {
	const body = stripFrontmatter(skillMarkdown);
	return [
		`# Suggested Additions For ${absorbTarget ?? "existing skill"}`,
		"",
		"```diff",
		"@@ append @@",
		...body.split("\n").map((line) => `+ ${line}`),
		"```",
	].join("\n");
}

function stripFrontmatter(skillMarkdown: string): string {
	return skillMarkdown.replace(/^---[\s\S]*?---\s*/u, "").trim();
}

async function resolveAbsorbTargetPath(result: LearnEvalResult): Promise<string | null> {
	const target = result.quality.absorbTarget;
	if (!target || target.trim().length === 0) {
		return null;
	}
	if (target === "MEMORY.md") {
		if (result.quality.scope === "global") {
			return join(getAgentDir(), "MEMORY.md");
		}
		const projectMemoryPath = join(result.projectRoot, ".pi", "MEMORY.md");
		if ((await readOptionalText(projectMemoryPath, 1)).length > 0) {
			return projectMemoryPath;
		}
		return join(result.projectRoot, "MEMORY.md");
	}
	if (target.startsWith("/")) {
		return target;
	}
	return join(result.projectRoot, target);
}

async function improveSkillOnce(
	project: ProjectInfo,
	llm: LearnEvalLlmContext,
	skillMarkdown: string,
	improvements: string[],
): Promise<string | null> {
	const userMessage: UserMessage = {
		role: "user",
		content: [
			{
				type: "text",
				text: [
					"Revise this learned skill once using the following improvements.",
					"",
					...improvements.map((item) => `- ${item}`),
					"",
					"Return only:",
					"<skill_markdown>",
					skillMarkdown,
					"</skill_markdown>",
				].join("\n"),
			},
		],
		timestamp: Date.now(),
	};
	try {
		const response = await complete(
			llm.model,
			{
				systemPrompt:
					"You revise learned skills. Preserve valid frontmatter and improve specificity, actionability, and scope fit. Return only <skill_markdown>.",
				messages: [userMessage],
			},
			{
				apiKey: llm.apiKey,
				headers: llm.headers,
				maxTokens: 4096,
				signal: AbortSignal.timeout(60_000),
			},
		);
		const text = response.content
			.filter((item): item is { type: "text"; text: string } => item.type === "text")
			.map((item) => item.text)
			.join("\n");
		const block = extractTaggedSection(text, "skill_markdown");
		return block ? normalizeSkillMarkdown(block, project) : null;
	} catch {
		return null;
	}
}

export async function evaluateSessionLearning(options: LearnEvalOptions): Promise<LearnEvalResult> {
	const entries = options.sessionManager.getEntries();
	const leafId = options.sessionManager.getLeafId();
	const sessionContext = buildSessionContext(entries, leafId);
	const transcript = serializeConversation(convertToLlm(sessionContext.messages));
	const existingSkills = await loadExistingSkills(options.project.root);
	const projectMemory = (
		(await readOptionalText(join(options.project.root, ".pi", "MEMORY.md"), 4000)) ||
		(await readOptionalText(join(options.project.root, "MEMORY.md"), 4000))
	).trim();
	const globalMemory = (await readOptionalText(join(getAgentDir(), "MEMORY.md"), 4000)).trim();

	const userMessage: UserMessage = {
		role: "user",
		content: [
			{
				type: "text",
				text: buildLearnEvalPrompt(options.project, transcript, existingSkills, projectMemory, globalMemory),
			},
		],
		timestamp: Date.now(),
	};

	const response = await complete(
		options.llm.model,
		{
			systemPrompt: LEARN_EVAL_SYSTEM_PROMPT,
			messages: [userMessage],
		},
		{
			apiKey: options.llm.apiKey,
			headers: options.llm.headers,
			maxTokens: 4096,
			signal: AbortSignal.timeout(60_000),
		},
	);
	const text = response.content
		.filter((item): item is { type: "text"; text: string } => item.type === "text")
		.map((item) => item.text)
		.join("\n");

	const rawSkill = extractTaggedSection(text, "skill_markdown");
	const rawQuality = extractTaggedSection(text, "quality_json");
	const skillMarkdown = rawSkill ? normalizeSkillMarkdown(rawSkill, options.project) : null;
	let parsedQuality: (SkillCreateQualityReport & { scope: InstinctScope }) | undefined;
	if (rawQuality) {
		try {
			parsedQuality = validateQualityReport(JSON.parse(rawQuality));
		} catch {
			parsedQuality = undefined;
		}
	}
	const quality = parsedQuality ?? {
		verdict: "drop",
		rationale: "模型未返回可解析的 learn-eval 结果。",
		checklist: [
			"skills overlap: unknown",
			"memory overlap: unknown",
			"append vs new file: unknown",
			"reusability: unknown",
		],
		overlapSkills: [],
		droppedInstinctIds: [],
		scope: "project" as const,
	};
	if (options.project.id === "global") {
		quality.scope = "global";
	}

	let finalSkillMarkdown = skillMarkdown;
	const finalQuality = { ...quality };

	if (
		quality.verdict === "improve-then-save" &&
		skillMarkdown &&
		quality.improvements &&
		quality.improvements.length > 0
	) {
		const improved = await improveSkillOnce(options.project, options.llm, skillMarkdown, quality.improvements);
		if (improved) {
			finalSkillMarkdown = improved;
			finalQuality.revised = true;
			finalQuality.verdict = "save";
		}
	}

	if (finalQuality.verdict === "absorb" && finalSkillMarkdown) {
		finalQuality.absorbContent = buildAbsorbContent(finalSkillMarkdown, finalQuality.absorbTarget);
	}

	if (finalSkillMarkdown && finalQuality.verdict !== "drop") {
		const overlaps = detectOverlappingSkills(finalSkillMarkdown, existingSkills, { limit: 3, threshold: 0.52 });
		if (overlaps.length > 0 && finalQuality.verdict !== "absorb") {
			finalQuality.verdict = "absorb";
			finalQuality.absorbTarget = overlaps[0]?.filePath;
			finalQuality.rationale =
				finalQuality.rationale.length > 0
					? `${finalQuality.rationale} 本地 overlap 检查还发现与现有 skill 正文存在较强重叠。`
					: "本地 overlap 检查发现与现有 skill 正文存在较强重叠。";
		}
		if (overlaps.length > 0) {
			finalQuality.overlapSkills = overlaps.map((item) => item.filePath);
			finalQuality.checklist = [
				...finalQuality.checklist,
				`local skill overlap: ${overlaps.map((item) => item.filePath).join(", ")}`,
			];
			if (finalQuality.verdict === "absorb") {
				finalQuality.absorbContent = buildAbsorbContent(finalSkillMarkdown, finalQuality.absorbTarget);
			}
		}
	}

	const slugSource =
		finalSkillMarkdown &&
		(parseFrontmatter<Record<string, unknown>>(finalSkillMarkdown).frontmatter.name as string | undefined);
	const slug = (slugSource ? basename(String(slugSource)) : `${options.project.name.toLowerCase()}-learned`).replace(
		/[^a-zA-Z0-9._-]+/gu,
		"-",
	);
	const targetPath =
		finalQuality.verdict === "drop" || finalQuality.verdict === "absorb"
			? null
			: finalQuality.scope === "global"
				? join(getAgentDir(), "skills", "learned", slug, "SKILL.md")
				: join(options.project.root, ".pi", "skills", "learned", slug, "SKILL.md");

	return {
		skillMarkdown: finalSkillMarkdown,
		quality: finalQuality,
		targetPath,
		projectRoot: options.project.root,
		transcript,
	};
}

export async function applyLearnEvalResult(result: LearnEvalResult): Promise<void> {
	if (result.quality.verdict === "absorb") {
		const absorbPath = await resolveAbsorbTargetPath(result);
		if (absorbPath && result.skillMarkdown) {
			const existing = await readOptionalText(absorbPath, 1_000_000);
			const addition = stripFrontmatter(result.skillMarkdown);
			const heading = absorbPath.endsWith("MEMORY.md") ? "## Learned Pattern" : "## Learned Addition";
			const next =
				existing.trim().length > 0
					? `${existing.trimEnd()}\n\n${heading}\n\n${addition}\n`
					: `${heading}\n\n${addition}\n`;
			await writeTextFile(absorbPath, next);
		}
		return;
	}
	if (!result.skillMarkdown || !result.targetPath) {
		return;
	}
	await writeTextFile(result.targetPath, result.skillMarkdown);
}
