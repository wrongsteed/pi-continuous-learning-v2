import { readdir, readFile, stat, unlink } from "node:fs/promises";
import { basename, extname, join } from "node:path";
import { parseFrontmatter } from "@earendil-works/pi-coding-agent";
import { extractInstinctAction, normalizeTriggerClusterKey, renderWhenClause } from "./instinct-quality.js";
import { fileExists, isDirectory, loadProjectRegistry, writeTextFile } from "./storage.js";
import type {
	ClusterCandidate,
	EvolveAnalysis,
	ImportSummary,
	InstinctDraft,
	InstinctRecord,
	InstinctScope,
	LoadedInstinct,
	ProjectInfo,
	StorageLayout,
} from "./types.js";

const ALLOWED_EXTENSIONS = new Set([".md", ".yaml", ".yml"]);
const PENDING_TTL_DAYS = 30;
const PENDING_EXPIRY_WARNING_DAYS = 7;

function clampConfidence(value: number): number {
	return Math.max(0.3, Math.min(0.95, Number.isFinite(value) ? value : 0.5));
}

function normalizeString(value: string): string {
	return value.trim().replace(/\s+/gu, " ");
}

function slugify(value: string): string {
	return value
		.toLowerCase()
		.replace(/[^a-z0-9]+/gu, "-")
		.replace(/^-+/u, "")
		.replace(/-+$/u, "")
		.slice(0, 40);
}

function humanizeSlug(value: string): string {
	return value
		.split("-")
		.filter((part) => part.length > 0)
		.map((part) => part[0]?.toUpperCase() + part.slice(1))
		.join(" ");
}

function buildFrontmatterLines(entries: Array<[string, string | string[]]>): string[] {
	const lines = ["---"];
	for (const [key, value] of entries) {
		if (Array.isArray(value)) {
			lines.push(`${key}:`);
			for (const item of value) {
				lines.push(`  - ${item}`);
			}
			continue;
		}
		lines.push(`${key}: ${value}`);
	}
	lines.push("---", "");
	return lines;
}

function inferTitle(body: string, fallback: string): string {
	const heading = body.match(/^#\s+(.+)$/mu)?.[1]?.trim();
	return heading && heading.length > 0 ? heading : fallback;
}

function buildInstinctContent(title: string, action: string, evidence: string[]): string {
	const lines = [`# ${title}`, "", "## Action", action.trim(), "", "## Evidence"];
	if (evidence.length === 0) {
		lines.push("- Inferred from repeated observations");
	} else {
		for (const item of evidence) {
			lines.push(`- ${item}`);
		}
	}
	return lines.join("\n");
}

function parseScope(value: unknown, fallback: InstinctScope): InstinctScope {
	return value === "global" ? "global" : fallback;
}

async function loadInstinctFile(
	filePath: string,
	sourceType: "personal" | "inherited",
	scopeLabel: InstinctScope,
): Promise<LoadedInstinct | null> {
	try {
		const raw = await readFile(filePath, "utf-8");
		const { frontmatter, body } = parseFrontmatter<Record<string, unknown>>(raw);
		const id = typeof frontmatter.id === "string" ? frontmatter.id.trim() : basename(filePath, extname(filePath));
		if (!id) {
			return null;
		}

		const title = typeof frontmatter.title === "string" ? frontmatter.title.trim() : inferTitle(body, id);
		const trigger = typeof frontmatter.trigger === "string" ? frontmatter.trigger.trim() : `when ${id}`;
		const confidence =
			typeof frontmatter.confidence === "number"
				? frontmatter.confidence
				: Number.parseFloat(String(frontmatter.confidence ?? "0.5"));
		const domain = typeof frontmatter.domain === "string" ? frontmatter.domain.trim() : "general";
		const source = typeof frontmatter.source === "string" ? frontmatter.source.trim() : sourceType;
		const scope = parseScope(frontmatter.scope, scopeLabel);

		return {
			id,
			title,
			trigger,
			confidence: clampConfidence(confidence),
			domain,
			source,
			scope,
			projectId: typeof frontmatter.project_id === "string" ? frontmatter.project_id : undefined,
			projectName: typeof frontmatter.project_name === "string" ? frontmatter.project_name : undefined,
			content: body.trim(),
			created: typeof frontmatter.created === "string" ? frontmatter.created : undefined,
			updated: typeof frontmatter.updated === "string" ? frontmatter.updated : undefined,
			importedFrom: typeof frontmatter.imported_from === "string" ? frontmatter.imported_from : undefined,
			promotedFrom: typeof frontmatter.promoted_from === "string" ? frontmatter.promoted_from : undefined,
			filePath,
			sourceType,
			scopeLabel,
		};
	} catch {
		return null;
	}
}

async function loadInstinctsFromDir(
	dirPath: string,
	sourceType: "personal" | "inherited",
	scopeLabel: InstinctScope,
): Promise<LoadedInstinct[]> {
	if (!(await fileExists(dirPath))) {
		return [];
	}

	const entries = await readdir(dirPath, { withFileTypes: true });
	const instincts: LoadedInstinct[] = [];
	for (const entry of entries) {
		if (!entry.isFile()) {
			continue;
		}
		if (!ALLOWED_EXTENSIONS.has(extname(entry.name).toLowerCase())) {
			continue;
		}
		const instinct = await loadInstinctFile(join(dirPath, entry.name), sourceType, scopeLabel);
		if (instinct) {
			instincts.push(instinct);
		}
	}
	return instincts;
}

async function loadPendingInstinctsFromDir(dirPath: string, scopeLabel: InstinctScope): Promise<LoadedInstinct[]> {
	const instincts = await loadInstinctsFromDir(dirPath, "personal", scopeLabel);
	return instincts.map((instinct) => ({
		...instinct,
		sourceType: "personal",
	}));
}

export async function loadProjectOnlyInstincts(layout: StorageLayout): Promise<LoadedInstinct[]> {
	if (layout.isGlobalProject) {
		return [
			...(await loadInstinctsFromDir(layout.globalPersonalDir, "personal", "global")),
			...(await loadInstinctsFromDir(layout.globalInheritedDir, "inherited", "global")),
		];
	}
	const personal = await loadInstinctsFromDir(layout.projectPersonalDir, "personal", "project");
	const inherited = await loadInstinctsFromDir(layout.projectInheritedDir, "inherited", "project");
	return [...personal, ...inherited];
}

export async function loadPendingInstincts(layout: StorageLayout): Promise<LoadedInstinct[]> {
	if (layout.isGlobalProject) {
		return loadPendingInstinctsFromDir(layout.globalPendingDir, "global");
	}
	return [
		...(await loadPendingInstinctsFromDir(layout.projectPendingDir, "project")),
		...(await loadPendingInstinctsFromDir(layout.globalPendingDir, "global")),
	];
}

export async function loadMergedInstincts(layout: StorageLayout): Promise<LoadedInstinct[]> {
	if (layout.isGlobalProject) {
		return [
			...(await loadInstinctsFromDir(layout.globalPersonalDir, "personal", "global")),
			...(await loadInstinctsFromDir(layout.globalInheritedDir, "inherited", "global")),
		];
	}
	const merged = new Map<string, LoadedInstinct>();
	const ordered = [
		...(await loadInstinctsFromDir(layout.projectPersonalDir, "personal", "project")),
		...(await loadInstinctsFromDir(layout.projectInheritedDir, "inherited", "project")),
		...(await loadInstinctsFromDir(layout.globalPersonalDir, "personal", "global")),
		...(await loadInstinctsFromDir(layout.globalInheritedDir, "inherited", "global")),
	];

	for (const instinct of ordered) {
		if (!merged.has(instinct.id)) {
			merged.set(instinct.id, instinct);
		}
	}
	return Array.from(merged.values());
}

export function serializeInstinct(instinct: InstinctRecord): string {
	const frontmatter = [
		"---",
		`id: ${instinct.id}`,
		`title: ${JSON.stringify(instinct.title)}`,
		`trigger: ${JSON.stringify(instinct.trigger)}`,
		`confidence: ${clampConfidence(instinct.confidence)}`,
		`domain: ${instinct.domain}`,
		`source: ${instinct.source}`,
		`scope: ${instinct.scope}`,
	];
	if (instinct.projectId) {
		frontmatter.push(`project_id: ${instinct.projectId}`);
	}
	if (instinct.projectName) {
		frontmatter.push(`project_name: ${JSON.stringify(instinct.projectName)}`);
	}
	if (instinct.created) {
		frontmatter.push(`created: ${instinct.created}`);
	}
	if (instinct.updated) {
		frontmatter.push(`updated: ${instinct.updated}`);
	}
	if (instinct.importedFrom) {
		frontmatter.push(`imported_from: ${JSON.stringify(instinct.importedFrom)}`);
	}
	if (instinct.promotedFrom) {
		frontmatter.push(`promoted_from: ${instinct.promotedFrom}`);
	}
	frontmatter.push("---", "");
	return `${frontmatter.join("\n")}${instinct.content.trim()}`;
}

export async function upsertDrafts(
	layout: StorageLayout,
	project: ProjectInfo,
	existingInstincts: LoadedInstinct[],
	drafts: InstinctDraft[],
): Promise<LoadedInstinct[]> {
	const existingById = new Map(existingInstincts.map((instinct) => [instinct.id, instinct]));
	const written: LoadedInstinct[] = [];

	for (const draft of drafts) {
		const existing = existingById.get(draft.id);
		const scope = layout.isGlobalProject ? "global" : draft.scope === "global" ? "global" : "project";
		const targetDir = scope === "global" ? layout.globalPersonalDir : layout.projectPersonalDir;
		const filePath = join(targetDir, `${draft.id}.md`);
		const now = new Date().toISOString();
		const record: InstinctRecord = {
			id: draft.id,
			title: normalizeString(draft.title || draft.id),
			trigger: normalizeString(draft.trigger || `when ${draft.id}`),
			confidence: clampConfidence(Math.max(existing?.confidence ?? 0.3, draft.confidence)),
			domain: normalizeString(draft.domain || existing?.domain || "general"),
			source: "session-observation",
			scope,
			projectId: scope === "project" ? project.id : undefined,
			projectName: scope === "project" ? project.name : undefined,
			content: buildInstinctContent(draft.title || draft.id, draft.action, draft.evidence),
			created: existing?.created ?? now,
			updated: now,
		};
		await writeTextFile(filePath, serializeInstinct(record));
		written.push({
			...record,
			filePath,
			sourceType: "personal",
			scopeLabel: scope,
		});
	}

	return written;
}

export async function stagePendingDrafts(
	layout: StorageLayout,
	project: ProjectInfo,
	drafts: InstinctDraft[],
): Promise<LoadedInstinct[]> {
	const existing = await loadPendingInstincts(layout);
	const existingById = new Map(existing.map((instinct) => [instinct.id, instinct]));
	const written: LoadedInstinct[] = [];

	for (const draft of drafts) {
		const scope = layout.isGlobalProject ? "global" : draft.scope === "global" ? "global" : "project";
		const targetDir = scope === "global" ? layout.globalPendingDir : layout.projectPendingDir;
		const filePath = join(targetDir, `${draft.id}.md`);
		const now = new Date().toISOString();
		const previous = existingById.get(draft.id);
		const record: InstinctRecord = {
			id: draft.id,
			title: normalizeString(draft.title || draft.id),
			trigger: normalizeString(draft.trigger || `when ${draft.id}`),
			confidence: clampConfidence(draft.confidence),
			domain: normalizeString(draft.domain || previous?.domain || "general"),
			source: "session-observation-pending",
			scope,
			projectId: scope === "project" ? project.id : undefined,
			projectName: scope === "project" ? project.name : undefined,
			content: buildInstinctContent(draft.title || draft.id, draft.action, draft.evidence),
			created: previous?.created ?? now,
			updated: now,
		};
		await writeTextFile(filePath, serializeInstinct(record));
		written.push({
			...record,
			filePath,
			sourceType: "personal",
			scopeLabel: scope,
		});
	}

	return written;
}

export async function removePendingInstincts(layout: StorageLayout, instinctIds: string[]): Promise<void> {
	for (const instinctId of instinctIds) {
		const dirs = layout.isGlobalProject
			? [layout.globalPendingDir]
			: [layout.projectPendingDir, layout.globalPendingDir];
		for (const dir of dirs) {
			const filePath = join(dir, `${instinctId}.md`);
			if (await fileExists(filePath)) {
				await unlink(filePath).catch(() => {});
			}
		}
	}
}

export function parseInstinctExport(content: string): LoadedInstinct[] {
	const chunks = content
		.split(/^---$/mu)
		.map((chunk) => chunk.trim())
		.filter((chunk) => chunk.length > 0);

	const instincts: LoadedInstinct[] = [];
	for (let index = 0; index < chunks.length; index += 2) {
		const rawFrontmatter = chunks[index];
		const rawBody = chunks[index + 1] ?? "";
		const frontmatter: Record<string, string> = {};
		for (const line of rawFrontmatter.split("\n")) {
			const separatorIndex = line.indexOf(":");
			if (separatorIndex <= 0) {
				continue;
			}
			const key = line.slice(0, separatorIndex).trim();
			const value = line
				.slice(separatorIndex + 1)
				.trim()
				.replace(/^"|"$/gu, "");
			frontmatter[key] = value;
		}
		if (!frontmatter.id) {
			continue;
		}
		instincts.push({
			id: frontmatter.id,
			title: frontmatter.title ?? inferTitle(rawBody, frontmatter.id),
			trigger: frontmatter.trigger ?? `when ${frontmatter.id}`,
			confidence: clampConfidence(Number.parseFloat(frontmatter.confidence ?? "0.5")),
			domain: frontmatter.domain ?? "general",
			source: frontmatter.source ?? "inherited",
			scope: parseScope(frontmatter.scope, "project"),
			projectId: frontmatter.project_id,
			projectName: frontmatter.project_name,
			content: rawBody.trim(),
			created: frontmatter.created,
			updated: frontmatter.updated,
			importedFrom: frontmatter.imported_from,
			promotedFrom: frontmatter.promoted_from,
			filePath: "",
			sourceType: "inherited",
			scopeLabel: parseScope(frontmatter.scope, "project"),
		});
	}
	return instincts;
}

export async function importInstincts(
	layout: StorageLayout,
	project: ProjectInfo,
	sourceName: string,
	instincts: LoadedInstinct[],
	targetScope: InstinctScope,
	minConfidence: number | undefined,
	dryRun: boolean,
): Promise<ImportSummary> {
	targetScope = layout.isGlobalProject && targetScope === "project" ? "global" : targetScope;
	const targetDir = targetScope === "global" ? layout.globalInheritedDir : layout.projectInheritedDir;
	const currentInstincts =
		targetScope === "global"
			? [
					...(await loadInstinctsFromDir(layout.globalPersonalDir, "personal", "global")),
					...(await loadInstinctsFromDir(layout.globalInheritedDir, "inherited", "global")),
				]
			: [
					...(await loadInstinctsFromDir(layout.projectPersonalDir, "personal", "project")),
					...(await loadInstinctsFromDir(layout.projectInheritedDir, "inherited", "project")),
				];
	const existingById = new Map(currentInstincts.map((instinct) => [instinct.id, instinct]));
	const summary: ImportSummary = {
		added: [],
		updated: [],
		skipped: [],
	};
	const dedupedInstincts = new Map<string, LoadedInstinct>();
	const staleFilePaths = new Set<string>();
	for (const instinct of instincts) {
		const existing = dedupedInstincts.get(instinct.id);
		if (!existing || instinct.confidence > existing.confidence) {
			dedupedInstincts.set(instinct.id, instinct);
		}
	}

	for (const instinct of dedupedInstincts.values()) {
		if (minConfidence !== undefined && instinct.confidence < minConfidence) {
			summary.skipped.push(instinct);
			continue;
		}
		const existing = existingById.get(instinct.id);
		if (!existing) {
			const next = {
				...instinct,
				scope: targetScope,
				scopeLabel: targetScope,
				projectId: targetScope === "project" ? project.id : undefined,
				projectName: targetScope === "project" ? project.name : undefined,
				importedFrom: sourceName,
			};
			summary.added.push(next);
			if (!dryRun) {
				await writeTextFile(join(targetDir, `${instinct.id}.md`), serializeInstinct(next));
			}
			continue;
		}
		if (instinct.confidence > existing.confidence) {
			const targetFilePath = join(targetDir, `${instinct.id}.md`);
			for (const current of currentInstincts) {
				if (current.id === instinct.id && current.filePath !== targetFilePath) {
					staleFilePaths.add(current.filePath);
				}
			}
			const next = {
				...existing,
				...instinct,
				scope: targetScope,
				scopeLabel: targetScope,
				projectId: targetScope === "project" ? project.id : undefined,
				projectName: targetScope === "project" ? project.name : undefined,
				importedFrom: sourceName,
				filePath: join(targetDir, `${instinct.id}.md`),
				sourceType: "inherited" as const,
			};
			summary.updated.push(next);
			if (!dryRun) {
				await writeTextFile(join(targetDir, `${instinct.id}.md`), serializeInstinct(next));
			}
			continue;
		}
		summary.skipped.push(existing);
	}

	if (!dryRun) {
		for (const filePath of staleFilePaths) {
			await unlink(filePath).catch(() => {});
		}
	}

	return summary;
}

export function renderInstinctExport(
	instincts: LoadedInstinct[],
	options?: {
		scope?: "project" | "global" | "all";
		project?: ProjectInfo;
	},
): string {
	const header = ["# Instincts export", `# Date: ${new Date().toISOString()}`, `# Total: ${instincts.length}`];
	if (options?.scope) {
		header.push(`# Scope: ${options.scope}`);
	}
	if (options?.project && options.project.id !== "global") {
		header.push(`# Project: ${options.project.name} (${options.project.id})`);
	}
	header.push("");
	return header.concat(instincts.map((instinct) => serializeInstinct(instinct))).join("\n\n");
}

export interface PendingInstinctInfo {
	path: string;
	name: string;
	scope: InstinctScope;
	created: string;
	ageDays: number;
}

async function collectPendingDirs(
	layout: StorageLayout,
	includeAllProjects: boolean,
): Promise<Array<{ dir: string; scope: InstinctScope }>> {
	const dirs = new Map<string, InstinctScope>();
	const addDir = (dir: string, scope: InstinctScope) => {
		if (!dirs.has(dir)) {
			dirs.set(dir, scope);
		}
	};

	if (layout.isGlobalProject) {
		addDir(layout.globalPendingDir, "global");
	} else {
		addDir(layout.projectPendingDir, "project");
		addDir(layout.globalPendingDir, "global");
	}

	if (!includeAllProjects) {
		return Array.from(dirs.entries()).map(([dir, scope]) => ({ dir, scope }));
	}

	const registry = await loadProjectRegistry(layout);
	for (const project of Object.values(registry)) {
		addDir(join(project.root, ".pi", "continuous-learning-v2", "instincts", "pending"), "project");
	}

	return Array.from(dirs.entries()).map(([dir, scope]) => ({ dir, scope }));
}

async function parsePendingCreatedDate(filePath: string): Promise<Date | null> {
	try {
		const raw = await readFile(filePath, "utf-8");
		const { frontmatter } = parseFrontmatter<Record<string, unknown>>(raw);
		if (typeof frontmatter.created === "string" && frontmatter.created.trim().length > 0) {
			const date = new Date(frontmatter.created);
			if (!Number.isNaN(date.getTime())) {
				return date;
			}
		}
	} catch {}
	try {
		const info = await stat(filePath);
		return info.mtime;
	} catch {
		return null;
	}
}

export async function collectPendingInstincts(
	layout: StorageLayout,
	options?: { includeAllProjects?: boolean },
): Promise<PendingInstinctInfo[]> {
	const dirs = await collectPendingDirs(layout, options?.includeAllProjects ?? false);
	const now = Date.now();
	const results: PendingInstinctInfo[] = [];
	for (const { dir, scope } of dirs) {
		if (!(await fileExists(dir))) {
			continue;
		}
		const entries = await readdir(dir, { withFileTypes: true });
		for (const entry of entries) {
			if (!entry.isFile() || !ALLOWED_EXTENSIONS.has(extname(entry.name).toLowerCase())) {
				continue;
			}
			const filePath = join(dir, entry.name);
			const created = await parsePendingCreatedDate(filePath);
			if (!created) {
				continue;
			}
			const ageDays = Math.floor((now - created.getTime()) / (24 * 60 * 60 * 1000));
			results.push({
				path: filePath,
				name: basename(entry.name, extname(entry.name)),
				scope,
				created: created.toISOString(),
				ageDays,
			});
		}
	}
	return results.sort((left, right) => right.ageDays - left.ageDays);
}

export async function prunePendingInstincts(
	layout: StorageLayout,
	maxAge: number = PENDING_TTL_DAYS,
	dryRun: boolean = false,
): Promise<{ pruned: PendingInstinctInfo[]; remaining: PendingInstinctInfo[] }> {
	const pending = await collectPendingInstincts(layout, { includeAllProjects: true });
	const pruned = pending.filter((item) => item.ageDays >= maxAge);
	const remaining = pending.filter((item) => item.ageDays < maxAge);
	if (!dryRun) {
		for (const item of pruned) {
			await unlink(item.path).catch(() => {});
		}
	}
	return { pruned, remaining };
}

export function pendingExpiryThresholdDays(): number {
	return PENDING_TTL_DAYS - PENDING_EXPIRY_WARNING_DAYS;
}

export function pendingTtlDays(): number {
	return PENDING_TTL_DAYS;
}

export function analyzeEvolution(instincts: LoadedInstinct[]): EvolveAnalysis {
	const groups = new Map<string, LoadedInstinct[]>();
	for (const instinct of instincts) {
		const key = normalizeTriggerClusterKey(instinct.trigger, instinct.domain);
		const existing = groups.get(key) ?? [];
		existing.push(instinct);
		groups.set(key, existing);
	}

	const clusterCandidates: ClusterCandidate[] = Array.from(groups.entries())
		.map(([key, group]) => {
			const averageConfidence = group.reduce((sum, instinct) => sum + instinct.confidence, 0) / group.length;
			const slug = slugify(key) || slugify(group[0].trigger) || group[0].id;
			return {
				key: slug,
				title: humanizeSlug(slug) || group[0].title,
				trigger: group[0].trigger,
				instincts: [...group].sort((left, right) => right.confidence - left.confidence),
				averageConfidence,
				domains: Array.from(new Set(group.map((instinct) => instinct.domain))),
				scopes: Array.from(new Set(group.map((instinct) => instinct.scopeLabel))),
			};
		})
		.sort((left, right) => {
			const sizeDelta = right.instincts.length - left.instincts.length;
			if (sizeDelta !== 0) {
				return sizeDelta;
			}
			return right.averageConfidence - left.averageConfidence;
		});

	return {
		skillCandidates: clusterCandidates.filter((candidate) => candidate.instincts.length >= 2),
		promptCandidates: instincts
			.filter((instinct) => instinct.domain === "workflow" && instinct.confidence >= 0.7)
			.sort((left, right) => right.confidence - left.confidence),
		agentCandidates: clusterCandidates.filter(
			(candidate) => candidate.instincts.length >= 3 && candidate.averageConfidence >= 0.75,
		),
	};
}

function resolveScopedEvolvedDir(
	layout: StorageLayout,
	scope: InstinctScope,
	type: "skills" | "prompts" | "agents",
): string {
	if (scope === "global") {
		switch (type) {
			case "skills":
				return layout.globalEvolvedSkillsDir;
			case "prompts":
				return layout.globalEvolvedPromptsDir;
			case "agents":
				return layout.globalEvolvedAgentsDir;
		}
	}
	switch (type) {
		case "skills":
			return layout.projectEvolvedSkillsDir;
		case "prompts":
			return layout.projectEvolvedPromptsDir;
		case "agents":
			return layout.projectEvolvedAgentsDir;
	}
}

export async function generateEvolvedOutputs(layout: StorageLayout, analysis: EvolveAnalysis): Promise<string[]> {
	const generated: string[] = [];

	for (const candidate of analysis.skillCandidates.slice(0, 5)) {
		const slug = slugify(candidate.key) || "instinct-skill";
		const dominantScope = candidate.scopes.includes("project") ? "project" : "global";
		const skillPath = join(resolveScopedEvolvedDir(layout, dominantScope, "skills"), slug, "SKILL.md");
		const evolvedFrom = candidate.instincts.map((instinct) => instinct.id);
		const body = [
			...buildFrontmatterLines([
				["name", slug],
				["description", `Evolved from ${candidate.instincts.length} instincts about ${candidate.trigger}`],
				["evolved_from", evolvedFrom],
			]),
			`# ${candidate.title}`,
			"",
			`Use this skill when ${renderWhenClause(candidate.trigger)}.`,
			"",
			`Evolved from ${candidate.instincts.length} instincts (avg confidence: ${Math.round(candidate.averageConfidence * 100)}%).`,
			"",
			"## Domains",
			...candidate.domains.map((domain) => `- ${domain}`),
			"",
			"## Actions",
			...candidate.instincts.map((instinct) => `- ${extractActionLine(instinct.content)}`),
			"",
			"## Source Instincts",
			...candidate.instincts.map(
				(instinct) => `- ${instinct.id} [${instinct.scopeLabel}] (${Math.round(instinct.confidence * 100)}%)`,
			),
		].join("\n");
		await writeTextFile(skillPath, body);
		generated.push(skillPath);
	}

	for (const instinct of analysis.promptCandidates.slice(0, 5)) {
		const slug =
			slugify(
				instinct.trigger
					.toLowerCase()
					.replace(/^when\s+/u, "")
					.replace(/^implementing\s+/u, "")
					.replace(/^creating\s+/u, ""),
			) ||
			slugify(instinct.id) ||
			"instinct-prompt";
		const promptPath = join(resolveScopedEvolvedDir(layout, instinct.scopeLabel, "prompts"), `${slug}.md`);
		if (generated.includes(promptPath)) {
			continue;
		}
		const body = [
			...buildFrontmatterLines([
				["description", `Evolved workflow prompt from instinct ${instinct.id}`],
				["evolved_from", [instinct.id]],
			]),
			`# ${humanizeSlug(slug) || instinct.title}`,
			"",
			`When handling tasks matching "${renderWhenClause(instinct.trigger)}", apply this workflow:`,
			"",
			"## Steps",
			`1. ${extractActionLine(instinct.content)}`,
			"",
			"## Source Instinct",
			`- ${instinct.id} [${instinct.scopeLabel}] (${Math.round(instinct.confidence * 100)}%)`,
		].join("\n");
		await writeTextFile(promptPath, body);
		generated.push(promptPath);
	}

	for (const candidate of analysis.agentCandidates.slice(0, 3)) {
		const slug = slugify(candidate.key) || "instinct-agent";
		const dominantScope = candidate.scopes.includes("project") ? "project" : "global";
		const agentPath = join(resolveScopedEvolvedDir(layout, dominantScope, "agents"), `${slug}-agent.md`);
		const body = [
			...buildFrontmatterLines([
				["name", `${slug}-agent`],
				["description", `Evolved from ${candidate.instincts.length} instincts about ${candidate.trigger}`],
				["model", "sonnet"],
				["execution_mode", "manual-artifact-only"],
				["evolved_from", candidate.instincts.map((instinct) => instinct.id)],
			]),
			`# ${candidate.title} Agent`,
			"",
			`Use this agent when ${renderWhenClause(candidate.trigger)} and the task benefits from a dedicated multi-step flow.`,
			"",
			"## Source Domains",
			...candidate.domains.map((domain) => `- ${domain}`),
			"",
			"## Source Instincts",
			...candidate.instincts.map(
				(instinct) => `- ${instinct.id} [${instinct.scopeLabel}] (${Math.round(instinct.confidence * 100)}%)`,
			),
			"",
			"## Actions",
			...candidate.instincts.map((instinct) => `- ${extractActionLine(instinct.content)}`),
			"",
			"## Execution",
			"This generated agent is a markdown artifact only. It is not auto-executed by pi and must be consumed manually or by another extension.",
		].join("\n");
		await writeTextFile(agentPath, body);
		generated.push(agentPath);
	}

	return generated;
}

function extractActionLine(content: string): string {
	return extractInstinctAction(content, "Apply the learned pattern");
}

export async function findPromotionCandidates(layout: StorageLayout): Promise<
	Array<{
		id: string;
		entries: LoadedInstinct[];
		averageConfidence: number;
	}>
> {
	const registry = await loadProjectRegistry(layout);
	if (Object.keys(registry).length === 0) {
		return [];
	}
	const groups = new Map<string, LoadedInstinct[]>();

	for (const project of Object.values(registry)) {
		const projectStateDir = join(project.root, ".pi", "continuous-learning-v2");
		if (!(await isDirectory(projectStateDir))) {
			continue;
		}
		const personal = await loadInstinctsFromDir(
			join(projectStateDir, "instincts", "personal"),
			"personal",
			"project",
		);
		const inherited = await loadInstinctsFromDir(
			join(projectStateDir, "instincts", "inherited"),
			"inherited",
			"project",
		);
		const seenInProject = new Set<string>();
		for (const instinct of [...personal, ...inherited]) {
			if (seenInProject.has(instinct.id)) {
				continue;
			}
			seenInProject.add(instinct.id);
			const group = groups.get(instinct.id) ?? [];
			group.push(instinct);
			groups.set(instinct.id, group);
		}
	}

	return Array.from(groups.entries())
		.map(([id, entries]) => ({
			id,
			entries,
			averageConfidence: entries.reduce((sum, instinct) => sum + instinct.confidence, 0) / entries.length,
		}))
		.filter((candidate) => candidate.entries.length >= 2 && candidate.averageConfidence >= 0.8)
		.sort((left, right) => right.averageConfidence - left.averageConfidence);
}
