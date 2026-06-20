import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { basename, extname, join, resolve } from "node:path";
import { promisify } from "node:util";
import { complete, type Model, type UserMessage } from "@earendil-works/pi-ai";
import { getAgentDir, loadSkills, type ModelRegistry, parseFrontmatter } from "@earendil-works/pi-coding-agent";
import { runAgenticSkillCreate } from "./agentic-skill-create.js";
import {
	dedupeComparableInstincts,
	extractInstinctAction,
	normalizeCompareText,
	overlapScore,
} from "./instinct-quality.js";
import { loadProjectOnlyInstincts, serializeInstinct } from "./instincts.js";
import { detectOverlappingSkills, type ExistingSkillReference } from "./skill-overlap.js";
import { writeTextFile } from "./storage.js";
import type { ProjectInfo, SkillCreateQualityReport, StorageLayout } from "./types.js";

const execFileAsync = promisify(execFile);

const MAX_FALLBACK_TOP_FILES = 8;
const MAX_MANIFEST_CHARS = 6000;

const NOISE_PREFIXES = [
	".agent/",
	".git/",
	".pi/",
	"node_modules/",
	"dist/",
	"build/",
	"coverage/",
	"target/",
	"docs/superpowers/plans/",
	"docs/plans/",
];

const NOISE_FILE_NAMES = new Set(["commit.log", "prompt.md", "gemini.md"]);
const NOISE_FILE_PREFIXES = ["autoresearch"];
const SOURCE_EXTENSIONS = new Set([
	".ts",
	".tsx",
	".js",
	".jsx",
	".mjs",
	".cjs",
	".py",
	".rb",
	".go",
	".rs",
	".java",
	".kt",
	".kts",
	".scala",
	".swift",
	".c",
	".cc",
	".cpp",
	".h",
	".hpp",
	".cs",
]);

const TRANSCRIPT_SYNTHESIS_SYSTEM_PROMPT = `You synthesize repository skills from a transcript of tool-assisted repository analysis.

Return exactly these sections:

<skill_markdown>
...full SKILL.md content including frontmatter...
</skill_markdown>

<instincts_json>
[
  {
    "id": "kebab-case-id",
    "title": "Short title",
    "trigger": "when ...",
    "confidence": 0.75,
    "domain": "git",
    "scope": "project",
    "action": "One concrete sentence",
    "evidence": ["short bullet", "short bullet"]
  }
]
</instincts_json>

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
  "absorbTarget": "optional existing skill path or MEMORY.md",
  "improvements": ["optional improvement", "optional improvement"]
}
</quality_json>

Rules:
- Use only facts present in the transcript and provided metadata
- Preserve repo specificity and avoid generic best practices
- When generating instincts, prefer atomic, clusterable rules over broad summary instincts
- It is acceptable for multiple instincts to share the same trigger when they represent distinct steps or checks in the same workflow
- Prefer 4-6 concrete instincts if the repository evidence supports them; otherwise return fewer
- If the transcript evidence is insufficient for a claim, omit it`;

interface CommitEntry {
	hash: string;
	subject: string;
	date: string;
	files: string[];
}

interface FileCount {
	path: string;
	count: number;
}

type ExistingSkillSummary = ExistingSkillReference;

interface SkillCreateLlmContext {
	model: Model<any>;
	apiKey: string;
	headers?: Record<string, string>;
	modelRegistry: ModelRegistry;
}

export interface SkillCreateOptions {
	cwd: string;
	project: ProjectInfo;
	layout: StorageLayout;
	commits: number;
	output?: string;
	includeInstincts: boolean;
	llm?: SkillCreateLlmContext;
}

export interface SkillCreateResult {
	skillPath: string;
	instinctPaths: string[];
	summary: string;
	generationMode: "agentic" | "fallback";
	llmStatus: string;
	quality: SkillCreateQualityReport;
	commitCount: number;
	prefixes: string[];
	representativeFiles: string[];
}

interface LlmInstinctDraft {
	id: string;
	title: string;
	trigger: string;
	confidence: number;
	domain: string;
	scope: "project" | "global";
	action: string;
	evidence: string[];
}

async function git(args: string[], cwd: string): Promise<string> {
	const result = await execFileAsync("git", args, {
		cwd,
		maxBuffer: 8 * 1024 * 1024,
	});
	return result.stdout.trim();
}

function isNoisePath(path: string): boolean {
	const name = basename(path).toLowerCase();
	if (NOISE_FILE_NAMES.has(name) || NOISE_FILE_PREFIXES.some((prefix) => name.startsWith(prefix))) {
		return true;
	}
	return NOISE_PREFIXES.some((prefix) => path.startsWith(prefix));
}

function isLikelyDocFile(path: string): boolean {
	if (isNoisePath(path)) {
		return false;
	}
	const lowered = path.toLowerCase();
	return (
		lowered.endsWith(".md") ||
		lowered.endsWith(".mdx") ||
		lowered.endsWith(".adoc") ||
		lowered.startsWith("docs/") ||
		lowered.includes("/docs/") ||
		lowered === "readme" ||
		lowered === "readme.md" ||
		lowered === "changelog.md"
	);
}

function isLikelyTestFile(path: string): boolean {
	if (isNoisePath(path)) {
		return false;
	}
	const lowered = path.toLowerCase();
	return (
		lowered.includes("/test/") ||
		lowered.includes("/tests/") ||
		lowered.includes("__tests__/") ||
		lowered.endsWith(".test.ts") ||
		lowered.endsWith(".test.tsx") ||
		lowered.endsWith(".test.js") ||
		lowered.endsWith(".test.jsx") ||
		lowered.endsWith(".spec.ts") ||
		lowered.endsWith(".spec.tsx") ||
		lowered.endsWith(".spec.js") ||
		lowered.endsWith(".spec.jsx") ||
		lowered.endsWith("_test.go") ||
		lowered.endsWith("_spec.rb") ||
		lowered.endsWith("test.py") ||
		lowered.endsWith("tests.py") ||
		lowered.endsWith("test.rs") ||
		lowered.endsWith("tests.rs")
	);
}

function isLikelyBuildFile(path: string): boolean {
	if (isNoisePath(path)) {
		return false;
	}
	const lowered = path.toLowerCase();
	return [
		"package.json",
		"pnpm-lock.yaml",
		"yarn.lock",
		"package-lock.json",
		"cargo.toml",
		"cargo.lock",
		"pom.xml",
		"build.gradle",
		"build.gradle.kts",
		"settings.gradle",
		"settings.gradle.kts",
		"pyproject.toml",
		"poetry.lock",
		"go.mod",
		"go.sum",
		"makefile",
		"justfile",
	].some((suffix) => lowered === suffix || lowered.endsWith(`/${suffix}`));
}

function isLikelySourceFile(path: string): boolean {
	if (isNoisePath(path) || isLikelyDocFile(path) || isLikelyTestFile(path) || isLikelyBuildFile(path)) {
		return false;
	}
	return SOURCE_EXTENSIONS.has(extname(path).toLowerCase());
}

async function collectGitHistory(repoRoot: string, commits: number): Promise<CommitEntry[]> {
	const output = await git(
		[
			"-C",
			repoRoot,
			"log",
			"--name-only",
			"-n",
			String(commits),
			"--pretty=format:__COMMIT__%n%H|%s|%ad",
			"--date=short",
		],
		repoRoot,
	);
	const lines = output.split("\n");
	const entries: CommitEntry[] = [];
	let current: CommitEntry | null = null;

	for (const rawLine of lines) {
		const line = rawLine.trim();
		if (line === "__COMMIT__") {
			if (current) {
				entries.push(current);
			}
			current = null;
			continue;
		}
		if (!current) {
			if (!line.includes("|")) {
				continue;
			}
			const [hash, subject, date] = line.split("|", 3);
			current = { hash, subject, date, files: [] };
			continue;
		}
		if (line.length > 0 && !isNoisePath(line)) {
			current.files.push(line);
		}
	}

	if (current) {
		entries.push(current);
	}
	return entries;
}

function summarizeCommitPrefixes(entries: CommitEntry[]): Array<{ prefix: string; count: number }> {
	const counts = new Map<string, number>();
	for (const entry of entries) {
		const match = entry.subject.match(/^([a-z]+)(\([^)]+\))?!?:/u);
		if (!match) {
			continue;
		}
		const prefix = match[1];
		counts.set(prefix, (counts.get(prefix) ?? 0) + 1);
	}
	return Array.from(counts.entries())
		.map(([prefix, count]) => ({ prefix, count }))
		.sort((left, right) => right.count - left.count)
		.slice(0, 12);
}

function buildRecentCommitSamples(entries: CommitEntry[], limit: number): string {
	return entries
		.slice(0, Math.max(1, limit))
		.map((entry) => {
			const files = entry.files
				.slice(0, 8)
				.map((file) => `  - ${file}`)
				.join("\n");
			return `${entry.date} ${entry.subject}\n${files || "  - (no files)"}`;
		})
		.join("\n\n");
}

function buildFileFrequencySummary(allFiles: FileCount[], limit: number): string {
	return allFiles
		.slice(0, Math.max(1, limit))
		.map((file) => `${String(file.count).padStart(4, " ")} ${file.path}`)
		.join("\n");
}

function countFiles(entries: CommitEntry[], predicate: (path: string) => boolean): FileCount[] {
	const counts = new Map<string, number>();
	for (const entry of entries) {
		for (const path of entry.files) {
			if (!predicate(path)) {
				continue;
			}
			counts.set(path, (counts.get(path) ?? 0) + 1);
		}
	}
	return Array.from(counts.entries())
		.map(([path, count]) => ({ path, count }))
		.sort((left, right) => right.count - left.count);
}

function buildCandidatePaths(
	entries: CommitEntry[],
	sourceFiles: FileCount[],
	testFiles: FileCount[],
	buildFiles: FileCount[],
): string[] {
	const candidates: string[] = [];
	const seen = new Set<string>();
	const add = (path: string) => {
		if (!path || isNoisePath(path) || seen.has(path)) {
			return;
		}
		seen.add(path);
		candidates.push(path);
	};

	for (const path of [
		"README.md",
		"ARCHITECTURE.md",
		"CONTRIBUTING.md",
		"docs/CONTRIBUTING.md",
		"docs/ARCHITECTURE.md",
		"AGENTS.md",
		"Cargo.toml",
		"package.json",
		"pom.xml",
		"pyproject.toml",
		"go.mod",
	]) {
		add(path);
	}

	for (const file of buildFiles.slice(0, 8)) {
		add(file.path);
	}
	for (const file of sourceFiles.slice(0, 10)) {
		add(file.path);
	}
	for (const file of testFiles.slice(0, 6)) {
		add(file.path);
	}
	for (const entry of entries.slice(0, 12)) {
		for (const file of entry.files.slice(0, 6)) {
			add(file);
		}
	}

	return candidates.slice(0, 24);
}

function extractJsonPayload(text: string): string | null {
	const fenced = text.match(/```json\s*([\s\S]*?)```/u)?.[1];
	if (fenced) {
		return fenced.trim();
	}
	const firstBrace = text.indexOf("{");
	const lastBrace = text.lastIndexOf("}");
	if (firstBrace >= 0 && lastBrace > firstBrace) {
		return text.slice(firstBrace, lastBrace + 1);
	}
	const firstBracket = text.indexOf("[");
	const lastBracket = text.lastIndexOf("]");
	if (firstBracket >= 0 && lastBracket > firstBracket) {
		return text.slice(firstBracket, lastBracket + 1);
	}
	return null;
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

function validateQualityReport(value: unknown): Partial<SkillCreateQualityReport> | undefined {
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
		: record.checklist && typeof record.checklist === "object"
			? Object.entries(record.checklist as Record<string, unknown>)
					.map(([key, item]) =>
						typeof item === "string" && item.trim().length > 0 ? `${key}: ${item.trim()}` : null,
					)
					.filter((item): item is string => Boolean(item))
			: [];
	return {
		verdict,
		rationale: typeof record.rationale === "string" ? record.rationale.trim() : "",
		checklist,
		absorbTarget:
			typeof record.absorbTarget === "string" && record.absorbTarget.trim().length > 0
				? record.absorbTarget.trim()
				: undefined,
		improvements: Array.isArray(record.improvements)
			? record.improvements.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
			: undefined,
	};
}

function normalizeSkillMarkdown(raw: string, project: ProjectInfo): string {
	const { frontmatter, body } = parseFrontmatter<Record<string, unknown>>(raw);
	const lines = body.trim().length > 0 ? body.trim().split("\n") : [];
	const skillName = `${project.name.toLowerCase().replace(/[^a-z0-9-]+/gu, "-")}-patterns`;
	const description =
		typeof frontmatter.description === "string" && frontmatter.description.trim().length > 0
			? frontmatter.description.trim()
			: `Coding patterns extracted from ${project.name}`;

	const frontmatterLines = [
		"---",
		`name: ${typeof frontmatter.name === "string" && frontmatter.name.trim().length > 0 ? frontmatter.name.trim() : skillName}`,
		`description: ${description}`,
		...(typeof frontmatter.version === "string" ? [`version: ${frontmatter.version}`] : ["version: 1.0.0"]),
		...(typeof frontmatter.source === "string" ? [`source: ${frontmatter.source}`] : ["source: local-git-analysis"]),
		"---",
		"",
	];

	if (lines.length === 0 || !lines[0]?.startsWith("# ")) {
		lines.unshift(`# ${project.name} Patterns`, "");
	}

	return [...frontmatterLines, ...lines].join("\n").trimEnd();
}

function validateInstinctDraft(value: unknown): LlmInstinctDraft | null {
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
	const evidence = Array.isArray(record.evidence)
		? record.evidence.filter((item): item is string => typeof item === "string" && item.trim().length > 0).slice(0, 5)
		: [];
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
		evidence,
	};
}

function parseInstinctDrafts(raw: string | undefined): LlmInstinctDraft[] {
	if (!raw) {
		return [];
	}
	const payload = extractJsonPayload(raw) ?? raw;
	try {
		const parsed = JSON.parse(payload) as unknown[];
		if (!Array.isArray(parsed)) {
			return [];
		}
		return parsed.map(validateInstinctDraft).filter((draft): draft is LlmInstinctDraft => Boolean(draft));
	} catch {
		return [];
	}
}

function parseQualityDraft(raw: string | undefined): Partial<SkillCreateQualityReport> | undefined {
	if (!raw) {
		return undefined;
	}
	const payload = extractJsonPayload(raw) ?? raw;
	try {
		return validateQualityReport(JSON.parse(payload));
	} catch {
		return undefined;
	}
}

function parseStructuredLlmResult(
	text: string,
	project: ProjectInfo,
): {
	skillMarkdown?: string;
	instincts: LlmInstinctDraft[];
	quality?: Partial<SkillCreateQualityReport>;
} | null {
	const taggedSkill = extractTaggedSection(text, "skill_markdown");
	if (!taggedSkill) {
		return null;
	}
	return {
		skillMarkdown: normalizeSkillMarkdown(taggedSkill, project),
		instincts: parseInstinctDrafts(extractTaggedSection(text, "instincts_json") ?? undefined),
		quality: parseQualityDraft(extractTaggedSection(text, "quality_json") ?? undefined),
	};
}

async function readOptionalText(filePath: string, maxChars: number): Promise<string> {
	try {
		const content = await readFile(filePath, "utf-8");
		return content.slice(0, maxChars);
	} catch {
		return "";
	}
}

async function loadExistingSkills(projectRoot: string, outputSkillPath: string): Promise<ExistingSkillSummary[]> {
	const skills = loadSkills({ cwd: projectRoot }).skills;
	const summaries: ExistingSkillSummary[] = [];
	for (const skill of skills.filter((item) => item.filePath !== outputSkillPath)) {
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

function summarizeTopAreas(allFiles: FileCount[]): string[] {
	const areas = new Map<string, number>();
	for (const file of allFiles) {
		const segments = file.path.split("/");
		const key =
			segments[0] === "crates" || segments[0] === "packages"
				? segments.slice(0, 2).join("/")
				: segments[0] === ".github"
					? ".github"
					: (segments[0] ?? file.path);
		if (!key) {
			continue;
		}
		areas.set(key, (areas.get(key) ?? 0) + file.count);
	}
	const lines = Array.from(areas.entries())
		.sort((left, right) => right[1] - left[1])
		.slice(0, MAX_FALLBACK_TOP_FILES)
		.map(([area, count]) => `- \`${area}\` 是高频变更区域（${count} 次），通常值得优先理解其职责边界。`);
	return lines.length > 0 ? lines : ["- 暂未提炼出稳定的高频目录区域。"];
}

function summarizeBuildSurface(manifests: Record<string, string>, allFiles: FileCount[]): string[] {
	const lines: string[] = [];
	if (manifests["Cargo.toml"]) {
		lines.push("- 仓库包含 `Cargo.toml`，构建与依赖约束应优先从 Cargo/Rust workspace 结构中确认。");
		if (manifests["Cargo.toml"].includes("[workspace]")) {
			lines.push("- 根 `Cargo.toml` 启用了 workspace，跨 crate 的边界和依赖关系应按 workspace 组织来理解。");
		}
	}
	if (manifests["package.json"]) {
		lines.push("- 仓库包含 `package.json`，前端/Node 工作流应以 package scripts 和 lockfile 为准。");
		if (manifests["package.json"].includes('"workspaces"')) {
			lines.push("- `package.json` 启用了 workspaces，改动时需要考虑多包联动。");
		}
	}
	if (manifests["pom.xml"]) {
		lines.push("- 仓库包含 `pom.xml`，Java 构建、测试和打包步骤应以 Maven 配置为准。");
	}
	if (manifests["pyproject.toml"]) {
		lines.push("- 仓库包含 `pyproject.toml`，Python 构建、依赖和工具链配置以该文件为准。");
	}
	if (manifests["go.mod"]) {
		lines.push("- 仓库包含 `go.mod`，Go 模块边界和依赖版本需遵守模块声明。");
	}
	const buildFiles = allFiles.filter((file) => isLikelyBuildFile(file.path)).slice(0, 5);
	if (buildFiles.length > 0) {
		lines.push(`- 高频构建/配置文件包括：${buildFiles.map((file) => `\`${file.path}\``).join("、")}。`);
	}
	if (lines.length === 0) {
		lines.push("- 构建与配置约定需以仓库中的 manifest、lockfile 和 CI 配置为准。");
	}
	return lines;
}

function summarizeWorkflows(entries: CommitEntry[]): string[] {
	let sourceAndTests = 0;
	let sourceAndDocs = 0;
	let sourceAndBuild = 0;

	for (const entry of entries) {
		const hasSource = entry.files.some((file) => isLikelySourceFile(file));
		const hasTests = entry.files.some((file) => isLikelyTestFile(file));
		const hasDocs = entry.files.some((file) => isLikelyDocFile(file));
		const hasBuild = entry.files.some((file) => isLikelyBuildFile(file));

		if (hasSource && hasTests) {
			sourceAndTests++;
		}
		if (hasSource && hasDocs) {
			sourceAndDocs++;
		}
		if (hasSource && hasBuild) {
			sourceAndBuild++;
		}
	}

	const lines: string[] = [];
	if (sourceAndTests >= 2) {
		lines.push("- 代码改动经常与测试一起提交，新增或重构实现时应同步补测试或更新测试夹具。");
	}
	if (sourceAndDocs >= 2) {
		lines.push("- 影响外部行为或重要设计的改动通常会伴随 README / docs 更新。");
	}
	if (sourceAndBuild >= 2) {
		lines.push("- 变更核心实现时，往往需要同步检查 manifest、脚本或构建配置。");
	}
	return lines.length > 0 ? lines : ["- 暂未检测到足够稳定的提交流程信号。"];
}

function summarizeTestingPatterns(testFiles: FileCount[]): string[] {
	if (testFiles.length === 0) {
		return ["- 暂未从 git 历史中观察到稳定的测试文件模式。"];
	}
	const lines: string[] = [];
	if (testFiles.some((file) => file.path.includes("__tests__/"))) {
		lines.push("- 测试中使用 `__tests__/` 目录组织用例。");
	}
	if (testFiles.some((file) => file.path.endsWith(".test.ts") || file.path.endsWith(".test.js"))) {
		lines.push("- JavaScript / TypeScript 测试常用 `.test.*` 后缀。");
	}
	if (testFiles.some((file) => file.path.endsWith(".spec.ts") || file.path.endsWith(".spec.js"))) {
		lines.push("- 部分测试文件采用 `.spec.*` 命名。");
	}
	if (testFiles.some((file) => file.path.endsWith("_test.go"))) {
		lines.push("- Go 测试遵循 `_test.go` 约定。");
	}
	if (testFiles.some((file) => file.path.includes("/tests/") || file.path.includes("/test/"))) {
		lines.push("- 测试主要位于 `test/` 或 `tests/` 目录。");
	}
	lines.push(
		`- 高频测试文件包括：${testFiles
			.slice(0, 5)
			.map((file) => `\`${file.path}\``)
			.join("、")}。`,
	);
	return lines;
}

function buildFallbackSkillMarkdown(
	project: ProjectInfo,
	prefixes: Array<{ prefix: string; count: number }>,
	allFiles: FileCount[],
	testFiles: FileCount[],
	entries: CommitEntry[],
	manifests: Record<string, string>,
): string {
	const skillName = `${project.name.toLowerCase().replace(/[^a-z0-9-]+/gu, "-")}-patterns`;
	const prefixLines =
		prefixes.length > 0
			? prefixes.map((item) => `- \`${item.prefix}:\` 是高频提交前缀（${item.count} 次）。`)
			: ["- 未检测到稳定的提交前缀约定。"];
	return [
		"---",
		`name: ${skillName}`,
		`description: Coding patterns extracted from ${project.name}`,
		"version: 1.0.0",
		"source: local-git-analysis",
		"---",
		"",
		`# ${project.name} Patterns`,
		"",
		"## Commit Conventions",
		...prefixLines,
		"",
		"## Code Architecture",
		...summarizeTopAreas(allFiles),
		"",
		"## Build And Runtime",
		...summarizeBuildSurface(manifests, allFiles),
		"",
		"## Workflows",
		...summarizeWorkflows(entries),
		"",
		"## Testing Patterns",
		...summarizeTestingPatterns(testFiles),
	].join("\n");
}

function buildFallbackInstincts(
	project: ProjectInfo,
	prefixes: Array<{ prefix: string; count: number }>,
	testFiles: FileCount[],
	buildFiles: FileCount[],
): LlmInstinctDraft[] {
	const repoSlug = project.name.toLowerCase().replace(/[^a-z0-9-]+/gu, "-");
	const drafts: LlmInstinctDraft[] = [];
	if (prefixes.length > 0) {
		drafts.push({
			id: `${repoSlug}-commit-convention`,
			title: "Follow Repository Commit Convention",
			trigger: "when writing a commit message",
			confidence: 0.8,
			domain: "git",
			scope: "project",
			action: "优先沿用仓库里高频出现的提交前缀和命名风格。",
			evidence: prefixes.slice(0, 5).map((item) => `提交前缀 ${item.prefix}: 出现 ${item.count} 次`),
		});
	}
	if (testFiles.length > 0) {
		drafts.push({
			id: `${repoSlug}-test-layout`,
			title: "Follow Repository Test Layout",
			trigger: "when adding or changing tests",
			confidence: 0.72,
			domain: "testing",
			scope: "project",
			action: "测试文件优先沿用仓库现有目录与命名模式，而不是自创新布局。",
			evidence: testFiles.slice(0, 4).map((file) => `高频测试路径 ${file.path}（${file.count} 次）`),
		});
		drafts.push({
			id: `${repoSlug}-test-fixture-reuse`,
			title: "Reuse Existing Test Fixtures",
			trigger: "when adding or changing tests",
			confidence: 0.7,
			domain: "testing",
			scope: "project",
			action: "优先复用仓库已有测试夹具、样例数据和共享测试工具，而不是临时发明一套新夹具。",
			evidence: testFiles.slice(0, 4).map((file) => `测试相关路径 ${file.path}（${file.count} 次）`),
		});
	}
	if (buildFiles.length > 0) {
		drafts.push({
			id: `${repoSlug}-build-manifest-awareness`,
			title: "Respect Workspace Build Manifests",
			trigger: "when changing workspace manifests",
			confidence: 0.68,
			domain: "workflow",
			scope: "project",
			action: "涉及构建和模块边界的改动前，先核对仓库的 manifest、lockfile 和构建脚本。",
			evidence: buildFiles.slice(0, 4).map((file) => `高频构建文件 ${file.path}（${file.count} 次）`),
		});
		drafts.push({
			id: `${repoSlug}-workspace-version-inheritance`,
			title: "Keep Workspace Version Inheritance",
			trigger: "when changing workspace manifests",
			confidence: 0.66,
			domain: "workflow",
			scope: "project",
			action: "改动 workspace 或子模块依赖时，优先延续仓库现有的统一版本继承方式，而不是在局部单独钉版本。",
			evidence: buildFiles.slice(0, 4).map((file) => `相关 manifest ${file.path}（${file.count} 次）`),
		});
	}
	return drafts.slice(0, 6);
}

function isRepoSpecificSkill(project: ProjectInfo, skillMarkdown: string): boolean {
	return normalizeCompareText(skillMarkdown).includes(normalizeCompareText(project.name));
}

function dedupeInstinctDrafts(
	drafts: LlmInstinctDraft[],
	existingInstincts: Array<{ id: string; title: string; trigger: string; domain: string; action?: string }>,
): { drafts: LlmInstinctDraft[]; droppedIds: string[] } {
	const result = dedupeComparableInstincts(drafts, existingInstincts, 6);
	return { drafts: result.kept, droppedIds: result.droppedIds };
}

function buildQualityReport(
	project: ProjectInfo,
	skillMarkdown: string,
	existingSkills: ExistingSkillSummary[],
	droppedInstinctIds: string[],
	projectMemory: string,
	globalMemory: string,
	llmQuality?: Partial<SkillCreateQualityReport>,
): SkillCreateQualityReport {
	const headingCount = (skillMarkdown.match(/^## /gmu) ?? []).length;
	const projectSpecific = isRepoSpecificSkill(project, skillMarkdown);
	const projectToken = normalizeCompareText(project.name);
	const overlapSkills = existingSkills
		.map((skill) => ({
			skill,
			match: detectOverlappingSkills(skillMarkdown, [skill], { limit: 1, threshold: 0.52 })[0],
		}))
		.filter((entry) => {
			if (!entry.match) {
				return false;
			}
			const isProjectLocal = entry.skill.filePath.startsWith(project.root);
			if (!projectSpecific || isProjectLocal) {
				return true;
			}
			const existingIdentity = normalizeCompareText(
				`${entry.skill.name} ${entry.skill.description} ${entry.skill.bodyText}`,
			);
			return existingIdentity.includes(projectToken);
		})
		.sort((left, right) => (right.match?.score ?? 0) - (left.match?.score ?? 0))
		.map((entry) => entry.skill.filePath)
		.slice(0, 3);
	const memoryOverlap =
		(projectMemory.length > 0 && overlapScore(skillMarkdown, projectMemory) >= 0.35) ||
		(globalMemory.length > 0 && overlapScore(skillMarkdown, globalMemory) >= 0.35);

	const checklist = [
		overlapSkills.length === 0 ? "与现有 skills 无明显重叠" : `与 ${overlapSkills.length} 个现有 skill 存在主题重叠`,
		memoryOverlap ? "与 MEMORY.md 存在主题重叠" : "与 MEMORY.md 无明显重叠",
		headingCount >= 3 ? "结构化章节足够" : "章节偏少，结构可能过薄",
		droppedInstinctIds.length === 0
			? "未发现重复 instinct"
			: `已丢弃 ${droppedInstinctIds.length} 个重复/泛化 instinct`,
	];

	let verdict: SkillCreateQualityReport["verdict"] = "save";
	let rationale = "技能内容具备可重用性，且没有发现需要吸收进已有 skill 的强重叠。";
	let absorbTarget: string | undefined = llmQuality?.absorbTarget;
	let improvements: string[] | undefined = llmQuality?.improvements;
	if (llmQuality?.verdict) {
		verdict = llmQuality.verdict;
		rationale = llmQuality.rationale?.trim() || rationale;
	} else if (overlapSkills.length > 0) {
		verdict = "absorb";
		rationale = "检测到与现有 skill 存在主题重叠，后续应考虑吸收或合并。";
		absorbTarget = overlapSkills[0];
	} else if (memoryOverlap) {
		verdict = "absorb";
		rationale = "检测到与 MEMORY.md 存在明显重叠，更适合吸收到现有记忆而非重复落 skill。";
		absorbTarget = "MEMORY.md";
	} else if (headingCount < 3) {
		verdict = "improve-then-save";
		rationale = "技能结构偏薄，但仍具有可保存价值。";
		improvements = ["补充更具体的仓库边界、命令约定和触发条件"];
	}

	return {
		verdict,
		rationale,
		checklist: llmQuality?.checklist && llmQuality.checklist.length > 0 ? llmQuality.checklist : checklist,
		overlapSkills,
		droppedInstinctIds,
		absorbTarget,
		improvements,
	};
}

function stripFrontmatter(raw: string): string {
	return raw.replace(/^---[\s\S]*?---\s*/u, "").trim();
}

function buildAbsorbContent(skillMarkdown: string, absorbTarget: string | undefined): string {
	const body = stripFrontmatter(skillMarkdown);
	const title = absorbTarget ?? "existing skill";
	return [
		`# Suggested Additions For ${title}`,
		"",
		"```diff",
		"@@ append @@",
		...body.split("\n").map((line) => `+ ${line}`),
		"```",
	].join("\n");
}

async function synthesizeFromTranscript(
	project: ProjectInfo,
	llm: SkillCreateLlmContext,
	transcript: string,
	existingSkills: ExistingSkillSummary[],
	existingInstincts: Array<{ id: string; title: string; trigger: string; domain: string }>,
	projectMemory: string,
	globalMemory: string,
	includeInstincts: boolean,
): Promise<{
	skillMarkdown?: string;
	instincts: LlmInstinctDraft[];
	quality?: Partial<SkillCreateQualityReport>;
	status: string;
}> {
	const userMessage: UserMessage = {
		role: "user",
		content: [
			{
				type: "text",
				text: [
					`Repository: ${project.name}`,
					project.remote ? `Remote: ${project.remote}` : "Remote: none",
					`Generate instincts: ${includeInstincts ? "yes" : "no"}`,
					"",
					"[EXISTING SKILLS]",
					existingSkills.length > 0
						? existingSkills
								.map(
									(skill) =>
										`- ${skill.name}: ${skill.description} (${skill.filePath})\n${skill.bodyText.slice(0, 1500)}`,
								)
								.join("\n")
						: "(none)",
					"",
					"[EXISTING INSTINCTS]",
					existingInstincts.length > 0
						? existingInstincts
								.map((instinct) => `- ${instinct.id}: ${instinct.trigger} [${instinct.domain}]`)
								.join("\n")
						: "(none)",
					"",
					"[PROJECT MEMORY]",
					projectMemory || "(missing)",
					"",
					"[GLOBAL MEMORY]",
					globalMemory || "(missing)",
					"",
					"[TRANSCRIPT]",
					transcript,
					"",
					"[TASK]",
					"Use the transcript above as the sole evidence source.",
					"Produce a repository-specific skill, optional project instincts, and a learn-eval style quality report.",
					includeInstincts
						? "You may produce up to 6 instincts. Prefer atomic instincts that can cluster into future skills."
						: "Return an empty instincts_json array.",
				].join("\n"),
			},
		],
		timestamp: Date.now(),
	};

	try {
		const response = await complete(
			llm.model,
			{
				systemPrompt: TRANSCRIPT_SYNTHESIS_SYSTEM_PROMPT,
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
		const parsed = parseStructuredLlmResult(text, project);
		return parsed ? { ...parsed, status: "success" } : { instincts: [], status: "parse-failed" };
	} catch (error) {
		return {
			instincts: [],
			status: `error:${error instanceof Error ? error.message : String(error)}`,
		};
	}
}

async function improveSkillWithLlm(
	project: ProjectInfo,
	llm: SkillCreateLlmContext,
	skillMarkdown: string,
	improvements: string[],
): Promise<{ skillMarkdown: string | null; status: string }> {
	const userMessage: UserMessage = {
		role: "user",
		content: [
			{
				type: "text",
				text: [
					"Revise this repository skill once using the following improvements.",
					"",
					"Return only:",
					"<skill_markdown>",
					"...full revised SKILL.md...",
					"</skill_markdown>",
					"",
					"Improvements:",
					...improvements.map((item) => `- ${item}`),
					"",
					"Current draft:",
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
					"You revise repository skills. Preserve valid frontmatter and improve specificity, actionability, and scope fit. Return only the requested <skill_markdown> block.",
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
		const revisedBlock = extractTaggedSection(text, "skill_markdown");
		return {
			skillMarkdown: revisedBlock ? normalizeSkillMarkdown(revisedBlock, project) : null,
			status: revisedBlock ? "success" : "parse-failed",
		};
	} catch (error) {
		return {
			skillMarkdown: null,
			status: `error:${error instanceof Error ? error.message : String(error)}`,
		};
	}
}

function serializeInstinctDraft(project: ProjectInfo, instinct: LlmInstinctDraft): string {
	const scope = project.id === "global" ? "global" : instinct.scope;
	return serializeInstinct({
		id: instinct.id,
		title: instinct.title,
		trigger: instinct.trigger,
		confidence: instinct.confidence,
		domain: instinct.domain,
		source: "local-repo-analysis",
		scope,
		projectId: scope === "project" ? project.id : undefined,
		projectName: scope === "project" ? project.name : undefined,
		content: [
			`# ${instinct.title}`,
			"",
			"## Action",
			instinct.action,
			"",
			"## Evidence",
			...(instinct.evidence.length > 0
				? instinct.evidence.map((line) => `- ${line}`)
				: ["- Derived from repository analysis"]),
		].join("\n"),
		created: new Date().toISOString(),
		updated: new Date().toISOString(),
	});
}

function resolveSkillOutputPath(project: ProjectInfo, layout: StorageLayout, output?: string): string {
	const skillDirName = `${project.name.toLowerCase().replace(/[^a-z0-9-]+/gu, "-")}-patterns`;
	if (!output) {
		return join(layout.projectEvolvedSkillsDir, skillDirName, "SKILL.md");
	}
	const resolvedOutput = resolve(output);
	if (extname(resolvedOutput).toLowerCase() === ".md") {
		return resolvedOutput;
	}
	return join(resolvedOutput, skillDirName, "SKILL.md");
}

export function resolveInstinctOutputDir(
	layout: StorageLayout,
	project: ProjectInfo,
	instinct: LlmInstinctDraft,
): string {
	if (project.id === "global" || instinct.scope === "global") {
		return layout.globalPersonalDir;
	}
	return layout.projectPersonalDir;
}

export async function createSkillFromRepository(options: SkillCreateOptions): Promise<SkillCreateResult> {
	const repoRoot = await git(["-C", options.project.root, "rev-parse", "--show-toplevel"], options.cwd);
	const entries = await collectGitHistory(repoRoot, options.commits);
	if (entries.length === 0) {
		throw new Error("没有可分析的 git 提交记录");
	}

	const prefixes = summarizeCommitPrefixes(entries);
	const allFiles = countFiles(entries, (path) => !isNoisePath(path));
	const sourceFiles = countFiles(entries, isLikelySourceFile);
	const testFiles = countFiles(entries, isLikelyTestFile);
	const buildFiles = countFiles(entries, isLikelyBuildFile);
	const candidatePaths = buildCandidatePaths(entries, sourceFiles, testFiles, buildFiles);
	const skillPath = resolveSkillOutputPath(options.project, options.layout, options.output);
	const existingSkills = await loadExistingSkills(repoRoot, skillPath);
	const existingInstincts = await loadProjectOnlyInstincts(options.layout);
	const projectMemoryPath =
		(await readOptionalText(join(repoRoot, ".pi", "MEMORY.md"), 1)).length > 0
			? join(repoRoot, ".pi", "MEMORY.md")
			: (await readOptionalText(join(repoRoot, "MEMORY.md"), 1)).length > 0
				? join(repoRoot, "MEMORY.md")
				: undefined;
	const globalMemoryPath =
		(await readOptionalText(join(getAgentDir(), "MEMORY.md"), 1)).length > 0
			? join(getAgentDir(), "MEMORY.md")
			: undefined;
	const projectMemory = projectMemoryPath ? (await readOptionalText(projectMemoryPath, 4000)).trim() : "";
	const globalMemory = globalMemoryPath ? (await readOptionalText(globalMemoryPath, 4000)).trim() : "";

	const manifests: Record<string, string> = {};
	for (const fileName of ["Cargo.toml", "package.json", "pom.xml", "pyproject.toml", "go.mod", "README.md"]) {
		const content = await readOptionalText(join(repoRoot, fileName), MAX_MANIFEST_CHARS);
		if (content.length > 0) {
			manifests[fileName] = content;
		}
	}

	let generationMode: SkillCreateResult["generationMode"] = "fallback";
	let llmStatus = "not-used";
	let agenticSkillMarkdown: string | undefined;
	let agenticInstincts: LlmInstinctDraft[] = [];
	let agenticQuality: Partial<SkillCreateQualityReport> | undefined;

	if (options.llm) {
		const artifacts = await runAgenticSkillCreate({
			repoRoot,
			project: options.project,
			layout: options.layout,
			commits: options.commits,
			includeInstincts: options.includeInstincts,
			model: options.llm.model,
			modelRegistry: options.llm.modelRegistry,
			commitPrefixSummary:
				prefixes.length > 0 ? prefixes.map((item) => `- ${item.prefix}: ${item.count}`).join("\n") : "(none)",
			fileFrequencySummary: buildFileFrequencySummary(allFiles, 25),
			recentCommitSamples: buildRecentCommitSamples(entries, 12),
			candidatePaths,
			existingSkills,
			existingInstincts: existingInstincts.map((instinct) => ({
				id: instinct.id,
				title: instinct.title,
				trigger: instinct.trigger,
				domain: instinct.domain,
			})),
			projectMemoryPath,
			globalMemoryPath,
		});
		llmStatus = `agentic:${artifacts.status}`;
		if (artifacts.skillMarkdown) {
			agenticSkillMarkdown = normalizeSkillMarkdown(artifacts.skillMarkdown, options.project);
			generationMode = "agentic";
		}
		agenticInstincts = parseInstinctDrafts(artifacts.instinctsJson);
		agenticQuality = parseQualityDraft(artifacts.qualityJson);
		if (!agenticSkillMarkdown && artifacts.transcript) {
			const synthesized = await synthesizeFromTranscript(
				options.project,
				options.llm,
				artifacts.transcript,
				existingSkills,
				existingInstincts.map((instinct) => ({
					id: instinct.id,
					title: instinct.title,
					trigger: instinct.trigger,
					domain: instinct.domain,
				})),
				projectMemory,
				globalMemory,
				options.includeInstincts,
			);
			llmStatus = `${llmStatus}; synth:${synthesized.status}`;
			if (synthesized.skillMarkdown) {
				agenticSkillMarkdown = synthesized.skillMarkdown;
				generationMode = "agentic";
			}
			if (synthesized.instincts.length > 0) {
				agenticInstincts = synthesized.instincts;
			}
			if (synthesized.quality) {
				agenticQuality = synthesized.quality;
			}
		}
	}

	const skillMarkdown =
		agenticSkillMarkdown ??
		buildFallbackSkillMarkdown(options.project, prefixes, allFiles, testFiles, entries, manifests);

	const instinctDrafts =
		agenticInstincts.length > 0
			? agenticInstincts
			: buildFallbackInstincts(options.project, prefixes, testFiles, buildFiles);

	const dedupedInstincts = dedupeInstinctDrafts(
		instinctDrafts,
		existingInstincts.map((instinct) => ({
			id: instinct.id,
			title: instinct.title,
			trigger: instinct.trigger,
			domain: instinct.domain,
			action: extractInstinctAction(instinct.content),
		})),
	);

	const quality = buildQualityReport(
		options.project,
		skillMarkdown,
		existingSkills,
		dedupedInstincts.droppedIds,
		projectMemory,
		globalMemory,
		agenticQuality,
	);

	let finalSkillMarkdown = skillMarkdown;
	let finalQuality = quality;
	let finalLlmStatus = llmStatus;

	if (
		quality.verdict === "improve-then-save" &&
		options.llm &&
		quality.improvements &&
		quality.improvements.length > 0
	) {
		const improved = await improveSkillWithLlm(options.project, options.llm, skillMarkdown, quality.improvements);
		if (improved.skillMarkdown) {
			finalSkillMarkdown = improved.skillMarkdown;
			finalLlmStatus = `${llmStatus}; revise:${improved.status}`;
			finalQuality = {
				...buildQualityReport(
					options.project,
					finalSkillMarkdown,
					existingSkills,
					dedupedInstincts.droppedIds,
					projectMemory,
					globalMemory,
					agenticQuality,
				),
				revised: true,
			};
		} else {
			finalLlmStatus = `${llmStatus}; revise:${improved.status}`;
		}
	}

	const representativeFiles = candidatePaths.slice(0, 6);

	if (finalQuality.verdict === "drop") {
		return {
			skillPath,
			instinctPaths: [],
			summary: [
				`分析仓库: ${options.project.name}`,
				`提交数: ${entries.length}`,
				`生成方式: ${generationMode}`,
				`LLM状态: ${finalLlmStatus}`,
				`质量判定: ${finalQuality.verdict}`,
				`原因: ${finalQuality.rationale}`,
			].join("\n"),
			generationMode,
			llmStatus: finalLlmStatus,
			quality: finalQuality,
			commitCount: entries.length,
			prefixes: prefixes.map((item) => `${item.prefix}(${item.count})`),
			representativeFiles,
		};
	}

	if (finalQuality.verdict === "absorb") {
		return {
			skillPath,
			instinctPaths: [],
			summary: [
				`分析仓库: ${options.project.name}`,
				`提交数: ${entries.length}`,
				`生成方式: ${generationMode}`,
				`LLM状态: ${finalLlmStatus}`,
				`质量判定: ${finalQuality.verdict}`,
				`吸收目标: ${finalQuality.absorbTarget ?? "existing skill"}`,
				`原因: ${finalQuality.rationale}`,
			].join("\n"),
			generationMode,
			llmStatus: finalLlmStatus,
			quality: {
				...finalQuality,
				absorbContent: buildAbsorbContent(finalSkillMarkdown, finalQuality.absorbTarget),
			},
			commitCount: entries.length,
			prefixes: prefixes.map((item) => `${item.prefix}(${item.count})`),
			representativeFiles,
		};
	}

	await writeTextFile(skillPath, finalSkillMarkdown);

	const instinctPaths: string[] = [];
	if (options.includeInstincts) {
		for (const instinct of dedupedInstincts.drafts) {
			const filePath = join(
				resolveInstinctOutputDir(options.layout, options.project, instinct),
				`${instinct.id}.md`,
			);
			await writeTextFile(filePath, serializeInstinctDraft(options.project, instinct));
			instinctPaths.push(filePath);
		}
	}

	const summaryLines = [
		`分析仓库: ${options.project.name}`,
		`提交数: ${entries.length}`,
		`技能文件: ${skillPath}`,
		`生成方式: ${generationMode}`,
		`LLM状态: ${finalLlmStatus}`,
		`质量判定: ${finalQuality.verdict}`,
	];
	if (prefixes.length > 0) {
		summaryLines.push(`提交前缀: ${prefixes.map((item) => `${item.prefix}(${item.count})`).join(", ")}`);
	}
	if (representativeFiles.length > 0) {
		summaryLines.push(`代表性路径: ${representativeFiles.join(", ")}`);
	}
	if (instinctPaths.length > 0) {
		summaryLines.push(`生成 instinct: ${instinctPaths.length}`);
	}
	if (finalQuality.droppedInstinctIds.length > 0) {
		summaryLines.push(`已丢弃 instinct: ${finalQuality.droppedInstinctIds.join(", ")}`);
	}
	if (finalQuality.revised) {
		summaryLines.push("已执行一次 improve-then-save 自动修订");
	}

	return {
		skillPath,
		instinctPaths,
		summary: summaryLines.join("\n"),
		generationMode,
		llmStatus: finalLlmStatus,
		quality: finalQuality,
		commitCount: entries.length,
		prefixes: prefixes.map((item) => `${item.prefix}(${item.count})`),
		representativeFiles,
	};
}
