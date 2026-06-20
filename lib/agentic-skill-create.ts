import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { promisify } from "node:util";
import type { Model } from "@earendil-works/pi-ai";
import {
	convertToLlm,
	createAgentSession,
	DefaultResourceLoader,
	getAgentDir,
	type ModelRegistry,
	SessionManager,
	serializeConversation,
	type ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import type { ProjectInfo, StorageLayout } from "./types.js";

const execFileAsync = promisify(execFile);

export interface AgenticSkillCreateInput {
	repoRoot: string;
	project: ProjectInfo;
	layout: StorageLayout;
	commits: number;
	includeInstincts: boolean;
	model: Model<any>;
	modelRegistry: ModelRegistry;
	commitPrefixSummary: string;
	fileFrequencySummary: string;
	recentCommitSamples: string;
	candidatePaths: string[];
	existingSkills: Array<{ name: string; description: string; filePath: string }>;
	existingInstincts: Array<{ id: string; title: string; trigger: string; domain: string }>;
	projectMemoryPath?: string;
	globalMemoryPath?: string;
}

export interface AgenticSkillCreateArtifacts {
	skillMarkdown?: string;
	instinctsJson?: string;
	qualityJson?: string;
	transcript?: string;
	status: string;
}

interface CommitEntry {
	hash: string;
	subject: string;
	date: string;
	files: string[];
}

async function git(args: string[], cwd: string): Promise<string> {
	const result = await execFileAsync("git", args, {
		cwd,
		maxBuffer: 8 * 1024 * 1024,
	});
	return result.stdout.trim();
}

async function collectGitHistory(repoRoot: string, commits: number, offset: number = 0): Promise<CommitEntry[]> {
	const output = await git(
		[
			"-C",
			repoRoot,
			"log",
			"--name-only",
			"--skip",
			String(Math.max(0, offset)),
			"-n",
			String(Math.max(1, commits)),
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
			current = {
				hash,
				subject,
				date,
				files: [],
			};
			continue;
		}
		if (line.length > 0) {
			current.files.push(line);
		}
	}

	if (current) {
		entries.push(current);
	}
	return entries;
}

function summarizeFileFrequency(entries: CommitEntry[], limit: number): string {
	const counts = new Map<string, number>();
	for (const entry of entries) {
		for (const file of entry.files) {
			counts.set(file, (counts.get(file) ?? 0) + 1);
		}
	}
	return Array.from(counts.entries())
		.sort((left, right) => right[1] - left[1])
		.slice(0, Math.max(1, limit))
		.map(([path, count]) => `${String(count).padStart(4, " ")} ${path}`)
		.join("\n");
}

function summarizeMessagePatterns(entries: CommitEntry[], limit: number): string {
	const prefixes = new Map<string, number>();
	for (const entry of entries) {
		const match = entry.subject.match(/^([a-z]+)(\([^)]+\))?!?:/u);
		if (match) {
			const prefix = match[1];
			prefixes.set(prefix, (prefixes.get(prefix) ?? 0) + 1);
		}
	}
	const prefixLines = Array.from(prefixes.entries())
		.sort((left, right) => right[1] - left[1])
		.slice(0, 12)
		.map(([prefix, count]) => `- ${prefix}: ${count}`);
	const subjectLines = entries.slice(0, Math.max(1, limit)).map((entry) => `- ${entry.subject}`);
	return [
		"[prefix-summary]",
		prefixLines.length > 0 ? prefixLines.join("\n") : "(none)",
		"",
		"[recent-subjects]",
		subjectLines.length > 0 ? subjectLines.join("\n") : "(none)",
	].join("\n");
}

function formatRecentCommits(entries: CommitEntry[]): string {
	return entries
		.map((entry) => {
			const files = entry.files.length > 0 ? entry.files.map((file) => `  - ${file}`).join("\n") : "  - (no files)";
			return `${entry.date} ${entry.hash.slice(0, 12)} ${entry.subject}\n${files}`;
		})
		.join("\n\n");
}

function formatCandidatePaths(paths: string[]): string {
	return paths.length > 0 ? paths.map((path) => `- ${path}`).join("\n") : "(none)";
}

function resolveRepoPath(repoRoot: string, path: string): string {
	const candidate = isAbsolute(path) ? resolve(path) : resolve(repoRoot, path);
	const rel = relative(repoRoot, candidate);
	if (rel.startsWith("..") || rel === "") {
		if (candidate !== resolve(repoRoot)) {
			throw new Error(`Path escapes repository root: ${path}`);
		}
	}
	return candidate;
}

async function readRepoFile(repoRoot: string, path: string, maxChars: number): Promise<string> {
	const resolved = resolveRepoPath(repoRoot, path);
	const content = await readFile(resolved, "utf-8");
	return content.slice(0, maxChars);
}

function buildAgenticPrompt(input: AgenticSkillCreateInput, artifactPaths: Record<string, string>): string {
	const lines = [
		"You are executing the ECC /skill-create workflow inside pi.",
		"",
		"Operate like the original ECC command:",
		"- analyze git history first",
		"- inspect real repository files to confirm patterns",
		"- produce a repo-specific SKILL.md draft",
		"- run a learn-eval style quality gate before deciding save/absorb/drop",
		"",
		"Environment:",
		`- repository: ${input.project.name}`,
		`- repo root: ${input.repoRoot}`,
		`- project id: ${input.project.id}`,
		`- analyze commits window: ${input.commits}`,
		`- generate instincts: ${input.includeInstincts ? "yes" : "no"}`,
		"",
		"Precomputed ECC git baseline (use this first; only call git_* tools for narrow follow-ups):",
		"[commit-prefix-summary]",
		input.commitPrefixSummary || "(none)",
		"",
		"[file-frequency-summary]",
		input.fileFrequencySummary || "(none)",
		"",
		"[recent-commit-samples]",
		input.recentCommitSamples || "(none)",
		"",
		"[candidate-files]",
		formatCandidatePaths(input.candidatePaths),
		"",
		"Available analysis tools:",
		"- git_recent_commits(limit, offset): inspect commit history with changed files",
		"- git_file_frequency(commits, limit): file change frequency over the requested history window",
		"- git_message_patterns(commits, limit): commit prefix and subject patterns over the requested history window",
		"- list_candidate_files(): list the preselected high-signal files for inspection",
		"- read_repo_file(path): read one candidate file with output truncated for speed",
		"- grep_candidate_files(pattern, limit): search within the candidate file set only",
		"- save_skill_create_artifact(name, content): persist required artifacts",
		"",
		"Required workflow:",
		"1. Start from the provided git baseline and identify a small set of high-signal files or directories worth verifying.",
		"2. Use list_candidate_files, read_repo_file, and grep_candidate_files to inspect only the files needed to confirm architecture, build, workflow, and testing patterns.",
		"3. Use the git_* tools only when the provided baseline is insufficient or you need a narrower follow-up slice.",
		"4. Check overlap against existing skills and MEMORY paths before finalizing.",
		"5. Save the artifacts with save_skill_create_artifact. Do not inline the final SKILL.md in the assistant response.",
		"",
		"Existing skills to review for overlap:",
		input.existingSkills.length > 0
			? input.existingSkills.map((skill) => `- ${skill.name}: ${skill.description} (${skill.filePath})`).join("\n")
			: "(none)",
		"",
		"Existing instincts to avoid duplicating:",
		input.existingInstincts.length > 0
			? input.existingInstincts
					.map((instinct) => `- ${instinct.id}: ${instinct.trigger} [${instinct.domain}]`)
					.join("\n")
			: "(none)",
		"",
		"Relevant MEMORY paths:",
		`- project memory: ${input.projectMemoryPath ?? "(missing)"}`,
		`- global memory: ${input.globalMemoryPath ?? "(missing)"}`,
		"",
		"Artifact requirements:",
		`- save ${artifactPaths["skill.md"]} as artifact name "skill.md"`,
		`- save ${artifactPaths["quality.json"]} as artifact name "quality.json"`,
		input.includeInstincts
			? `- if high-signal instincts exist, save ${artifactPaths["instincts.json"]} as artifact name "instincts.json"`
			: "- do not save instincts.json when instincts are disabled",
		"",
		"skill.md requirements:",
		"- valid SKILL.md markdown with YAML frontmatter including name and description",
		"- repo-specific guidance only",
		"- include practical sections for commit conventions, architecture, workflows, testing, and safety where evidence exists",
		"- do not include meta planning noise or temporary scratch workflow notes unless they are true repo conventions",
		"",
		"quality.json requirements:",
		`- save strict JSON only with keys verdict, rationale, checklist, absorbTarget?, improvements?`,
		'- verdict must be one of "save", "improve-then-save", "absorb", "drop"',
		"- checklist must explicitly cover existing skills overlap, MEMORY overlap, append-vs-new-file, and reusability",
		"",
		"instincts.json requirements:",
		"- save strict JSON array only",
		"- at most 3 instincts",
		"- use objects with id, title, trigger, confidence, domain, scope, action, evidence",
		"- prefer fewer instincts over weak generic instincts",
		"",
		"Important rules:",
		"- verify claims by reading files before writing artifacts",
		"- do not scan the whole repository once you have enough evidence",
		"- prefer candidate files and files directly referenced by the git baseline over broad exploration",
		"- do not modify repository files",
		"- only save final artifacts through save_skill_create_artifact",
		"- if verdict is absorb or drop, still save skill.md draft and quality.json",
		"- if evidence is weak, omit the instinct instead of inventing one",
	];
	return lines.join("\n");
}

async function writeTranscript(
	artifactDir: string,
	messages: Parameters<typeof convertToLlm>[0],
): Promise<string | undefined> {
	try {
		const transcript = serializeConversation(convertToLlm(messages));
		await writeFile(join(artifactDir, "transcript.md"), transcript, "utf-8");
		return transcript;
	} catch (error) {
		await writeFile(
			join(artifactDir, "transcript-error.txt"),
			error instanceof Error ? (error.stack ?? error.message) : String(error),
			"utf-8",
		).catch(() => {});
		return undefined;
	}
}

async function runPromptWithTimeout(
	promptRunner: Promise<void>,
	abort: () => Promise<void>,
	timeoutMs: number,
): Promise<void> {
	let timeoutId: NodeJS.Timeout | undefined;
	const timeoutPromise = new Promise<never>((_resolve, reject) => {
		timeoutId = setTimeout(() => {
			void abort().catch(() => {});
			reject(new Error(`agentic skill-create timed out after ${timeoutMs}ms`));
		}, timeoutMs);
	});
	try {
		await Promise.race([promptRunner, timeoutPromise]);
	} finally {
		if (timeoutId) {
			clearTimeout(timeoutId);
		}
	}
}

async function readArtifact(filePath: string): Promise<string | undefined> {
	try {
		const content = await readFile(filePath, "utf-8");
		const trimmed = content.trim();
		return trimmed.length > 0 ? trimmed : undefined;
	} catch {
		return undefined;
	}
}

export async function runAgenticSkillCreate(input: AgenticSkillCreateInput): Promise<AgenticSkillCreateArtifacts> {
	const artifactsBaseDir = join(input.layout.projectStateDir, "skill-create-artifacts");
	await mkdir(artifactsBaseDir, { recursive: true });
	const artifactDir = await mkdtemp(join(artifactsBaseDir, "run-"));
	const savedArtifacts = new Set<string>();
	let abortedAfterArtifacts = false;
	const candidatePaths = input.candidatePaths.filter((path) => path.trim().length > 0);
	const candidatePathSet = new Set(candidatePaths.map((path) => resolveRepoPath(input.repoRoot, path)));
	const artifactPaths = {
		"skill.md": join(artifactDir, "skill.md"),
		"instincts.json": join(artifactDir, "instincts.json"),
		"quality.json": join(artifactDir, "quality.json"),
	};

	const gitRecentCommitsTool: ToolDefinition = {
		name: "git_recent_commits",
		label: "Git Recent Commits",
		description: "Inspect recent git commits with changed files. Use offset to inspect older commit windows.",
		promptSnippet: "git_recent_commits(limit, offset): inspect raw commit windows with changed files",
		promptGuidelines: [
			"Use git_recent_commits for concrete commit-level evidence before writing architecture or workflow claims.",
		],
		parameters: Type.Object({
			limit: Type.Integer({ minimum: 1, maximum: 200 }),
			offset: Type.Optional(Type.Integer({ minimum: 0, maximum: 5000 })),
		}),
		execute: async (_toolCallId, params) => {
			const args = params as { limit: number; offset?: number };
			const entries = await collectGitHistory(input.repoRoot, args.limit, args.offset ?? 0);
			return {
				content: [{ type: "text", text: formatRecentCommits(entries) || "(no commits found)" }],
				details: { count: entries.length },
			};
		},
	};

	const gitFileFrequencyTool: ToolDefinition = {
		name: "git_file_frequency",
		label: "Git File Frequency",
		description: "Summarize file change frequency over a git history window.",
		promptSnippet: "git_file_frequency(commits, limit): summarize the most frequently changed files in git history",
		promptGuidelines: [
			"Start ECC-style analysis with git_file_frequency to locate high-signal areas before reading files.",
		],
		parameters: Type.Object({
			commits: Type.Integer({ minimum: 1, maximum: 2000 }),
			limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 100 })),
		}),
		execute: async (_toolCallId, params) => {
			const args = params as { commits: number; limit?: number };
			const entries = await collectGitHistory(input.repoRoot, args.commits);
			return {
				content: [{ type: "text", text: summarizeFileFrequency(entries, args.limit ?? 25) || "(none)" }],
				details: { count: entries.length },
			};
		},
	};

	const gitMessagePatternsTool: ToolDefinition = {
		name: "git_message_patterns",
		label: "Git Message Patterns",
		description: "Summarize commit prefixes and recent commit subjects over a git history window.",
		promptSnippet: "git_message_patterns(commits, limit): summarize commit prefix and subject patterns",
		promptGuidelines: ["Use git_message_patterns to confirm commit conventions before drafting the skill."],
		parameters: Type.Object({
			commits: Type.Integer({ minimum: 1, maximum: 2000 }),
			limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 100 })),
		}),
		execute: async (_toolCallId, params) => {
			const args = params as { commits: number; limit?: number };
			const entries = await collectGitHistory(input.repoRoot, args.commits);
			return {
				content: [{ type: "text", text: summarizeMessagePatterns(entries, args.limit ?? 40) }],
				details: { count: entries.length },
			};
		},
	};

	const listCandidateFilesTool: ToolDefinition = {
		name: "list_candidate_files",
		label: "List Candidate Files",
		description: "List the preselected high-signal repository files chosen from git history and repo manifests.",
		promptSnippet: "list_candidate_files(): list the preselected high-signal files you should inspect first",
		promptGuidelines: [
			"Use list_candidate_files early and stay within that set unless a git_* follow-up explicitly justifies another file.",
		],
		parameters: Type.Object({}),
		execute: async () => ({
			content: [{ type: "text", text: formatCandidatePaths(candidatePaths) }],
			details: { count: candidatePaths.length },
		}),
	};

	const readRepoFileTool: ToolDefinition = {
		name: "read_repo_file",
		label: "Read Repo File",
		description: "Read one candidate repository file with output truncated for speed.",
		promptSnippet: "read_repo_file(path): read one candidate repo file with truncated output",
		promptGuidelines: ["Prefer read_repo_file over broad reads. Stay within candidate files whenever possible."],
		parameters: Type.Object({
			path: Type.String({ minLength: 1 }),
		}),
		execute: async (_toolCallId, params) => {
			const args = params as { path: string };
			const resolved = resolveRepoPath(input.repoRoot, args.path);
			if (!candidatePathSet.has(resolved)) {
				return {
					content: [
						{
							type: "text",
							text: `Path not in candidate file set. Use list_candidate_files first.\nAllowed candidates:\n${formatCandidatePaths(candidatePaths)}`,
						},
					],
					details: { allowed: candidatePaths.length },
				};
			}
			return {
				content: [{ type: "text", text: await readRepoFile(input.repoRoot, args.path, 12_000) }],
				details: { path: args.path },
			};
		},
	};

	const grepCandidateFilesTool: ToolDefinition = {
		name: "grep_candidate_files",
		label: "Grep Candidate Files",
		description: "Search within the candidate file set only, with result limits for speed.",
		promptSnippet: "grep_candidate_files(pattern, limit): search only inside the candidate file set",
		promptGuidelines: ["Use grep_candidate_files for quick confirmation instead of scanning the whole repo."],
		parameters: Type.Object({
			pattern: Type.String({ minLength: 1 }),
			limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 80 })),
		}),
		execute: async (_toolCallId, params) => {
			const args = params as { pattern: string; limit?: number };
			if (candidatePaths.length === 0) {
				return {
					content: [{ type: "text", text: "No candidate files available." }],
					details: {},
				};
			}
			try {
				const result = await execFileAsync(
					"rg",
					[
						"-n",
						"--max-count",
						String(args.limit ?? 40),
						args.pattern,
						...candidatePaths.map((path) => resolveRepoPath(input.repoRoot, path)),
					],
					{
						cwd: input.repoRoot,
						maxBuffer: 2 * 1024 * 1024,
					},
				);
				return {
					content: [{ type: "text", text: result.stdout.trim() || "No matches found" }],
					details: {},
				};
			} catch (error) {
				const message =
					error && typeof error === "object" && "stdout" in error
						? String((error as { stdout?: string }).stdout || "").trim() || "No matches found"
						: `grep failed: ${error instanceof Error ? error.message : String(error)}`;
				return {
					content: [{ type: "text", text: message }],
					details: {},
				};
			}
		},
	};

	const saveArtifactTool: ToolDefinition = {
		name: "save_skill_create_artifact",
		label: "Save Skill Artifact",
		description:
			"Persist one final /skill-create artifact. Only use this for skill.md, quality.json, and instincts.json in the private artifact directory.",
		promptSnippet:
			"save_skill_create_artifact(name, content): save skill.md, quality.json, or instincts.json instead of replying with the final draft inline",
		promptGuidelines: [
			"Before finishing, save skill.md and quality.json with save_skill_create_artifact. Save instincts.json only when justified.",
		],
		parameters: Type.Object({
			name: Type.Union([Type.Literal("skill.md"), Type.Literal("quality.json"), Type.Literal("instincts.json")]),
			content: Type.String({ minLength: 1 }),
		}),
		execute: async (_toolCallId, params, _signal, _onUpdate, ctx) => {
			const args = params as { name: "skill.md" | "quality.json" | "instincts.json"; content: string };
			const filePath = artifactPaths[args.name];
			await mkdir(dirname(filePath), { recursive: true });
			await writeFile(filePath, `${args.content.trim()}\n`, "utf-8");
			savedArtifacts.add(args.name);
			if (savedArtifacts.has("skill.md") && savedArtifacts.has("quality.json")) {
				abortedAfterArtifacts = true;
				ctx.abort();
			}
			return {
				content: [{ type: "text", text: `Saved ${args.name}` }],
				details: { filePath },
			};
		},
	};

	const resourceLoader = new DefaultResourceLoader({
		cwd: input.repoRoot,
		agentDir: getAgentDir(),
		noExtensions: true,
		noSkills: true,
		noPromptTemplates: true,
		noThemes: true,
		agentsFilesOverride: () => ({ agentsFiles: [] }),
		systemPromptOverride: () => undefined,
		appendSystemPromptOverride: () => [],
	});
	await resourceLoader.reload();

	const { session } = await createAgentSession({
		cwd: input.repoRoot,
		model: input.model,
		thinkingLevel: "low",
		modelRegistry: input.modelRegistry,
		resourceLoader,
		sessionManager: SessionManager.inMemory(),
		tools: [],
		customTools: [
			gitRecentCommitsTool,
			gitFileFrequencyTool,
			gitMessagePatternsTool,
			listCandidateFilesTool,
			readRepoFileTool,
			grepCandidateFilesTool,
			saveArtifactTool,
		],
	});

	try {
		await runPromptWithTimeout(
			session.prompt(buildAgenticPrompt(input, artifactPaths)),
			() => session.abort(),
			120_000,
		);
		const transcript = await writeTranscript(artifactDir, session.messages);
		const skillMarkdown = await readArtifact(artifactPaths["skill.md"]);
		const instinctsJson = await readArtifact(artifactPaths["instincts.json"]);
		const qualityJson = await readArtifact(artifactPaths["quality.json"]);
		if (!skillMarkdown) {
			return {
				status: "missing-skill-artifact",
				instinctsJson,
				qualityJson,
				transcript,
			};
		}
		return {
			skillMarkdown,
			instinctsJson,
			qualityJson,
			transcript,
			status: abortedAfterArtifacts ? "saved-and-aborted" : "success",
		};
	} catch (error) {
		const transcript = await writeTranscript(artifactDir, session.messages);
		const skillMarkdown = await readArtifact(artifactPaths["skill.md"]);
		const instinctsJson = await readArtifact(artifactPaths["instincts.json"]);
		const qualityJson = await readArtifact(artifactPaths["quality.json"]);
		if (skillMarkdown) {
			return {
				status: abortedAfterArtifacts
					? "saved-and-aborted"
					: `error:${error instanceof Error ? error.message : String(error)}`,
				skillMarkdown,
				instinctsJson,
				qualityJson,
				transcript,
			};
		}
		return {
			status: `error:${error instanceof Error ? error.message : String(error)}`,
			instinctsJson,
			qualityJson,
			transcript,
		};
	} finally {
		session.dispose();
	}
}
