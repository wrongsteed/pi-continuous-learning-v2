import { cp, mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { getAgentDir, withFileMutationQueue } from "@earendil-works/pi-coding-agent";
import type { ContinuousLearningConfig, ObservationEntry, ObserverState, ProjectInfo, StorageLayout } from "./types.js";
import { DEFAULT_CONFIG } from "./types.js";

interface ProjectRegistryEntry {
	id: string;
	name: string;
	root: string;
	remote?: string;
	createdAt: string;
	lastSeen: string;
}

function withTrailingNewline(text: string): string {
	return text.endsWith("\n") ? text : `${text}\n`;
}

async function readJsonFile<T>(filePath: string, fallback: T): Promise<T> {
	try {
		const content = await readFile(filePath, "utf-8");
		return JSON.parse(content) as T;
	} catch {
		return fallback;
	}
}

async function writeJsonFile(filePath: string, data: unknown): Promise<void> {
	await mkdir(dirname(filePath), { recursive: true });
	await withFileMutationQueue(filePath, async () => {
		await writeFile(filePath, withTrailingNewline(JSON.stringify(data, null, 2)), "utf-8");
	});
}

function normalizeConfig(config: Partial<ContinuousLearningConfig> | undefined): ContinuousLearningConfig {
	return {
		version: config?.version ?? DEFAULT_CONFIG.version,
		observer: {
			enabled: config?.observer?.enabled ?? DEFAULT_CONFIG.observer.enabled,
			runIntervalMinutes: config?.observer?.runIntervalMinutes ?? DEFAULT_CONFIG.observer.runIntervalMinutes,
			minObservationsToAnalyze:
				config?.observer?.minObservationsToAnalyze ?? DEFAULT_CONFIG.observer.minObservationsToAnalyze,
			maxRecentObservations:
				config?.observer?.maxRecentObservations ?? DEFAULT_CONFIG.observer.maxRecentObservations,
			model: config?.observer?.model,
		},
	};
}

export function getStorageLayout(project: ProjectInfo): StorageLayout {
	const rootDir = join(getAgentDir(), "continuous-learning-v2");
	const isGlobal = project.id === "global";
	const projectStateDir = isGlobal ? rootDir : join(project.root, ".pi", "continuous-learning-v2");
	const projectDir = projectStateDir;
	return {
		isGlobalProject: isGlobal,
		rootDir,
		configPath: join(rootDir, "config.json"),
		registryPath: join(rootDir, "projects.json"),
		globalPersonalDir: join(rootDir, "instincts", "global", "personal"),
		globalInheritedDir: join(rootDir, "instincts", "global", "inherited"),
		globalPendingDir: join(rootDir, "instincts", "global", "pending"),
		globalEvolvedSkillsDir: join(rootDir, "evolved", "skills"),
		globalEvolvedPromptsDir: join(rootDir, "evolved", "prompts"),
		globalEvolvedAgentsDir: join(rootDir, "evolved", "agents"),
		projectStateDir,
		projectDir,
		projectMetadataPath: isGlobal ? join(rootDir, "global-project.json") : join(projectDir, "project.json"),
		projectPersonalDir: isGlobal
			? join(rootDir, "instincts", "global", "personal")
			: join(projectDir, "instincts", "personal"),
		projectInheritedDir: isGlobal
			? join(rootDir, "instincts", "global", "inherited")
			: join(projectDir, "instincts", "inherited"),
		projectPendingDir: isGlobal
			? join(rootDir, "instincts", "global", "pending")
			: join(projectDir, "instincts", "pending"),
		projectEvolvedSkillsDir: isGlobal ? join(rootDir, "evolved", "skills") : join(project.root, ".pi", "skills"),
		projectEvolvedPromptsDir: isGlobal ? join(rootDir, "evolved", "prompts") : join(project.root, ".pi", "prompts"),
		projectEvolvedAgentsDir: isGlobal ? join(rootDir, "evolved", "agents") : join(project.root, ".pi", "agents"),
		observationsPath: isGlobal ? join(rootDir, "observations.jsonl") : join(projectDir, "observations.jsonl"),
		observerStatePath: isGlobal ? join(rootDir, "observer-state.json") : join(projectDir, "observer-state.json"),
	};
}

export async function ensureStorage(project: ProjectInfo, layout: StorageLayout): Promise<void> {
	const isGlobal = project.id === "global";
	await mkdir(layout.rootDir, { recursive: true });
	await mkdir(layout.globalPersonalDir, { recursive: true });
	await mkdir(layout.globalInheritedDir, { recursive: true });
	await mkdir(layout.globalPendingDir, { recursive: true });
	await mkdir(layout.globalEvolvedSkillsDir, { recursive: true });
	await mkdir(layout.globalEvolvedPromptsDir, { recursive: true });
	await mkdir(layout.globalEvolvedAgentsDir, { recursive: true });
	await mkdir(layout.projectStateDir, { recursive: true });
	await mkdir(layout.projectPersonalDir, { recursive: true });
	await mkdir(layout.projectInheritedDir, { recursive: true });
	await mkdir(layout.projectPendingDir, { recursive: true });
	await mkdir(layout.projectEvolvedSkillsDir, { recursive: true });
	await mkdir(layout.projectEvolvedPromptsDir, { recursive: true });
	await mkdir(layout.projectEvolvedAgentsDir, { recursive: true });
	await migrateLegacyProjectStorage(project, layout);

	const config = await readJsonFile<Partial<ContinuousLearningConfig> | undefined>(layout.configPath, DEFAULT_CONFIG);
	await writeJsonFile(layout.configPath, normalizeConfig(config));

	const observerState = await readJsonFile<ObserverState>(layout.observerStatePath, {
		lastAnalyzedIndex: 0,
	});
	await writeJsonFile(layout.observerStatePath, observerState);

	if (!isGlobal) {
		const now = new Date().toISOString();
		const currentProject: ProjectRegistryEntry = {
			id: project.id,
			name: project.name,
			root: project.root,
			remote: project.remote,
			createdAt: now,
			lastSeen: now,
		};

		const registry = await readJsonFile<Record<string, ProjectRegistryEntry>>(layout.registryPath, {});
		const existing = registry[project.id];
		registry[project.id] = {
			...currentProject,
			createdAt: existing?.createdAt ?? currentProject.createdAt,
		};
		await writeJsonFile(layout.registryPath, registry);
		await writeJsonFile(layout.projectMetadataPath, registry[project.id]);
	}
}

export async function loadConfig(layout: StorageLayout): Promise<ContinuousLearningConfig> {
	const config = await readJsonFile<Partial<ContinuousLearningConfig> | undefined>(layout.configPath, DEFAULT_CONFIG);
	const normalized = normalizeConfig(config);
	await writeJsonFile(layout.configPath, normalized);
	return normalized;
}

export async function loadObserverState(layout: StorageLayout): Promise<ObserverState> {
	const state = await readJsonFile<ObserverState>(layout.observerStatePath, {
		lastAnalyzedIndex: 0,
	});
	return {
		lastAnalyzedIndex: Math.max(0, state.lastAnalyzedIndex ?? 0),
		lastAnalyzedAt: state.lastAnalyzedAt,
	};
}

export async function saveObserverState(layout: StorageLayout, state: ObserverState): Promise<void> {
	await writeJsonFile(layout.observerStatePath, state);
}

export async function appendObservation(layout: StorageLayout, observation: ObservationEntry): Promise<void> {
	await mkdir(dirname(layout.observationsPath), { recursive: true });
	await withFileMutationQueue(layout.observationsPath, async () => {
		const current = await readFileSafe(layout.observationsPath);
		const line = `${JSON.stringify(observation)}\n`;
		await writeFile(layout.observationsPath, current + line, "utf-8");
	});
}

async function readFileSafe(filePath: string): Promise<string> {
	try {
		return await readFile(filePath, "utf-8");
	} catch {
		return "";
	}
}

export async function readObservations(layout: StorageLayout): Promise<ObservationEntry[]> {
	const content = await readFileSafe(layout.observationsPath);
	if (!content.trim()) {
		return [];
	}

	const observations: ObservationEntry[] = [];
	for (const line of content.split("\n")) {
		const trimmed = line.trim();
		if (!trimmed) {
			continue;
		}
		try {
			observations.push(JSON.parse(trimmed) as ObservationEntry);
		} catch {}
	}
	return observations;
}

export async function countObservationLines(layout: StorageLayout): Promise<number> {
	const observations = await readObservations(layout);
	return observations.length;
}

export async function archiveProcessedObservations(
	layout: StorageLayout,
	processedCount: number,
): Promise<{ archivedCount: number; remainingCount: number; archivePath?: string }> {
	if (processedCount <= 0) {
		const remaining = await countObservationLines(layout);
		return { archivedCount: 0, remainingCount: remaining };
	}

	await mkdir(dirname(layout.observationsPath), { recursive: true });
	return withFileMutationQueue(layout.observationsPath, async () => {
		const current = await readFileSafe(layout.observationsPath);
		const lines = current
			.split("\n")
			.map((line) => line.trim())
			.filter((line) => line.length > 0);
		if (lines.length === 0) {
			return { archivedCount: 0, remainingCount: 0 };
		}

		const archivedLines = lines.slice(0, processedCount);
		const remainingLines = lines.slice(processedCount);
		if (archivedLines.length === 0) {
			return { archivedCount: 0, remainingCount: remainingLines.length };
		}

		const archiveDir = join(layout.projectStateDir, "observations.archive");
		await mkdir(archiveDir, { recursive: true });
		const suffix = `${new Date().toISOString().replace(/[:.]/gu, "-")}-${Math.random().toString(36).slice(2, 8)}`;
		const archivePath = join(archiveDir, `processed-${suffix}.jsonl`);
		await writeFile(archivePath, withTrailingNewline(archivedLines.join("\n")), "utf-8");
		await writeFile(
			layout.observationsPath,
			remainingLines.length > 0 ? withTrailingNewline(remainingLines.join("\n")) : "",
			"utf-8",
		);
		return {
			archivedCount: archivedLines.length,
			remainingCount: remainingLines.length,
			archivePath,
		};
	});
}

export async function writeTextFile(filePath: string, content: string): Promise<void> {
	await mkdir(dirname(filePath), { recursive: true });
	await withFileMutationQueue(filePath, async () => {
		await writeFile(filePath, withTrailingNewline(content), "utf-8");
	});
}

export async function fileExists(filePath: string): Promise<boolean> {
	try {
		await stat(filePath);
		return true;
	} catch {
		return false;
	}
}

export async function loadProjectRegistry(layout: StorageLayout): Promise<Record<string, ProjectRegistryEntry>> {
	return readJsonFile<Record<string, ProjectRegistryEntry>>(layout.registryPath, {});
}

export interface LoadedProjectRegistryEntry extends ProjectRegistryEntry {}

export async function isDirectory(path: string): Promise<boolean> {
	try {
		return (await stat(path)).isDirectory();
	} catch {
		return false;
	}
}

async function isRegularFile(path: string): Promise<boolean> {
	try {
		return (await stat(path)).isFile();
	} catch {
		return false;
	}
}

async function directoryHasEntries(path: string): Promise<boolean> {
	if (!(await isDirectory(path))) {
		return false;
	}
	const entries = await readdir(path);
	return entries.length > 0;
}

async function copyFileIfMissing(sourcePath: string, targetPath: string): Promise<void> {
	if (!(await isRegularFile(sourcePath)) || (await isRegularFile(targetPath))) {
		return;
	}
	await mkdir(dirname(targetPath), { recursive: true });
	await cp(sourcePath, targetPath);
}

async function copyDirectoryIfTargetEmpty(sourcePath: string, targetPath: string): Promise<void> {
	if (!(await isDirectory(sourcePath)) || (await directoryHasEntries(targetPath))) {
		return;
	}
	await mkdir(dirname(targetPath), { recursive: true });
	await cp(sourcePath, targetPath, { recursive: true });
}

async function migrateLegacyProjectStorage(project: ProjectInfo, layout: StorageLayout): Promise<void> {
	const legacyProjectDir = join(layout.rootDir, "projects", project.id);
	if (!(await isDirectory(legacyProjectDir))) {
		return;
	}

	await copyFileIfMissing(join(legacyProjectDir, "project.json"), layout.projectMetadataPath);
	await copyFileIfMissing(join(legacyProjectDir, "observations.jsonl"), layout.observationsPath);
	await copyFileIfMissing(join(legacyProjectDir, "observer-state.json"), layout.observerStatePath);
	await copyDirectoryIfTargetEmpty(join(legacyProjectDir, "instincts", "personal"), layout.projectPersonalDir);
	await copyDirectoryIfTargetEmpty(join(legacyProjectDir, "instincts", "inherited"), layout.projectInheritedDir);
	await copyDirectoryIfTargetEmpty(join(legacyProjectDir, "instincts", "pending"), layout.projectPendingDir);
	await copyDirectoryIfTargetEmpty(join(legacyProjectDir, "evolved", "skills"), layout.projectEvolvedSkillsDir);
	await copyDirectoryIfTargetEmpty(join(legacyProjectDir, "evolved", "prompts"), layout.projectEvolvedPromptsDir);
	await copyDirectoryIfTargetEmpty(join(legacyProjectDir, "evolved", "agents"), layout.projectEvolvedAgentsDir);
}
