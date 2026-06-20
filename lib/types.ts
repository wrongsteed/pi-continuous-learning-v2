export type InstinctScope = "project" | "global";

export interface ContinuousLearningConfig {
	version: string;
	observer: {
		enabled: boolean;
		runIntervalMinutes: number;
		minObservationsToAnalyze: number;
		maxRecentObservations: number;
		model?: string;
	};
}

export const DEFAULT_CONFIG: ContinuousLearningConfig = {
	version: "2.1",
	observer: {
		enabled: false,
		runIntervalMinutes: 5,
		minObservationsToAnalyze: 20,
		maxRecentObservations: 200,
	},
};

export interface ProjectInfo {
	id: string;
	name: string;
	root: string;
	remote?: string;
}

export interface StorageLayout {
	isGlobalProject: boolean;
	rootDir: string;
	configPath: string;
	registryPath: string;
	globalPersonalDir: string;
	globalInheritedDir: string;
	globalPendingDir: string;
	globalEvolvedSkillsDir: string;
	globalEvolvedPromptsDir: string;
	globalEvolvedAgentsDir: string;
	projectStateDir: string;
	projectDir: string;
	projectMetadataPath: string;
	projectPersonalDir: string;
	projectInheritedDir: string;
	projectPendingDir: string;
	projectEvolvedSkillsDir: string;
	projectEvolvedPromptsDir: string;
	projectEvolvedAgentsDir: string;
	observationsPath: string;
	observerStatePath: string;
}

export interface ObservationEntry {
	timestamp: string;
	event: "input" | "tool_call" | "tool_result" | "turn_end";
	projectId: string;
	projectName: string;
	cwd: string;
	inputText?: string;
	toolName?: string;
	toolInput?: string;
	toolOutput?: string;
	assistantText?: string;
	isError?: boolean;
}

export interface InstinctRecord {
	id: string;
	title: string;
	trigger: string;
	confidence: number;
	domain: string;
	source: string;
	scope: InstinctScope;
	projectId?: string;
	projectName?: string;
	content: string;
	created?: string;
	updated?: string;
	importedFrom?: string;
	promotedFrom?: string;
}

export interface LoadedInstinct extends InstinctRecord {
	filePath: string;
	sourceType: "personal" | "inherited";
	scopeLabel: InstinctScope;
}

export interface InstinctDraft {
	id: string;
	title: string;
	trigger: string;
	confidence: number;
	domain: string;
	scope: InstinctScope;
	action: string;
	evidence: string[];
}

export interface ObserverState {
	lastAnalyzedIndex: number;
	lastAnalyzedAt?: string;
}

export interface ClusterCandidate {
	key: string;
	title: string;
	trigger: string;
	instincts: LoadedInstinct[];
	averageConfidence: number;
	domains: string[];
	scopes: InstinctScope[];
}

export interface EvolveAnalysis {
	skillCandidates: ClusterCandidate[];
	promptCandidates: LoadedInstinct[];
	agentCandidates: ClusterCandidate[];
}

export interface ImportSummary {
	added: LoadedInstinct[];
	updated: LoadedInstinct[];
	skipped: LoadedInstinct[];
}

export interface SkillCreateQualityReport {
	verdict: "save" | "improve-then-save" | "absorb" | "drop";
	rationale: string;
	checklist: string[];
	overlapSkills: string[];
	droppedInstinctIds: string[];
	absorbTarget?: string;
	improvements?: string[];
	absorbContent?: string;
	revised?: boolean;
}

export interface SkillCreateMessageDetails {
	repoName: string;
	commitCount: number;
	generationMode: "agentic" | "fallback";
	llmStatus: string;
	modelLabel?: string;
	modelSource: "active" | "settings" | "none";
	skillPath: string;
	instinctPaths: string[];
	prefixes: string[];
	representativeFiles: string[];
	quality: SkillCreateQualityReport;
}

export type LearnEvalApplyStatus =
	| "running"
	| "applied"
	| "not-applied"
	| "skipped-existing"
	| "skipped-missing-target"
	| "skipped-no-content";

export interface LearnEvalMessageDetails {
	projectLabel: string;
	verdict: SkillCreateQualityReport["verdict"];
	scope: InstinctScope;
	target: string;
	targetPath: string | null;
	applied: boolean;
	awaitingConfirmation: boolean;
	applyStatus?: LearnEvalApplyStatus;
	applyMessage?: string;
	phase?: string;
	rationale: string;
	checklist: string[];
	improvements?: string[];
	absorbTarget?: string;
	absorbContent?: string;
	skillMarkdown?: string | null;
}

export interface AgentRunMessageDetails {
	agentName: string;
	agentPath: string;
	executionMode?: string;
	modelLabel: string;
	task: string;
	output: string;
	sessionId: string;
}
