import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { homedir, tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import type { Model } from "@earendil-works/pi-ai";
import {
	AuthStorage,
	createAgentSession,
	DefaultResourceLoader,
	ModelRegistry,
	SessionManager,
	type ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import { extractInstinctAction } from "../lib/instinct-quality.js";
import { loadPendingInstincts, loadProjectOnlyInstincts } from "../lib/instincts.js";
import { maybeAnalyzeObservations } from "../lib/observer.js";
import { detectProject } from "../lib/project.js";
import { ensureStorage, getStorageLayout } from "../lib/storage.js";

type ValidationMode = "regression" | "soak";

interface CliArgs {
	mode: ValidationMode;
	provider?: string;
	model?: string;
	rounds: number;
}

interface ValidationSnapshot {
	round: number;
	observerResult: Awaited<ReturnType<typeof maybeAnalyzeObservations>>;
	activeIds: string[];
	pendingIds: string[];
}

function parseArgs(argv: string[]): CliArgs {
	let mode: ValidationMode = "regression";
	let provider: string | undefined;
	let model: string | undefined;
	let rounds = 3;

	for (let index = 0; index < argv.length; index++) {
		const token = argv[index];
		switch (token) {
			case "--mode":
				mode = (argv[index + 1] as ValidationMode | undefined) ?? mode;
				index++;
				break;
			case "--provider":
				provider = argv[index + 1];
				index++;
				break;
			case "--model":
				model = argv[index + 1];
				index++;
				break;
			case "--rounds":
				rounds = Math.max(1, Number.parseInt(argv[index + 1] ?? "3", 10) || 3);
				index++;
				break;
			default:
				break;
		}
	}

	return { mode, provider, model, rounds };
}

async function resolveModel(
	realAgentDir: string,
	modelRegistry: ModelRegistry,
	args: CliArgs,
): Promise<Model<any>> {
	if (args.provider && args.model) {
		const explicit = modelRegistry.find(args.provider, args.model);
		if (!explicit) {
			throw new Error(`Model ${args.provider}/${args.model} not found`);
		}
		return explicit;
	}

	const settingsPath = join(realAgentDir, "settings.json");
	const rawSettings = await readFile(settingsPath, "utf-8");
	const settings = JSON.parse(rawSettings) as { defaultProvider?: string; defaultModel?: string };
	if (!settings.defaultProvider || !settings.defaultModel) {
		throw new Error("settings.json does not define defaultProvider/defaultModel; pass --provider and --model");
	}
	const configured = modelRegistry.find(settings.defaultProvider, settings.defaultModel);
	if (!configured) {
		throw new Error(`Default model ${settings.defaultProvider}/${settings.defaultModel} not found`);
	}
	return configured;
}

function buildPromptRounds(mode: ValidationMode, rounds: number): string[][] {
	const baseRounds = [
		[
			"Inspect Cargo.toml and explain how workspace manifest updates should preserve dependency inheritance. Do not edit files.",
			"Inspect config.example.toml and README.md, then explain how config examples should stay aligned when config defaults change. Do not edit files.",
			"Inspect tests/fixtures/shared.ts and explain how to keep test layout consistent while reusing shared fixtures. Do not edit files.",
		],
		[
			"Review Cargo.toml again and restate the rule about keeping dependency versions inherited from the workspace instead of pinning child crates. Do not edit files.",
			"Review README.md and config.example.toml again and restate the rule about updating examples whenever defaults change. Do not edit files.",
			"Review tests/fixtures/shared.ts again and restate the rule about reusing shared fixtures instead of inventing one-off test data. Do not edit files.",
		],
		[
			"Summarize the manifest-change rule in one sentence, focusing on workspace dependency inheritance. Do not edit files.",
			"Summarize the config rule in one sentence, focusing on keeping documented examples in sync with defaults. Do not edit files.",
			"Summarize the testing rule in one sentence, focusing on shared fixtures over one-off fixtures. Do not edit files.",
		],
	];

	if (mode === "regression") {
		return [baseRounds[0]];
	}
	return baseRounds.slice(0, Math.min(rounds, baseRounds.length));
}

async function seedProject(projectRoot: string): Promise<void> {
	await mkdir(join(projectRoot, "src"), { recursive: true });
	await mkdir(join(projectRoot, "tests", "fixtures"), { recursive: true });

	await writeFile(
		join(projectRoot, "Cargo.toml"),
		[
			"[workspace]",
			'members = ["crates/core", "crates/cli"]',
			"",
			"[workspace.dependencies]",
			'serde = "1.0"',
			'tracing = "0.1"',
		].join("\n"),
		"utf-8",
	);
	await writeFile(
		join(projectRoot, "README.md"),
		[
			"# Temp Repo",
			"",
			"Use workspace inheritance for shared dependency versions.",
			"Update config examples whenever config defaults change.",
			"Reuse shared fixtures when adding tests.",
		].join("\n"),
		"utf-8",
	);
	await writeFile(
		join(projectRoot, "config.example.toml"),
		[
			"[scanner]",
			"parallel = 2",
			'output = "./out"',
		].join("\n"),
		"utf-8",
	);
	await writeFile(
		join(projectRoot, "src", "lib.ts"),
		[
			"export function computeValue(input: number): number {",
			"\treturn input * 2;",
			"}",
		].join("\n"),
		"utf-8",
	);
	await writeFile(
		join(projectRoot, "tests", "fixtures", "shared.ts"),
		[
			"export function makeFixture() {",
			"\treturn { flag: true };",
			"}",
		].join("\n"),
		"utf-8",
	);
}

async function createValidationContext(projectRoot: string, pluginRoot: string, model: Model<any>) {
	const tempAgentDir = await mkdtemp(join(tmpdir(), "pi-agent-dir-"));
	process.env.PI_CODING_AGENT_DIR = tempAgentDir;

	const project = await detectProject(projectRoot);
	const layout = getStorageLayout(project);
	await ensureStorage(project, layout);
	await writeFile(
		layout.configPath,
		JSON.stringify(
			{
				version: "2.1",
				observer: {
					enabled: true,
					runIntervalMinutes: 0,
					minObservationsToAnalyze: 1,
					maxRecentObservations: 100,
				},
			},
			null,
			2,
		),
		"utf-8",
	);
	await writeFile(
		layout.observerStatePath,
		JSON.stringify(
			{
				lastAnalyzedIndex: 0,
			},
			null,
			2,
		),
		"utf-8",
	);

	const realAgentDir = join(homedir(), ".pi", "agent");
	const authStorage = AuthStorage.create(join(realAgentDir, "auth.json"));
	const modelRegistry = ModelRegistry.create(authStorage, join(realAgentDir, "models.json"));
	const resourceLoader = new DefaultResourceLoader({
		cwd: projectRoot,
		agentDir: tempAgentDir,
		noExtensions: true,
		noSkills: true,
		noPromptTemplates: true,
		noThemes: true,
		agentsFilesOverride: () => ({ agentsFiles: [] }),
		additionalExtensionPaths: [pluginRoot],
	});
	await resourceLoader.reload();

	const { session } = await createAgentSession({
		cwd: projectRoot,
		agentDir: tempAgentDir,
		model,
		modelRegistry,
		resourceLoader,
		sessionManager: SessionManager.inMemory(projectRoot),
		tools: ["read", "grep", "bash", "edit", "write"],
	});

	const fakeCtx = {
		cwd: projectRoot,
		hasUI: false,
		model,
		modelRegistry,
		isIdle: () => true,
		signal: undefined,
		abort: () => {},
		shutdown: () => {},
		hasPendingMessages: () => false,
		getContextUsage: () => undefined,
		compact: () => {},
		getSystemPrompt: () => "",
		ui: {
			notify: () => {},
			setStatus: () => {},
			clearStatus: () => {},
			confirm: async () => false,
			select: async () => undefined,
			input: async () => undefined,
			editor: async () => undefined,
			custom: async () => undefined,
			setTitle: () => {},
			setWidget: () => {},
			clearWidget: () => {},
			setFooter: () => {},
			clearFooter: () => {},
			setEditorText: () => {},
			setToolsExpanded: () => {},
		},
	} as unknown as ExtensionContext;

	return { session, project, layout, fakeCtx, tempAgentDir };
}

async function runValidation(args: CliArgs) {
	const pluginRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
	const projectRoot = await mkdtemp(join(tmpdir(), "pi-observer-project-"));
	await seedProject(projectRoot);

	const realAgentDir = join(homedir(), ".pi", "agent");
	const authStorage = AuthStorage.create(join(realAgentDir, "auth.json"));
	const modelRegistry = ModelRegistry.create(authStorage, join(realAgentDir, "models.json"));
	const model = await resolveModel(realAgentDir, modelRegistry, args);
	const { session, project, layout, fakeCtx, tempAgentDir } = await createValidationContext(
		projectRoot,
		pluginRoot,
		model,
	);

	try {
		const rounds = buildPromptRounds(args.mode, args.rounds);
		const snapshots: ValidationSnapshot[] = [];

		for (const [index, prompts] of rounds.entries()) {
			for (const prompt of prompts) {
				await session.prompt(prompt);
			}

			const observerResult = await maybeAnalyzeObservations(fakeCtx, project, layout, {
				running: false,
				timer: null,
				scheduledAnalysis: null,
			});
			const activeInstincts = await loadProjectOnlyInstincts(layout);
			const pendingInstincts = await loadPendingInstincts(layout);
			snapshots.push({
				round: index + 1,
				observerResult,
				activeIds: activeInstincts.map((instinct) => instinct.id).sort(),
				pendingIds: pendingInstincts.map((instinct) => instinct.id).sort(),
			});
		}

		const activeInstincts = await loadProjectOnlyInstincts(layout);
		const pendingInstincts = await loadPendingInstincts(layout);
		const activeIds = activeInstincts.map((instinct) => instinct.id).sort();
		const pendingIds = pendingInstincts.map((instinct) => instinct.id).sort();
		const learnedInstincts = [...activeInstincts, ...pendingInstincts];
		const learnedIds = [...activeIds, ...pendingIds];
		const actions = learnedInstincts.map((instinct) => extractInstinctAction(instinct.content)).sort();
		const learnedText = learnedInstincts
			.map((instinct) => `${instinct.id}\n${instinct.trigger}\n${extractInstinctAction(instinct.content)}`)
			.join("\n")
			.toLowerCase();

		if (learnedIds.length < 1) {
			throw new Error("Expected at least 1 learned active or pending instinct, got 0");
		}
		if (new Set(learnedIds).size !== learnedIds.length) {
			throw new Error("Learned instinct IDs are not unique across active and pending outputs");
		}
		if (!learnedText.includes("workspace") || !learnedText.includes("depend")) {
			throw new Error("Observer did not learn the workspace dependency inheritance theme");
		}
		if (!learnedText.includes("config") || (!learnedText.includes("example") && !learnedText.includes("default"))) {
			throw new Error("Observer did not learn the config example/default alignment theme");
		}
		if (!learnedText.includes("fixture")) {
			throw new Error("Observer did not learn the shared fixture reuse theme");
		}
		if (args.mode === "soak" && learnedIds.length > 5) {
			throw new Error(`Expected soak run to stay within 5 learned instincts, got ${learnedIds.length}`);
		}

		const summary = {
			mode: args.mode,
			rounds: rounds.length,
			projectRoot,
			tempAgentDir,
			model: `${model.provider}/${model.id}`,
			active: activeInstincts.map((instinct) => ({
				id: instinct.id,
				trigger: instinct.trigger,
				confidence: instinct.confidence,
				action: extractInstinctAction(instinct.content),
			})),
			pending: pendingInstincts.map((instinct) => ({
				id: instinct.id,
				trigger: instinct.trigger,
				confidence: instinct.confidence,
				action: extractInstinctAction(instinct.content),
			})),
			snapshots,
			actions,
		};

		console.log(JSON.stringify(summary, null, 2));
	} finally {
		session.dispose();
		await Promise.all([
			rm(projectRoot, { recursive: true, force: true }),
			rm(tempAgentDir, { recursive: true, force: true }),
		]);
	}
}

const args = parseArgs(process.argv.slice(2));
void runValidation(args).catch((error) => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exitCode = 1;
});
