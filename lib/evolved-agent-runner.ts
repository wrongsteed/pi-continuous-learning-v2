import { readFile } from "node:fs/promises";
import { basename, extname, join, resolve } from "node:path";
import type { Model, TextContent } from "@earendil-works/pi-ai";
import {
	buildSessionContext,
	createAgentSession,
	createBashTool,
	createEditTool,
	createFindTool,
	createGrepTool,
	createLsTool,
	createReadTool,
	createWriteTool,
	DefaultResourceLoader,
	type ExtensionContext,
	type ModelRegistry,
	parseFrontmatter,
	SessionManager,
} from "@earendil-works/pi-coding-agent";
import type { ProjectInfo, StorageLayout } from "./types.js";

export interface ResolvedEvolvedAgent {
	name: string;
	description?: string;
	preferredModel?: string;
	executionMode?: string;
	tools?: string[];
	filePath: string;
	systemPrompt: string;
}

export interface RunEvolvedAgentResult {
	agent: ResolvedEvolvedAgent;
	modelLabel: string;
	task: string;
	output: string;
	sessionId: string;
}

function parseToolList(value: unknown): string[] | undefined {
	if (typeof value !== "string") {
		return undefined;
	}
	const tools = value
		.split(",")
		.map((item) => item.trim().toLowerCase())
		.filter((item) => item.length > 0);
	return tools.length > 0 ? tools : undefined;
}

function flattenAssistantText(content: unknown): string {
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
		.join("\n")
		.trim();
}

function resolveToolSet(cwd: string, tools?: string[]) {
	const allTools = {
		read: createReadTool(cwd),
		bash: createBashTool(cwd),
		edit: createEditTool(cwd),
		write: createWriteTool(cwd),
		grep: createGrepTool(cwd),
		find: createFindTool(cwd),
		ls: createLsTool(cwd),
	};

	if (!tools || tools.length === 0) {
		return [allTools.read, allTools.bash, allTools.edit, allTools.write];
	}

	const selected = tools.map((tool) => allTools[tool as keyof typeof allTools]).filter(Boolean);
	return selected.length > 0 ? selected : [allTools.read, allTools.bash, allTools.edit, allTools.write];
}

function buildManualAgentSystemPrompt(agent: ResolvedEvolvedAgent): string {
	return [
		"You are executing an evolved agent artifact manually inside pi.",
		"Treat the following artifact body as the authoritative operating instructions for this run.",
		"Do not mention frontmatter or artifact mechanics unless asked.",
		"",
		agent.systemPrompt.trim(),
	].join("\n\n");
}

function parsePreferredModel(
	modelHint: string | undefined,
	modelRegistry: ModelRegistry,
	fallback: Model<any>,
): Model<any> {
	if (!modelHint || !modelHint.includes("/")) {
		return fallback;
	}
	const [provider, ...rest] = modelHint.split("/");
	const modelId = rest.join("/");
	const explicit = modelRegistry.find(provider, modelId);
	return explicit ?? fallback;
}

export async function resolveEvolvedAgent(
	cwd: string,
	layout: StorageLayout,
	ref: string,
): Promise<ResolvedEvolvedAgent> {
	const candidates =
		ref.includes("/") || ref.endsWith(".md")
			? [resolve(cwd, ref)]
			: [
					join(layout.projectEvolvedAgentsDir, ref),
					join(layout.projectEvolvedAgentsDir, `${ref}.md`),
					join(layout.projectEvolvedAgentsDir, `${ref}-agent.md`),
					join(layout.globalEvolvedAgentsDir, ref),
					join(layout.globalEvolvedAgentsDir, `${ref}.md`),
					join(layout.globalEvolvedAgentsDir, `${ref}-agent.md`),
				];

	let filePath: string | undefined;
	for (const candidate of candidates) {
		try {
			await readFile(candidate, "utf-8");
			filePath = candidate;
			break;
		} catch {}
	}
	if (!filePath) {
		throw new Error(`未找到 evolved agent: ${ref}`);
	}

	const raw = await readFile(filePath, "utf-8");
	const { frontmatter, body } = parseFrontmatter<Record<string, unknown>>(raw);
	return {
		name:
			typeof frontmatter.name === "string" && frontmatter.name.trim().length > 0
				? frontmatter.name.trim()
				: basename(filePath, extname(filePath)),
		description:
			typeof frontmatter.description === "string" && frontmatter.description.trim().length > 0
				? frontmatter.description.trim()
				: undefined,
		preferredModel:
			typeof frontmatter.model === "string" && frontmatter.model.trim().length > 0
				? frontmatter.model.trim()
				: undefined,
		executionMode:
			typeof frontmatter.execution_mode === "string" && frontmatter.execution_mode.trim().length > 0
				? frontmatter.execution_mode.trim()
				: undefined,
		tools: parseToolList(frontmatter.tools),
		filePath,
		systemPrompt: body.trim(),
	};
}

export async function runEvolvedAgent(options: {
	ctx: ExtensionContext;
	project: ProjectInfo;
	layout: StorageLayout;
	agentRef: string;
	task: string;
	modelOverride?: string;
}): Promise<RunEvolvedAgentResult> {
	const agent = await resolveEvolvedAgent(options.ctx.cwd, options.layout, options.agentRef);
	const fallbackModel = options.ctx.model;
	if (!fallbackModel) {
		throw new Error("当前没有可用模型");
	}
	const preferredModel = options.modelOverride ?? agent.preferredModel;
	const model = parsePreferredModel(preferredModel, options.ctx.modelRegistry, fallbackModel);

	const resourceLoader = new DefaultResourceLoader({
		cwd: options.ctx.cwd,
		noExtensions: true,
		noSkills: true,
		noPromptTemplates: true,
		noThemes: true,
		systemPrompt: buildManualAgentSystemPrompt(agent),
		agentsFilesOverride: () => ({ agentsFiles: [] }),
	});
	await resourceLoader.reload();

	const { session } = await createAgentSession({
		cwd: options.ctx.cwd,
		model,
		modelRegistry: options.ctx.modelRegistry,
		resourceLoader,
		sessionManager: SessionManager.inMemory(options.ctx.cwd),
		tools: resolveToolSet(options.ctx.cwd, agent.tools),
	});

	try {
		await session.prompt(options.task, {
			expandPromptTemplates: false,
			source: "interactive",
		});
		const context = buildSessionContext(session.sessionManager.getEntries(), session.sessionManager.getLeafId());
		const assistantMessages = context.messages.filter((message) => message.role === "assistant");
		const lastAssistant = assistantMessages.at(-1);
		const output = flattenAssistantText(lastAssistant?.content ?? "") || "(no output)";
		return {
			agent,
			modelLabel: `${model.provider}/${model.id}`,
			task: options.task,
			output,
			sessionId: session.sessionId,
		};
	} finally {
		session.dispose();
	}
}
