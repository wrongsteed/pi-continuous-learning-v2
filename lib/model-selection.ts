import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { getAgentDir } from "@earendil-works/pi-coding-agent";

interface ModelLike {
	provider: string;
	id: string;
}

interface ModelRegistryLike {
	find(provider: string, modelId: string): ModelLike | undefined;
}

interface SettingsFile {
	defaultProvider?: string;
	defaultModel?: string;
}

async function loadSettingsFile(): Promise<SettingsFile> {
	try {
		const content = await readFile(join(getAgentDir(), "settings.json"), "utf-8");
		return JSON.parse(content) as SettingsFile;
	} catch {
		return {};
	}
}

export async function resolveActiveOrDefaultModel<T extends ModelLike>(
	activeModel: T | undefined,
	modelRegistry: ModelRegistryLike,
): Promise<{ model: T | undefined; source: "active" | "settings" | "none" }> {
	if (activeModel) {
		return {
			model: activeModel,
			source: "active",
		};
	}

	const settings = await loadSettingsFile();
	if (!settings.defaultProvider || !settings.defaultModel) {
		return {
			model: undefined,
			source: "none",
		};
	}

	const model = modelRegistry.find(settings.defaultProvider, settings.defaultModel) as T | undefined;
	if (model) {
		return {
			model,
			source: "settings",
		};
	}

	return {
		model: undefined,
		source: "none",
	};
}

export async function resolveActivePreferredOrDefaultModel<T extends ModelLike>(
	activeModel: T | undefined,
	preferredModelRef: string | undefined,
	modelRegistry: ModelRegistryLike,
): Promise<{ model: T | undefined; source: "active" | "preferred" | "settings" | "none" }> {
	if (activeModel) {
		return {
			model: activeModel,
			source: "active",
		};
	}

	const normalizedRef = preferredModelRef?.trim();
	if (normalizedRef?.includes("/")) {
		const [provider, ...rest] = normalizedRef.split("/");
		const modelId = rest.join("/");
		if (provider && modelId) {
			const preferred = modelRegistry.find(provider, modelId) as T | undefined;
			if (preferred) {
				return {
					model: preferred,
					source: "preferred",
				};
			}
		}
	}

	const fallback = await resolveActiveOrDefaultModel<T>(undefined, modelRegistry);
	return {
		model: fallback.model,
		source: fallback.source,
	};
}
