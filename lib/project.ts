import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { realpathSync, statSync } from "node:fs";
import { basename, resolve } from "node:path";
import { promisify } from "node:util";
import type { ProjectInfo } from "./types.js";

const execFileAsync = promisify(execFile);

async function runGit(args: string[], cwd?: string): Promise<string | undefined> {
	try {
		const { stdout } = await execFileAsync("git", args, {
			cwd,
			timeout: 5000,
		});
		const value = stdout.trim();
		return value.length > 0 ? value : undefined;
	} catch {
		return undefined;
	}
}

function stripRemoteCredentials(remote: string | undefined): string | undefined {
	if (!remote) {
		return undefined;
	}
	return remote.replace(/:\/\/[^@]+@/u, "://");
}

function canonicalizePath(path: string): string {
	const resolved = resolve(path);
	try {
		return realpathSync.native(resolved);
	} catch {
		return resolved;
	}
}

function resolveExistingDirectory(path: string): string | undefined {
	const resolved = resolve(path);
	try {
		if (!statSync(resolved).isDirectory()) {
			return undefined;
		}
		return canonicalizePath(resolved);
	} catch {
		return undefined;
	}
}

function resolveProjectRootOverride(cwd: string): string | undefined {
	const candidates = [process.env.PI_PROJECT_DIR, process.env.CLAUDE_PROJECT_DIR]
		.map((value) => value?.trim())
		.filter((value): value is string => Boolean(value));
	for (const candidate of candidates) {
		if (candidate.length > 0) {
			const resolved = resolveExistingDirectory(resolve(cwd, candidate));
			if (resolved) {
				return resolved;
			}
		}
	}
	return undefined;
}

export async function detectProject(cwd: string): Promise<ProjectInfo> {
	const resolvedCwd = canonicalizePath(cwd);
	const overrideRoot = resolveProjectRootOverride(resolvedCwd);
	const gitRoot = overrideRoot ?? (await runGit(["-C", resolvedCwd, "rev-parse", "--show-toplevel"]));
	if (!gitRoot) {
		return {
			id: "global",
			name: "global",
			root: resolvedCwd,
		};
	}
	const projectRoot = canonicalizePath(gitRoot);
	const remote = stripRemoteCredentials(await runGit(["-C", projectRoot, "remote", "get-url", "origin"]));
	const projectHashSource = remote ?? projectRoot;
	const id = createHash("sha256").update(projectHashSource).digest("hex").slice(0, 12);
	const name = basename(projectRoot) || "project";
	return {
		id,
		name,
		root: projectRoot,
		remote,
	};
}
