import { parseFrontmatter } from "@earendil-works/pi-coding-agent";
import { normalizeCompareText, overlapScore } from "./instinct-quality.js";

export interface ExistingSkillReference {
	name: string;
	description: string;
	filePath: string;
	bodyText: string;
}

interface SkillSignals {
	title: string;
	headings: string[];
	actions: string[];
	body: string;
	sentences: string[];
	lines: string[];
	paragraphs: string[];
	sections: Array<{ heading: string; body: string }>;
}

export interface SkillOverlapMatch {
	filePath: string;
	score: number;
	titleScore: number;
	headingScore: number;
	actionScore: number;
	sentenceScore: number;
	lineScore: number;
	sectionScore: number;
	paragraphScore: number;
	bodyScore: number;
}

function stripFencedCode(body: string): string {
	return body.replace(/```[\s\S]*?```/gu, "\n");
}

function extractHeadings(body: string): string[] {
	return body
		.split("\n")
		.map((line) => line.trim())
		.filter((line) => /^#{1,6}\s+/u.test(line))
		.map((line) => line.replace(/^#{1,6}\s+/u, "").trim())
		.filter((line) => line.length > 0);
}

function extractActions(body: string): string[] {
	return body
		.split("\n")
		.map((line) => line.trim())
		.filter((line) => /^([-*]|\d+\.)\s+/u.test(line))
		.map((line) => line.replace(/^([-*]|\d+\.)\s+/u, "").trim())
		.filter((line) => line.length > 0);
}

function extractSentences(body: string): string[] {
	return body
		.split(/(?<=[.!?。！？])\s+|\n/gu)
		.map((chunk) => normalizeCompareText(chunk))
		.filter((chunk) => chunk.length >= 30);
}

function extractLongLines(body: string): string[] {
	return body
		.split("\n")
		.map((line) => normalizeCompareText(line))
		.filter((line) => line.length >= 30);
}

function extractParagraphs(body: string): string[] {
	return body
		.split(/\n\s*\n/gu)
		.map((chunk) => normalizeCompareText(chunk))
		.filter((chunk) => chunk.length >= 40);
}

function extractSections(body: string): Array<{ heading: string; body: string }> {
	const lines = body.split("\n");
	const sections: Array<{ heading: string; body: string }> = [];
	let currentHeading = "";
	let buffer: string[] = [];

	const flush = () => {
		const normalizedBody = normalizeCompareText(buffer.join("\n"));
		if (normalizedBody.length === 0) {
			buffer = [];
			return;
		}
		sections.push({
			heading: normalizeCompareText(currentHeading),
			body: normalizedBody,
		});
		buffer = [];
	};

	for (const rawLine of lines) {
		const line = rawLine.trim();
		if (/^#{1,6}\s+/u.test(line)) {
			flush();
			currentHeading = line.replace(/^#{1,6}\s+/u, "").trim();
			continue;
		}
		buffer.push(rawLine);
	}
	flush();
	return sections.filter((section) => section.body.length >= 40);
}

function bestAverageOverlap(candidates: string[], existing: string[], topN: number = 3): number {
	if (candidates.length === 0 || existing.length === 0) {
		return 0;
	}
	const scores = candidates.map((candidate) => {
		let best = 0;
		for (const existingEntry of existing) {
			best = Math.max(best, overlapScore(candidate, existingEntry));
		}
		return best;
	});
	const top = scores.sort((left, right) => right - left).slice(0, topN);
	return top.reduce((sum, value) => sum + value, 0) / top.length;
}

function bestAverageSectionOverlap(
	candidates: Array<{ heading: string; body: string }>,
	existing: Array<{ heading: string; body: string }>,
): number {
	if (candidates.length === 0 || existing.length === 0) {
		return 0;
	}
	const scores = candidates.map((candidate) => {
		let best = 0;
		for (const existingSection of existing) {
			const headingScore = overlapScore(candidate.heading, existingSection.heading);
			const bodyScore = overlapScore(candidate.body, existingSection.body);
			const combined = headingScore * 0.35 + bodyScore * 0.65;
			best = Math.max(best, combined);
		}
		return best;
	});
	const top = scores.sort((left, right) => right - left).slice(0, 3);
	return top.reduce((sum, value) => sum + value, 0) / top.length;
}

function buildSkillSignals(markdown: string, fallbackName: string, fallbackDescription: string): SkillSignals {
	const { frontmatter, body } = parseFrontmatter<Record<string, unknown>>(markdown);
	const normalizedBody = stripFencedCode(body).trim();
	const title =
		(typeof frontmatter.name === "string" && frontmatter.name.trim().length > 0
			? frontmatter.name.trim()
			: normalizedBody.match(/^#\s+(.+)$/mu)?.[1]?.trim()) ?? fallbackName;

	return {
		title: normalizeCompareText(`${title} ${fallbackDescription}`),
		headings: extractHeadings(normalizedBody).map((item) => normalizeCompareText(item)),
		actions: extractActions(normalizedBody).map((item) => normalizeCompareText(item)),
		body: normalizeCompareText(normalizedBody),
		sentences: extractSentences(normalizedBody),
		lines: extractLongLines(normalizedBody),
		paragraphs: extractParagraphs(normalizedBody),
		sections: extractSections(normalizedBody),
	};
}

function overlapRatio(left: string[], right: string[]): number {
	if (left.length === 0 || right.length === 0) {
		return 0;
	}
	const rightSet = new Set(right);
	const matches = left.filter((item) => rightSet.has(item)).length;
	return matches / Math.min(left.length, right.length);
}

export function scoreSkillOverlap(candidateMarkdown: string, existingSkill: ExistingSkillReference): SkillOverlapMatch {
	const candidate = buildSkillSignals(candidateMarkdown, "", "");
	const existing = buildSkillSignals(existingSkill.bodyText, existingSkill.name, existingSkill.description);

	const titleScore = overlapScore(candidate.title, existing.title);
	const headingScore = overlapScore(candidate.headings.join(" "), existing.headings.join(" "));
	const actionScore = overlapScore(candidate.actions.join(" "), existing.actions.join(" "));
	const sentenceScore = bestAverageOverlap(candidate.sentences, existing.sentences, 4);
	const lineScore = overlapRatio(candidate.lines, existing.lines);
	const sectionScore = bestAverageSectionOverlap(candidate.sections, existing.sections);
	const paragraphScore = bestAverageOverlap(candidate.paragraphs, existing.paragraphs);
	const bodyScore = overlapScore(candidate.body, existing.body);

	let score =
		titleScore * 0.1 +
		headingScore * 0.11 +
		actionScore * 0.15 +
		sentenceScore * 0.12 +
		lineScore * 0.1 +
		sectionScore * 0.2 +
		paragraphScore * 0.16 +
		bodyScore * 0.06;
	if (titleScore >= 0.82 && (headingScore >= 0.45 || actionScore >= 0.45 || sectionScore >= 0.45)) {
		score = Math.max(score, 0.82);
	}
	if (sentenceScore >= 0.72 && lineScore >= 0.5) {
		score = Math.max(score, 0.84);
	}
	if (sectionScore >= 0.62 && paragraphScore >= 0.58) {
		score = Math.max(score, 0.86);
	}
	if (paragraphScore >= 0.68 && (actionScore >= 0.48 || headingScore >= 0.48 || sentenceScore >= 0.55)) {
		score = Math.max(score, 0.84);
	}
	if (
		bodyScore >= 0.55 &&
		(headingScore >= 0.35 || actionScore >= 0.35 || paragraphScore >= 0.4 || lineScore >= 0.4)
	) {
		score = Math.max(score, 0.78);
	}

	return {
		filePath: existingSkill.filePath,
		score,
		titleScore,
		headingScore,
		actionScore,
		sentenceScore,
		lineScore,
		sectionScore,
		paragraphScore,
		bodyScore,
	};
}

export function detectOverlappingSkills(
	candidateMarkdown: string,
	existingSkills: ExistingSkillReference[],
	options?: { limit?: number; threshold?: number },
): SkillOverlapMatch[] {
	const threshold = options?.threshold ?? 0.52;
	const limit = options?.limit ?? 3;
	return existingSkills
		.map((skill) => scoreSkillOverlap(candidateMarkdown, skill))
		.filter((match) => match.score >= threshold)
		.sort((left, right) => right.score - left.score)
		.slice(0, limit);
}
