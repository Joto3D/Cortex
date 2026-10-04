// JavaScript port of cortex/assignment.py's KeywordInterpreter, used by the
// website demo. tests/test_site.py checks that it gives the same plans as the
// Python version. The real app also uses Claude to understand free-form phrasing.

const SPLIT = /\s*(?:[,;.!]|\bthen\b|\bafter that\b|\bafterwards\b|\band\b|\bnext\b|\bfinally\b)\s*/;
const EVERYTHING = /\b(everything|all (?:the |my )?(?:chores|work|tasks|jobs)|whole farm|do it all|all of it)\b/;
const NUMBER_WORDS = {
  one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8,
  nine: 9, ten: 10, eleven: 11, twelve: 12, fifteen: 15, twenty: 20, "a dozen": 12,
};
const NUMBER = new RegExp("\\b(\\d+|" + Object.keys(NUMBER_WORDS).join("|") + ")\\b");
const FILLER = new Set([
  "first", "please", "can", "you", "could", "go", "the", "a", "an", "my", "all", "of", "to", "then",
  "it", "i", "want", "would", "like", "do", "some", "start", "by", "with", "also", "now", "and",
]);

const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function parseNumber(clause) {
  const m = clause.match(NUMBER);
  if (!m) return null;
  const n = /^\d+$/.test(m[1]) ? parseInt(m[1], 10) : NUMBER_WORDS[m[1]];
  return n > 0 ? n : null;
}

/** Turn an assignment into {steps: [{task, limit}], unsupported: [string]}. */
export function interpret(assignment, tasks) {
  const patterns = tasks
    .filter((t) => t.keywords.length)
    .map((t) => [t.name, new RegExp("\\b(?:" + t.keywords.map(escape).join("|") + ")(?:s|es|ing|ed)?\\b")]);
  const steps = [];
  const unsupported = [];
  for (let clause of assignment.toLowerCase().split(SPLIT)) {
    clause = clause.trim();
    if (!clause) continue;
    const hits = [];
    for (const [name, re] of patterns) {
      const m = re.exec(clause);
      if (m) hits.push([m.index, name]);
    }
    hits.sort((a, b) => a[0] - b[0] || (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0));
    if (hits.length) {
      const limit = parseNumber(clause);
      for (const [, name] of hits) steps.push({ task: name, limit });
    } else if (EVERYTHING.test(clause)) {
      for (const t of tasks) steps.push({ task: t.name, limit: null });
    } else {
      const words = clause.match(/[a-z']+/g) || [];
      if (words.some((w) => !FILLER.has(w))) unsupported.push(clause);
    }
  }
  return { steps, unsupported };
}
