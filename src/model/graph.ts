export type ConversionRule<Input = unknown, Output = unknown> = Readonly<{
  source: string;
  target: string;
  cost: number;
  convert: (input: Input) => Output;
}>;

/**
 * A rule as the graph stores it, with its input and output types erased so that
 * rules over different representations share one list. Any
 * `ConversionRule<Labelmap, Mesh>` is assignable here: `convert` is
 * contravariant in its parameter, and `never` sits below every input type.
 * The erasure is one-way by design -- a stored rule cannot be called without
 * re-widening its parameter, which is what `applyConversionRule` is for.
 */
export type StoredConversionRule = ConversionRule<never, unknown>;

export type ConversionGraph = Readonly<{
  rules: readonly StoredConversionRule[];
}>;

export function createConversionGraph(
  rules: readonly StoredConversionRule[] = [],
): ConversionGraph {
  return { rules: [...rules] };
}

export function registerConversionRule(
  graph: ConversionGraph,
  rule: StoredConversionRule,
): ConversionGraph {
  return { rules: [...graph.rules, rule] };
}

/**
 * Invoke a stored rule. This is the single place the erased parameter is
 * re-widened: routing picked this rule because its `source` matches the
 * representation being passed, and that link between a string key and a value's
 * type is the part TypeScript cannot follow. Keeping the cast here means the
 * rule list itself stays honest instead of every slot being `any`.
 */
export function applyConversionRule(rule: StoredConversionRule, input: unknown): unknown {
  return (rule.convert as (value: unknown) => unknown)(input);
}

export function findCheapestPath(graph: ConversionGraph, source: string, target: string) {
  if (source === target) {
    return [];
  }

  let cheapest: readonly StoredConversionRule[] | undefined;
  let cheapestCost = Infinity;

  function visit(
    current: string,
    visited: ReadonlySet<string>,
    path: readonly StoredConversionRule[],
    cost: number,
  ) {
    for (const rule of graph.rules) {
      if (rule.source !== current || visited.has(rule.target)) {
        continue;
      }

      const nextPath = [...path, rule];
      const nextCost = cost + rule.cost;
      if (rule.target === target) {
        if (nextCost < cheapestCost) {
          cheapest = nextPath;
          cheapestCost = nextCost;
        }
        continue;
      }

      visit(rule.target, new Set([...visited, rule.target]), nextPath, nextCost);
    }
  }

  visit(source, new Set([source]), [], 0);
  return cheapest;
}
