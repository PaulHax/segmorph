export type ConversionRule<Input = unknown, Output = unknown> = Readonly<{
  source: string;
  target: string;
  cost: number;
  convert: (input: Input) => Output;
}>;

export type ConversionGraph = Readonly<{
  rules: readonly ConversionRule<any, any>[];
}>;

export function createConversionGraph(
  rules: readonly ConversionRule<any, any>[] = [],
): ConversionGraph {
  return { rules: [...rules] };
}

export function registerConversionRule(
  graph: ConversionGraph,
  rule: ConversionRule<any, any>,
): ConversionGraph {
  return { rules: [...graph.rules, rule] };
}

export function findCheapestPath(graph: ConversionGraph, source: string, target: string) {
  if (source === target) {
    return [];
  }

  let cheapest: readonly ConversionRule<any, any>[] | undefined;
  let cheapestCost = Infinity;

  function visit(
    current: string,
    visited: ReadonlySet<string>,
    path: readonly ConversionRule<any, any>[],
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
