import { describe, expect, it } from 'vitest';

import {
  createConversionGraph,
  findCheapestPath,
  registerConversionRule,
  type ConversionRule,
} from '../src/index.js';

function rule(source: string, target: string, cost: number): ConversionRule {
  return {
    source,
    target,
    cost,
    convert: (input) => input,
  };
}

describe('conversion graph', () => {
  it('returns the lowest-cost composed rule chain', () => {
    const aToB = rule('A', 'B', 1);
    const bToC = rule('B', 'C', 1);
    const aToC = rule('A', 'C', 5);
    const graph = createConversionGraph([aToB, bToC, aToC]);

    expect(findCheapestPath(graph, 'A', 'C')).toEqual([aToB, bToC]);
  });

  it('returns undefined when the target is unreachable', () => {
    const graph = createConversionGraph([rule('A', 'B', 1)]);

    expect(findCheapestPath(graph, 'A', 'C')).toBeUndefined();
  });

  it('breaks equal-cost ties by stable rule order', () => {
    const aToB = rule('A', 'B', 1);
    const bToD = rule('B', 'D', 1);
    const aToC = rule('A', 'C', 1);
    const cToD = rule('C', 'D', 1);
    const graph = createConversionGraph([aToB, bToD, aToC, cToD]);

    expect(findCheapestPath(graph, 'A', 'D')).toEqual([aToB, bToD]);
  });

  it('enumerates only simple paths when rules contain cycles', () => {
    const aToB = rule('A', 'B', 1);
    const bToA = rule('B', 'A', 1);
    const bToC = rule('B', 'C', 1);
    const graph = createConversionGraph([aToB, bToA, bToC]);

    expect(findCheapestPath(graph, 'A', 'C')).toEqual([aToB, bToC]);
  });

  it('registers rules without mutating the existing graph', () => {
    const empty = createConversionGraph();
    const aToB = rule('A', 'B', 1);
    const graph = registerConversionRule(empty, aToB);

    expect(empty.rules).toEqual([]);
    expect(graph.rules).toEqual([aToB]);
  });
});
