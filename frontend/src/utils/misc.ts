export const mapObjectValues = <T, U>(obj: Record<string, T>, converter: (x: T) => U): Record<string, U> =>
    Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, converter(v)]));
