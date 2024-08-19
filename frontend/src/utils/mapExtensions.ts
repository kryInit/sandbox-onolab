export {};

declare global {
    interface Map<K, V> {
        keysAsArray(): K[];
        valuesAsArray(): V[];
        entriesAsArray(): [K, V][];
        overwrite(key: K, value: V): V | undefined;
        rollbackOverwrite(key: K, prevValue: V | undefined): void;
        map<T>(mapper: (key: K, value: V) => T): Map<K, T>;
        toArray<T>(mapper: (key: K, value: V) => T): T[];
        toSet<T>(mapper: (key: K, value: V) => T): Set<T>;
        toObject<TKey extends string | number | symbol, TValue>(this: Map<TKey, TValue>): Record<TKey, TValue>;
        filter<S extends V>(predicate: (key: K, value: V) => value is S): Map<K, S>;
        filter(predicate: (key: K, value: V) => boolean): Map<K, V>;
        concat<T>(other: Map<K, T>): Map<K, V | T>;
        zip<U>(other: Map<K, U>): Map<K, [V, U]>;
    }
}

Map.prototype.keysAsArray = function <TKey, TValue>(this: Map<TKey, TValue>): TKey[] {
    return Array.from(this.keys());
};
Map.prototype.valuesAsArray = function <TKey, TValue>(this: Map<TKey, TValue>): TValue[] {
    return Array.from(this.values());
};
Map.prototype.entriesAsArray = function <TKey, TValue>(this: Map<TKey, TValue>): [TKey, TValue][] {
    return Array.from(this.entries());
};
Map.prototype.entriesAsArray = function <TKey, TValue>(this: Map<TKey, TValue>): [TKey, TValue][] {
    return Array.from(this.entries());
};
Map.prototype.overwrite = function <TKey, TValue>(this: Map<TKey, TValue>, key: TKey, value: TValue): TValue | undefined {
    const prev = this.get(key);
    this.set(key, value);
    return prev;
};
Map.prototype.rollbackOverwrite = function <TKey, TValue>(key: TKey, prevValue: TValue | undefined): void {
    if (prevValue === undefined) this.delete(key);
    else this.set(key, prevValue);
};

Map.prototype.map = function <TKey, TValue, TNextValue>(this: Map<TKey, TValue>, mapper: (key: TKey, value: TValue) => TNextValue): Map<TKey, TNextValue> {
    return Array.from(this.entries()).toDictionary(
        ([key]) => key,
        ([key, value]) => mapper(key, value),
    );
};

Map.prototype.toArray = function <TKey, TValue, TNextValue>(this: Map<TKey, TValue>, mapper: (key: TKey, value: TValue) => TNextValue): TNextValue[] {
    return Array.from(this.entries()).map(([key, value]) => mapper(key, value));
};
Map.prototype.toSet = function <TKey, TValue, TNextValue>(this: Map<TKey, TValue>, mapper: (key: TKey, value: TValue) => TNextValue): Set<TNextValue> {
    return new Set(this.toArray(mapper));
};
Map.prototype.toObject = function <TKey extends string | number | symbol, TValue>(this: Map<TKey, TValue>): Record<TKey, TValue> {
    const ret = {} as Record<TKey, TValue>;
    this.forEach((value, key) => {
        ret[key] = value;
    });
    return ret;
};

Map.prototype.filter = function <TKey, TValue, TResultValue extends TValue>(
    this: Map<TKey, TValue>,
    cond: (key: TKey, value: TValue) => value is TResultValue,
): Map<TKey, TResultValue> {
    return Array.from(this.entries())
        .filter((kvp): kvp is [TKey, TResultValue] => cond(kvp[0], kvp[1]))
        .toDictionary(
            ([key]) => key,
            ([, value]) => value,
        );
};

Map.prototype.concat = function <TKey, TValue0, TValue1>(this: Map<TKey, TValue0>, other: Map<TKey, TValue1>): Map<TKey, TValue0 | TValue1> {
    return ([] as [TKey, TValue0 | TValue1][])
        .concat(this.entriesAsArray())
        .concat(other.entriesAsArray())
        .toDictionary(
            ([key]) => key,
            ([_, value]) => value,
        );
};

Map.prototype.zip = function <K, V, U>(this: Map<K, V>, other: Map<K, U>): Map<K, [V, U]> {
    return this.keysAsArray()
        .filter((key) => other.has(key))
        .toDictionary(
            (key) => key,
            (key) => [this.get(key)!, other.get(key)!],
        );
};
