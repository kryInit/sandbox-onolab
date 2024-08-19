export {};

declare global {
    interface Array<T> {
        mapSelf<S>(mapper: (self: T[]) => S): S;
        back(): T;
        clampSize(size: number): void;
        resize(size: number, defaultValue: T): void;
        zip<U>(arr: Array<U>): Array<[T, U]>;
        filterMap<S extends T, U>(predicate: (value: T, index: number, array: T[]) => value is S, mapper: (value: S, index: number, array: T[]) => U): U[];
        groupBy<K>(getKey: (value: T, idx: number, src: T[]) => K): [K, T[]][];
        count(filter: (val: T) => boolean): number;
        scan<U>(callback: (previousValue: U, currentValue: T, currentIndex: number, array: T[]) => U, initialValue: U): U[];
        max<U extends number>(this: Array<U>): number;
        min<U extends number>(this: Array<U>): number;
        prod<U extends number>(this: Array<U>): number;
        sum(mapper?: (value: T) => number): number;
        unique(): T[];
        uniqueCount(): number;
        toSet(): Set<T>;
        toObject<TKey extends string | number | symbol, TValue>(
            toKey: (value: T, idx: number, src: T[]) => TKey,
            toValue: (value: T, idx: number, src: T[]) => TValue,
        ): Record<TKey, TValue>;
        toDictionary<TKey, TValue>(toKey: (value: T, idx: number, src: T[]) => TKey, toValue: (value: T, idx: number, src: T[]) => TValue): Map<TKey, TValue>;
        toGroupedDictionary<TKey, TValue>(toKey: (value: T, idx: number, src: T[]) => TKey, toValue: (value: T, idx: number, src: T[]) => TValue): Map<TKey, TValue[]>;
    }
}

Array.prototype.mapSelf = function <T, S>(this: Array<T>, mapper: (self: T[]) => S): S {
    return mapper(this);
};
Array.prototype.back = function <T>(this: Array<T>): T {
    return this[this.length - 1];
};
Array.prototype.clampSize = function (size: number) {
    while (this.length > size) this.pop();
};
Array.prototype.resize = function <T>(this: Array<T>, size: number, defaultValue: T) {
    this.clampSize(size);
    while (this.length < size) this.push(defaultValue);
};

Array.prototype.zip = function <T, U>(this: Array<T>, arr: Array<U>): Array<[T, U]> {
    return Array.from({ length: Math.min(this.length, arr.length) }, (_, i) => [this[i], arr[i]]);
};

Array.prototype.filterMap = function <T, S extends T, U>(
    this: Array<T>,
    predicate: (value: T, index: number, array: T[]) => value is S,
    mapper: (value: S, index: number, array: T[]) => U,
): Array<U> {
    return this.map((x, i) => [x, i] as const)
        .filter((args): args is [S, number] => predicate(args[0], args[1], this))
        .map(([x, i]) => mapper(x, i, this));
};

Array.prototype.groupBy = function <K, V>(this: Array<V>, getKey: (value: V, idx: number, src: V[]) => K): [K, V[]][] {
    const retMap = this.reduce((acc: Map<K, V[]>, value, idx) => {
        const key = getKey(value, idx, this);
        const values = acc.get(key);
        if (values !== undefined) values.push(value);
        else acc.set(key, [value]);
        return acc;
    }, new Map());

    // jsのMapは挿入順の順序が保たれる
    return Array.from(retMap);
};
Array.prototype.count = function <T>(this: Array<T>, filter: (val: T) => boolean): number {
    return this.reduce((count, val) => (filter(val) ? count + 1 : count), 0);
};
Array.prototype.scan = function <T, U>(this: Array<T>, callback: (previousValue: U, currentValue: T, currentIndex: number, array: T[]) => U, initialValue: U): U[] {
    let prev = initialValue;
    return this.map((current, idx) => (prev = callback(prev, current, idx, this)));
};
Array.prototype.max = function <T extends number>(this: Array<T>): number {
    return Math.max(...this.map((e) => Number(e)));
};
Array.prototype.min = function <T extends number>(this: Array<T>): number {
    return Math.min(...this.map((e) => Number(e)));
};
Array.prototype.prod = function <T extends number>(this: Array<T>): number {
    return this.reduce((current, x) => current * Number(x), 1);
};
Array.prototype.sum = function <T>(this: Array<T>, mapper: (value: T) => number = Number): number {
    return this.reduce((current, x) => current + mapper(x), 0);
};

Array.prototype.unique = function <T>(this: Array<T>): T[] {
    return Array.from(new Set(this));
};
Array.prototype.uniqueCount = function <T>(this: Array<T>): number {
    return new Set(this).size;
};
Array.prototype.toSet = function <T>(this: Array<T>): Set<T> {
    return new Set(this);
};
Array.prototype.toObject = function <T, TKey extends string | number | symbol, TValue>(
    this: Array<T>,
    toKey: (value: T, idx: number, src: T[]) => TKey,
    toValue: (value: T, idx: number, src: T[]) => TValue,
): Record<TKey, TValue> {
    const ret = {} as Record<TKey, TValue>;
    this.forEach((x, index, self) => {
        const key = toKey(x, index, self);
        const value = toValue(x, index, self);
        ret[key] = value;
    });
    return ret;
};
Array.prototype.toDictionary = function <T, TKey, TValue>(
    this: Array<T>,
    toKey: (value: T, idx: number, src: T[]) => TKey,
    toValue: (value: T, idx: number, src: T[]) => TValue,
): Map<TKey, TValue> {
    const ret = new Map<TKey, TValue>();
    this.forEach((x, index, self) => {
        const key = toKey(x, index, self);
        const value = toValue(x, index, self);
        ret.set(key, value);
    });
    return ret;
};
Array.prototype.toGroupedDictionary = function <T, TKey, TValue>(
    this: Array<T>,
    toKey: (value: T, idx: number, src: T[]) => TKey,
    toValue: (value: T, idx: number, src: T[]) => TValue,
): Map<TKey, TValue[]> {
    const ret = new Map<TKey, TValue[]>();
    for (let i = 0; i < this.length; ++i) {
        const key = toKey(this[i], i, this);
        const value = toValue(this[i], i, this);
        if (!ret.has(key)) ret.set(key, [value]);
        else ret.get(key)!.push(value);
    }
    return ret;
};
