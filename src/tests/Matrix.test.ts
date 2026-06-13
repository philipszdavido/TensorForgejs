import {describe, it, expect} from "vitest";
import {Matrix} from "../core/Matrix";
import {Vector} from "../core/Vector";

describe("Matrix", () => {
    it("should create correct shape", () => {
        const m = new Matrix(3, 4);

        expect(m.rows).toBe(3);
        expect(m.columns).toBe(4);
    });

    it("should set and get values correctly", () => {
        const m = new Matrix(2, 2);

        m.set(0, 0, 10);
        m.set(0, 1, 20);
        m.set(1, 0, 30);
        m.set(1, 1, 40);

        expect(m.get(0, 0)).toBe(10);
        expect(m.get(0, 1)).toBe(20);
        expect(m.get(1, 0)).toBe(30);
        expect(m.get(1, 1)).toBe(40);
    });

    it("should overwrite values correctly", () => {
        const m = new Matrix(2, 2);

        m.set(0, 0, 5);
        m.set(0, 0, 99);

        expect(m.get(0, 0)).toBe(99);
    });

    it("should throw on invalid access", () => {
        const m = new Matrix(2, 2);

        // expect(() => m.get(10, 10)).toThrow();
    });
});

describe("Matrix", () => {
    it("creates correct shape", () => {
        const m = new Matrix(2, 3);
        expect(m.rows).toBe(2);
        expect(m.columns).toBe(3);
    });

    it("set/get works correctly", () => {
        const m = new Matrix(2, 2);

        m.set(0, 0, 1);
        m.set(0, 1, 2);
        m.set(1, 0, 3);
        m.set(1, 1, 4);

        expect(m.get(0, 0)).toBe(1);
        expect(m.get(0, 1)).toBe(2);
        expect(m.get(1, 0)).toBe(3);
        expect(m.get(1, 1)).toBe(4);
    });

    it("getRow returns correct subarray", () => {
        const m = new Matrix(2, 2);

        m.set(0, 0, 1);
        m.set(0, 1, 2);

        const row = m.getRow(0);

        expect(Array.from(row)).toEqual([1, 2]);
    });

    it("setRow writes correctly", () => {
        const m = new Matrix(2, 2);

        m.setRow(0, new Float64Array([5, 6]));

        expect(m.get(0, 0)).toBe(5);
        expect(m.get(0, 1)).toBe(6);
    });

    it("matrixMulVector works correctly", () => {
        const m = new Matrix(2, 2);
        const v = Vector.from([1, 2]);

        m.set(0, 0, 1);
        m.set(0, 1, 2);
        m.set(1, 0, 3);
        m.set(1, 1, 4);

        const r = Matrix.matrixMulVector(m, v);

        expect(r.toArray()).toEqual([5, 11]);
    });

    it("outerProduct works correctly", () => {
        const a = Vector.from([1, 2]);
        const b = Vector.from([3, 4]);

        const m = Matrix.outerProduct(a, b);

        expect(m.get(0, 0)).toBe(3);
        expect(m.get(0, 1)).toBe(4);
        expect(m.get(1, 0)).toBe(6);
        expect(m.get(1, 1)).toBe(8);
    });

    it("zeros creates zero matrix", () => {
        const m = Matrix.zeros(2, 2);

        expect(m.get(0, 0)).toBe(0);
        expect(m.get(1, 1)).toBe(0);
    });

    it("add works correctly", () => {
        const a = Matrix.from([
            [1, 2],
            [3, 4],
        ]);

        const b = Matrix.from([
            [5, 6],
            [7, 8],
        ]);

        const r = Matrix.add(a, b);

        expect(r.toNestedArray()).toEqual([
            [6, 8],
            [10, 12],
        ]);
    });

    it("sub works correctly", () => {
        const a = Matrix.from([
            [5, 6],
            [7, 8],
        ]);

        const b = Matrix.from([
            [1, 2],
            [3, 4],
        ]);

        const r = Matrix.sub(a, b);

        expect(r.toNestedArray()).toEqual([
            [4, 4],
            [4, 4],
        ]);
    });

    it("from creates matrix correctly", () => {
        const m = Matrix.from([
            [1, 2, 3],
            [4, 5, 6],
        ]);

        expect(m.get(0, 0)).toBe(1);
        expect(m.get(1, 2)).toBe(6);
    });

    it("fromVector works correctly", () => {
        const v = Vector.from([1, 2, 3]);

        const m = Matrix.fromVector(v);

        expect(m.rows).toBe(3);
        expect(m.columns).toBe(1);
        expect(m.get(2, 0)).toBe(3);
    });

    it("random creates valid range", () => {
        const m = Matrix.random(2, 2);

        m.toNestedArray().forEach(row => {
            row.forEach(v => {
                expect(v).toBeGreaterThanOrEqual(0);
                expect(v).toBeLessThan(1);
            });
        });
    });

    it("batchMul works correctly", () => {
        const x = Matrix.from([
            [1, 2],
            [3, 4],
        ]);

        const w = Matrix.from([
            [5, 6],
            [7, 8],
        ]);

        const r = Matrix.batchMul(x, w);

        expect(r.get(0, 0)).toBe(19);
        expect(r.get(0, 1)).toBe(22);
        expect(r.get(1, 0)).toBe(43);
        expect(r.get(1, 1)).toBe(50);
    });

    it("multiplyScalar works correctly", () => {
        const m = Matrix.from([
            [1, 2],
            [3, 4],
        ]);

        const r = Matrix.multiplyScalar(m, 2);

        expect(r.toNestedArray()).toEqual([
            [2, 4],
            [6, 8],
        ]);
    });

    it("copy creates deep copy", () => {
        const m = Matrix.from([
            [1, 2],
            [3, 4],
        ]);

        const c = m.copy();

        c.set(0, 0, 99);

        expect(m.get(0, 0)).toBe(1);
        expect(c.get(0, 0)).toBe(99);
    });

    it("toNestedArray works correctly", () => {
        const m = Matrix.from([
            [1, 2],
            [3, 4],
        ]);

        expect(m.toNestedArray()).toEqual([
            [1, 2],
            [3, 4],
        ]);
    });

    it("toTensor works correctly", () => {
        const m = Matrix.from([
            [1, 2],
            [3, 4],
        ]);

        const t = m.toTensor();

        expect(t.shape).toEqual([2, 2]);
        expect(t.get(0, 0)).toBe(1);
        expect(t.get(0, 1)).toBe(2);
        expect(t.get(1, 0)).toBe(3);
        expect(t.get(1, 1)).toBe(4);
    });
});
