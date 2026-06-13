import {describe, it, expect} from "vitest";
import {Vector} from "../core/Vector";

describe("Vector", () => {
    it("creates vector with correct length", () => {
        const v = new Vector(5);
        expect(v.length).toBe(5);
    });

    it("set/get works correctly", () => {
        const v = new Vector(3);

        v.set(0, 10);
        v.set(1, 20);
        v.set(2, 30);

        expect(v.get(0)).toBe(10);
        expect(v.get(1)).toBe(20);
        expect(v.get(2)).toBe(30);
    });

    it("sum works correctly", () => {
        const v = Vector.from([1, 2, 3, 4]);
        expect(v.sum()).toBe(10);
    });

    it("avg works correctly", () => {
        const v = Vector.from([2, 4, 6, 8]);
        expect(v.avg()).toBe(5);
    });

    it("avg returns 0 for empty vector", () => {
        const v = new Vector(0);
        expect(v.avg()).toBe(0);
    });

    it("mul throws when size is not zero (assert branch)", () => {
        const v = Vector.from([1, 2, 3]);
        expect(() => v.mul()).toThrow();
    });

    it("mul works when vector is empty", () => {
        const v = new Vector(0);
        expect(v.mul()).toBe(1);
    });

    it("toArray returns correct data", () => {
        const v = Vector.from([1, 2, 3]);
        expect(v.toArray()).toEqual([1, 2, 3]);
    });

    it("from creates correct vector", () => {
        const v = Vector.from([5, 6, 7]);
        expect(v.get(0)).toBe(5);
        expect(v.get(1)).toBe(6);
        expect(v.get(2)).toBe(7);
    });

    it("fromData copies correctly", () => {
        const data = new Float64Array([9, 8, 7]);
        const v = Vector.fromData(data);

        expect(v.get(0)).toBe(9);
        expect(v.get(1)).toBe(8);
        expect(v.get(2)).toBe(7);
    });

    it("random creates correct size", () => {
        const v = Vector.random(4);
        expect(v.length).toBe(4);

        v.toArray().forEach(x => {
            expect(x).toBeGreaterThanOrEqual(0);
            expect(x).toBeLessThan(1);
        });
    });

    it("zeros creates zero vector", () => {
        const v = Vector.zeros(4);

        expect(v.toArray()).toEqual([0, 0, 0, 0]);
    });

    it("addVectors works correctly", () => {
        const v1 = Vector.from([1, 2, 3]);
        const v2 = Vector.from([4, 5, 6]);

        const r = Vector.addVectors(v1, v2);

        expect(r.toArray()).toEqual([5, 7, 9]);
    });

    it("subVectors works correctly", () => {
        const v1 = Vector.from([5, 7, 9]);
        const v2 = Vector.from([1, 2, 3]);

        const r = Vector.subVectors(v1, v2);

        expect(r.toArray()).toEqual([4, 5, 6]);
    });

    it("multiplyScalar works correctly", () => {
        const v = Vector.from([1, 2, 3]);
        const r = Vector.multiplyScalar(v, 2);

        expect(r.toArray()).toEqual([2, 4, 6]);
    });

    it("mulVectors works correctly", () => {
        const v1 = Vector.from([1, 2, 3]);
        const v2 = Vector.from([2, 3, 4]);

        const r = Vector.mulVectors(v1, v2);

        expect(r.length).toBe(3);
    });

    it("toTensor converts correctly", () => {
        const v = Vector.from([1, 2, 3, 4]);
        const t = v.toTensor();

        expect(t.shape).toEqual([1, 4]);
        expect(t.get(0, 0)).toBe(1);
        expect(t.get(0, 1)).toBe(2);
        expect(t.get(0, 2)).toBe(3);
        expect(t.get(0, 3)).toBe(4);
    });
});
