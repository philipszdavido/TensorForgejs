import {describe, it, expect} from "vitest";
import {Matrix} from "../core/Matrix";

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

        expect(() => m.get(10, 10)).toThrow();
    });
});
