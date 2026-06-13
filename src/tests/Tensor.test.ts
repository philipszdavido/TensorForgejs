import {describe, it, expect} from "vitest";
import {Tensor} from "../core/Tensor";
import {Matrix} from "../core/Matrix";

describe("Tensor", () => {
    it("creates correct size", () => {
        const t = new Tensor([2, 3, 4]);
        expect(t.size()).toBe(24);
    });

    it("computes correct strides", () => {
        const strides = Tensor.computeStrides([2, 3, 4]);
        expect(strides).toEqual([12, 4, 1]);
    });

    it("get/set works correctly for 2D tensor", () => {
        const t = new Tensor([2, 2]);

        t.set(10, 0, 0);
        t.set(20, 0, 1);
        t.set(30, 1, 0);
        t.set(40, 1, 1);

        expect(t.get(0, 0)).toBe(10);
        expect(t.get(0, 1)).toBe(20);
        expect(t.get(1, 0)).toBe(30);
        expect(t.get(1, 1)).toBe(40);
    });

    it("throws on invalid index length", () => {
        const t = new Tensor([2, 2]);
        expect(() => t.get(1)).toThrow();
    });

    it("fill works correctly", () => {
        const t = new Tensor([2, 2]);
        t.fill(7);

        expect(t.get(0, 0)).toBe(7);
        expect(t.get(0, 1)).toBe(7);
        expect(t.get(1, 0)).toBe(7);
        expect(t.get(1, 1)).toBe(7);
    });

    it("fromArray initializes correctly", () => {
        const t = Tensor.fromArray([2, 2], [1, 2, 3, 4]);

        expect(t.get(0, 0)).toBe(1);
        expect(t.get(0, 1)).toBe(2);
        expect(t.get(1, 0)).toBe(3);
        expect(t.get(1, 1)).toBe(4);
    });

    it("clone creates deep copy", () => {
        const t1 = Tensor.fromArray([2, 2], [1, 2, 3, 4]);
        const t2 = t1.clone();

        t2.set(99, 0, 0);

        expect(t1.get(0, 0)).toBe(1);
        expect(t2.get(0, 0)).toBe(99);
    });

    it("reshape preserves data", () => {
        const t = Tensor.fromArray([2, 2], [1, 2, 3, 4]);
        const r = t.reshape([4]);

        expect(r.size()).toBe(4);
        expect(r.get(0)).toBe(1);
        expect(r.get(1)).toBe(2);
        expect(r.get(2)).toBe(3);
        expect(r.get(3)).toBe(4);
    });

    it("reshape throws on invalid size", () => {
        const t = new Tensor([2, 2]);
        expect(() => t.reshape([3])).toThrow();
    });

    it("toMatrix converts correctly", () => {
        const t = Tensor.fromArray([2, 2], [1, 2, 3, 4]);
        const m = t.toMatrix();

        expect(m.get(0, 0)).toBe(1);
        expect(m.get(0, 1)).toBe(2);
        expect(m.get(1, 0)).toBe(3);
        expect(m.get(1, 1)).toBe(4);
    });

    it("fromMatrix converts correctly", () => {
        const m = new Matrix(2, 2);
        m.set(0, 0, 1);
        m.set(0, 1, 2);
        m.set(1, 0, 3);
        m.set(1, 1, 4);

        const t = Tensor.fromMatrix(m);

        expect(t.get(0, 0)).toBe(1);
        expect(t.get(0, 1)).toBe(2);
        expect(t.get(1, 0)).toBe(3);
        expect(t.get(1, 1)).toBe(4);
    });

    it("toVector works for 1D tensor", () => {
        const t = Tensor.fromArray([4], [1, 2, 3, 4]);
        const v = t.toVector();

        expect(v.get(0)).toBe(1);
        expect(v.get(1)).toBe(2);
        expect(v.get(2)).toBe(3);
        expect(v.get(3)).toBe(4);
    });

    it("toVector throws for non-1D tensor", () => {
        const t = new Tensor([2, 2]);
        expect(() => t.toVector()).toThrow();
    });

    it("zeros creates zero-filled tensor", () => {
        const t = Tensor.zeros([2, 2]);

        expect(t.get(0, 0)).toBe(0);
        expect(t.get(1, 1)).toBe(0);
    });

    it("setFloatArray overwrites correctly", () => {
        const t = new Tensor([2, 2]);
        const arr = new Float64Array([5, 6, 7, 8]);

        t.setFloatArray(arr);

        expect(t.get(0, 0)).toBe(5);
        expect(t.get(0, 1)).toBe(6);
        expect(t.get(1, 0)).toBe(7);
        expect(t.get(1, 1)).toBe(8);
    });
});
