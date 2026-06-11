export default function randomNormal(): number {

    const u1 = Math.random();
    const u2 = Math.random();

    return Math.sqrt(
        -2 * Math.log(u1)
    ) * Math.cos(
        2 * Math.PI * u2
    );
}
