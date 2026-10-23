export function tileBase(tile) {
  return ({E: "1z", S: "2z", W: "3z", N: "4z", P: "5z",
    F: "6z", C: "7z"})[tile] || tile?.replace(/^0/, "5").replace(/r$/, "");
}

export const baseTiles = Array.from({length: 34}, (_, tile) =>
  `${tile % 9 + 1}${"mpsz"[Math.floor(tile / 9)]}`);
