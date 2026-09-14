// plotly.js-cartesian-dist-min ships no types of its own; it exposes the same
// module shape as the full plotly.js package (minus gl3d/mapbox trace types,
// which this app never uses — only scatter/bar). @types/plotly.js is a
// types-only devDependency (no runtime code) reused here for typing.
declare module "plotly.js-cartesian-dist-min" {
  import * as Plotly from "plotly.js";
  export = Plotly;
}
