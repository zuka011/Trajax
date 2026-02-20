import * as esbuild from "esbuild";

const args = process.argv[2];

const browser = {
    entryPoints: ["src/browser/index.tsx"],
    bundle: true,
    format: "iife",
    outfile: "src/visualization/browser.bundle.js",
    jsx: "automatic",
    jsxImportSource: "preact",
};

const cli = {
    entryPoints: ["src/cli/index.ts"],
    bundle: true,
    platform: "node",
    format: "cjs",
    loader: {
        ".html": "text",
        ".css": "text",
        ".bundle.js": "text",
    },
    outfile: "../faran_visualizer/assets/cli.js",
    banner: {
        js: "#!/usr/bin/env node",
    },
};

switch (args) {
    case "browser":
        await esbuild.build(browser);
        break;
    case "cli":
        await esbuild.build(cli);
        break;
    default:
        await esbuild.build(browser);
        await esbuild.build(cli);
}
