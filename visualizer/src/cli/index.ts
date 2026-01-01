import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { program } from "commander";
import { generate } from "../visualization/visualizer.js";

const readJson = (path: string) => JSON.parse(readFileSync(path, "utf-8"));

const ensureDir = (path: string) => {
    const dir = dirname(path);
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
};

program
    .name("visualizer")
    .description("Generate HTML visualizations from simulation data")
    .version("1.0.0");

program
    .command("generate")
    .description("Generate HTML visualization from JSON")
    .argument("<input>", "Input JSON file")
    .option("-o, --output <path>", "Output HTML file", "visualization.html")
    .option("-t, --title <string>", "Page title", "Simulation Visualization")
    .action((input: string, options) => {
        const inputPath = resolve(input);
        const outputPath = resolve(options.output);

        console.log(`Reading: ${inputPath}`);
        const html = generate(readJson(inputPath), options.title);

        ensureDir(outputPath);
        writeFileSync(outputPath, html, "utf-8");
        console.log(`Generated: ${outputPath}`);
    });

program.parse();
