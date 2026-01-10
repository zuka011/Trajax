import bodyHtml from "./body.html";

type TemplateOptions = {
    title: string;
    styles: string;
    data: string;
    script: string;
};

export const visualizerTemplate = (options: TemplateOptions) =>
    template(options, `<script type="module">\n${options.script}\n</script>`);

const template = (options: TemplateOptions, scriptTag: string) => `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${options.title}</title>
    <style>
    ${options.styles}
    </style>
  </head>
  <body>
    ${bodyHtml}
    <script>
    window.SIMULATION_DATA = ${options.data};
    </script>
    ${scriptTag}
  </body>
</html>`;
