# Demos

This repo holds all the jupyter notebook demos for the examples tab of Ivy's web. Relevant links:
- [List of open tasks avilable](https://github.com/unifyai/demos/issues/2) 
- [Open discussion to suggest new examples](https://github.com/unifyai/demos/issues/1)

All of the examples should be as comprehensible as possible, using easy-to-follow and attractive visuals (graphs, nicely formatted results, etc.), and follow the general tone of Ivy (feel free to include emojis and don't keep things super serious!).

Given that an internal release of the graph compiler is around the corner anybody should be able to start working on these examples shortly, so don't worry about not having access to the graph compiler / transpiler code for now, you can start to work on the notebook style/story!

If anyone has any question feel free to ping me (Guillermo) or use the Community/UX team discord channel!

## Creating a Notebook for Demo

To ensure that similar formats are used across the demo notebooks, a template is created to help you get started! It is located in [assets/01_template.ipynb](assets/01_template.ipynb)! Please make a copy of it to start creating a demo!

1. Firstly, please update the file name to be match the topic of your demo. Then, place the notebook in its relevant folder.

2. Next, please edit the title and description accordingly to ensure that they are rendered correctly in the webpage.

3. All contents should start behind the existing template cells, where:
- The h2 (##) tags are used for section titles.
- The h3 (###) tags are used for subsection titles.
- All steps and explanation should go with the default, which is text or paragraph (p) without any tags.

4. Lastly, you have to include the new notebook path in the corresponding position in the listings of the files `_quarto.yml` and `index.qmd`.