<div>
    <h2 align="center">M2R Project</h2>
    <h3>Contributors:</h3>
    <ul>
        <li>Michael Williams</li>
        <li>Kane Rowley</li>
        <li>Dan Haywood</li>
    </ul>
</div>

## About the Project

This is the repository containing code used for our 2nd year research project.
This repo contains code used to generate figures for the report / presentation, as well as
code that computes numerical solutions to the problem posed in our project.
All code contained in this repo was written by Dan Haywood and Michael Williams.

We make use of the `mayavi` and `matplotlib` packages to plot results and figures,
and the `numpy` and `scipy` packages for general calculation purposes.

## Usage

To install the Python package, navigate to the repository folder in `cmd` and run:

`python -m pip install -e .`

Once installed, you can generate visualisations through the command line by using following scripts:
<ul>
    <li><code>run-bem-visuals</code>: Generate visuals based on the indirect BEM algorithm.</li>
    <li><code>run-general-visuals</code>: Generate useful visuals for the project.</li>
    <li><code>test-bem-visuals</code>: Test the accuracy of the indirect BEM algorithm.</li>
</ul>
