# How to Contribute

## Preparing the coding environment

The first step to start coding new features or correcting bugs in the `cucaracha` library is doing the repository fork, directly on GitHub, and following to the repository clone:

```bash
git clone git@github.com:<YOUR_USERNAME>/cucaracha.git
```

Where `<YOUR_USERNAME>` indicates your GitHub account that has the repository fork.

!!! tip
    See more details on [GitHub](https://docs.github.com/pt/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) for forking a repository

After the repository been set in your local machine, the following setup steps can be done to prepare the coding environment:

!!! warning
    We assume the Poetry tool for project management, then make sure that the Poetry version is 1.8 or above. See more information about [Poetry installation](https://python-poetry.org/docs/#installing-with-pipx)

```bash
cd asltk
poetry shell && poetry install
```

Then all the dependencies will be installed and the virtual environment will be created. After all being done successfully, the shortcuts for `test` and `doc` can be called:

```bash
task test
```

```bash
task doc
```

More details about the entire project configuration is provided in the `pyproject.toml` file.

### Basic tools

We assume the following list of developing, testing and documentation tools:

1. blue
2. isort
3. numpy
4. OpenCV
5. PyMuPDF
6. rich
7. pytest
8. taskipy
9. mkdocs-material
10. pymdown-extensions

Further adjustments in the set of tools for the project can be modified in the future. However, the details about these modifications are directly reported in new releases, regarding the specific tool versioning (more details at Version Control section)

## Code Structure

The general structure of the `cucaracha` library is given as the following:

``` mermaid
classDiagram
  class Document{
    +string doc_path
    +dict metadata
  }
  class Aligment{
    +function inplane_deskew
  }
  class Noise_Removal{
    +function sparse_dots
  }
  class Threshold{
    +function otsu
    +function binary_threshold
  }
```

Where the `Documen` class informs the basic data structure for the document file representation. All the others files are Python modules that contains the image processing methods represented by unique functions.

!!! note
    The general structure to be followed to create an image processing method is using the pattern: i) input = numpy array, ii) output = a tuple with the first item as a numpy array (data output) and the second item as a dictionary informing any additional output parameter that the function may offer.


!!! question
    In case of any doubt, discuss with the community using a [issue card](https://github.com/acsenrafilho/cucaracha/issues) in the repo.

## Testing

Another coding pattern expected in new contributions in the `cucaracha` library is the uses of unit tests. 

!!! info
    A good way to implement test together with coding steps is using a Test-Driven Desing (TDD). Further details can be found at [TDD tutorial](https://codefellows.github.io/sea-python-401d2/lectures/tdd_with_pytest.html) and in many other soruces on internet

Each module or class implemented in the `cucaracha` library should have a series of tests to ensure the quality of the coding and more stability for production usage. We adopted the Python `codecov` tool to help in collecting the code coverage status, which can be accessed by the HTML page that is generated on the call

```bash
task test
```

## Code Documentation

The coding documentation pattern is the [Google Docstring](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

Please, provide as much details as possible in the methods, classes and modules implemented in the `cucaracha` library. By the way, if one may want to get deeper in the explanation of some functionality, then use the documentation webpage itself, which can be easier to add figures, graphs, diagrams and much more simple to read.

!!! tip
    As a good form to assist further users is providing `Examples` in the Google Docstring format. Then, when it is possible, add a few examples in the code documentation. 

!!! info
    The docstring also passes to a test analysis, then take care about adding `Examples` in the docstring, respecting the same usage pattern for input/output as the code provides

## Version Control

The `cucaracha` project adopts the [Semantic Versioning 2.0.0 SemVer](https://semver.org/) versioning pattern. Please, also take care about the specific version changes that will be added by further implementations.

Another important consideration is that the `cucaracha` repository has two permanent branches: `main` and `develop`. The `main` branch is placed to stable, versioning controled releases, and the `develop` branch is for unstable most up-to-date functionalities. In order to keep the library as more reliable as possible, please consider making a Pull Request (PR) at the `develop` branch before passing it to the `main` branch.

!!! info
    The `main` branch is marked by the repository `tag` using the standard `vM.m.p`, where `M` is a major update, `m` minor update and `p` a patch update. All based on SemVer pattern.


## Extending the library

### Extending core functionalities

If you want to provide a new functionality in the `cucaracha`, e.g. a new class that supports a novel ASL processing method, please keep the same data and coding structure as described in the `Code Structure` section.

Any new ideas to improve the project readbility and coding organization is also welcome. If it is the case, please raise a new issue ticket at GitHub, using the Feature option to open an community debate about your suggestion. Once it is approved, a new project version is release with the new implementations glued in the core code.

### Scripts

A easier and less burocratic way to provide new code in the project is using a Python script. In this way, a simple calling script can be added in the repository, under the `scripts` folder, that can be used directly using the python command:

```bash
python -m cucaracha.scripts.YOUR_SCRIPT [input options]
```

In this way, you can share a code that can be called for a specific execution and can be used as a command-line interface (CLI). There are some examples already implemented in the `cucaracha.scripts`, and you can use then to get a general idea on how to apply it.

!!! tip
    Feel free to get inspired adding new scripts in the `cucaracha` project. A quick way to get this is simply making a copy of an existing python script and making your specific modifications.

!!! info
    We adopted the general Python `Argparse` scripting module to create a standarized code. More details on how to use it can be found at the [official documentation](https://docs.python.org/3/library/argparse.html)