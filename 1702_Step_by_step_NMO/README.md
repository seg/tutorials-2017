# Geophysical tutorial: Step-by-step NMO correction

by [Leonardo Uieda](http://www.leouieda.com)

*This is a part of The Leading Edge [tutorials
series](https://dx.doi.org/10.1190/tle35020190.1).*

The manuscript was written in [Authorea](https://www.authorea.com).
You can view and comment on the text at https://www.authorea.com/users/1856/articles/142722/_show_article

Open any text book about seismic data processing and you will inevitably find a
section about the Normal Moveout (NMO) correction.
When applied to a Common Mid Point (CMP) section, the correction is supposed to
turn the hyperbola associated with a reflection into a straight horizontal
line.
What most text books won't tell you is *how, exactly, do you apply this
equation to the data*?

That is what this tutorial will teach you (hopefully).

<img src="figures/nmo-application/nmo-application.png" width="800px">

## Running the code

The code that accompanies the tutorial is in the `step-by-step-nmo.ipynb`
[Jupyter notebook](http://jupyter.org/).

You can run it in our own machine by following these steps:

1. Make sure you have Python 3.5 with the latest versions of numpy, scipy,
   matplotlib, and jupyter installed. The easiest way to do that is by
   installing the [Anaconda distribution](https://www.continuum.io/downloads#all).
2. Download a copy of this repository ([click here to get a zip
   file](https://github.com/pinga-lab/nmo-tutorial/archive/master.zip)). Unzip
   the repository, preferably in your Desktop folder.
3. Open a terminal (or `cmd.exe` in Windows) and start the Jupyter notebook
   server by running the command `jupyter notebook`.
4. In the Jupyter web interface, navigate to the repository and click to open
   the notebook.
5. Rejoice!

![](http://i.giphy.com/WIg8P0VNpgH8Q.gif)

## License

All text and figures are licensed under a
[CC-BY](http://creativecommons.org/licenses/by/4.0/deed.en_US).
This means that you can use, copy, modify, and redistribute it provided you
give attribution to the original authors.

All source code is licensed under a [BSD
3-clause](https://opensource.org/licenses/BSD-3-Clause) license.
This means that you can do whatever you want with the code provided you give
attribution to the original authors and that you cannot blame the authors if
something bad happens as a consequence of the code.
