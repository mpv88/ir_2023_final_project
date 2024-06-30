# ir_2023_final_project

## PageRank Algorithm Implementation

- 1 On a dataset of webpages to be decided (main problem is of finding a small and significant dataset).
- 2 Implement the PageRank algorithm[1][2].
- 3 Also implement the ability to get a list of pages and use them to define the jump vector, thus performing topic-specific PageRank.

## References
<a id="1">[1]</a> Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual web search engine. Computer networks and ISDN systems, 30(1-7), 107-117. \
<a id="2">[2]</a> Page, L., Brin, S., Motwani, R., & Winograd, T. (1998). The PageRank citation ranking: Bringing order to the web. In Proceedings of the 7<sup>th</sup> International World Wide Web Conference

## Folder structure
Current folder contains a simple implementation of the PageRank algorithm, together some datasets and some utility functions.

* `requirements.txt` file includes the python libraries to be installed to allow running the code.
* `PageRank.pptx` file includes a brief theoretical recap together with an explanation of what has been implemented in our code.
* `Pagerank_main.ipynb` notebook containing the actual code to be run to replicate in a practical setting what explained in the slides.
* `src` folder includes all the source code, which is made by the `dense_pr.py`, the `sparse_pr.py`, the `ts_pr.py` and the `utils.py` files.
* `data` folder, includes the `input` folder containing the SG and the WV datasets, used in the practical demonstration, and the `output` folder containing some of the plots generated.
* `docs` folder containing the original papers of the above-mentioned main theoretical references for our implementation.

## Execution
Download the repository on your computer.
Run the requirements file to install the dependencies needed to run the project.
Execute the `Pagerank_main.ipynb` notebook following the instructions which contains.