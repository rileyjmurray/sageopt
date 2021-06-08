.. _MCW2019: https://arxiv.org/abs/1907.00814

.. _MCW2018: https://arxiv.org/abs/1810.01614

.. _SDSOS: https://arxiv.org/abs/1706.02586

References
==========

If you use sageopt in your work, please cite the following paper:

@article{MCW-2019,
  doi = {10.1007/s12532-020-00193-4},
  url = {https://doi.org/10.1007/s12532-020-00193-4},
  year = {2020},
  month = oct,
  publisher = {Springer Science and Business Media {LLC}},
  author = {R.~Murray and V.~Chandrasekaran and A.~Wierman},
  title = {Signomial and polynomial optimization via relative entropy and partial dualization},
  journal = {Mathematical Programming Computation}
}

If you use the "SAGE polynomial" functionality within sageopt, please cite this paper as well:

@article{MCW-2018,
  doi = {10.1007/s10208-021-09497-w},
  url = {https://doi.org/10.1007/s10208-021-09497-w},
  year = {2021},
  month = mar,
  publisher = {Springer Science and Business Media {LLC}},
  author = {R.~Murray and V.~Chandrasekaran and A.~Wierman},
  title = {Newton Polytopes and Relative Entropy Optimization},
  journal = {Foundations of Computational Mathematics},
}

Success stories
===============

Here is a paper that used sageopt (particularly, conditional SAGE signomials) to solve high-degree
polynomial optimization problems arising in power systems engineering.


@misc{WKDC-2020,
    Author = {Lukas Wachter and Orcun Karaca and Georgios Darivianakis and Themistoklis Charalambous},
    Title = {A convex relaxation approach for the optimized pulse pattern problem},
    Year = {2020},
    Eprint = {arXiv:2010.14853}
}



Historical Background
=====================

SAGE certificates were originally developed for signomials
(`Chandrasekaran and Shah, 2016 <https://arxiv.org/abs/1409.7640>`_).

Subsequent work by Murray, Chandrasekaran, and Wierman (MCW2018_) described how
these methods could be adapted to polynomial optimization, and they referred to their proof system as "SAGE
polynomials".  These ideas are especially well-suited to sparse polynomials or polynomials of high degree. The bounds
returned are always at least as strong as those computed by SDSOS_, and can be
stronger. The expressive power of SAGE polyomials is equivalent to that of "SONC polynomials," however there is no
known method to efficiently optimize over the set of SONC polynomials without using the SAGE proof system. For
example, the `2018 algorithm <https://arxiv.org/abs/1808.08431>`_ for computing SONC certificates is efficient, but
is also known to have simple cases where it fails to find SONC certificates that do in fact exist.

Sageopt also implements a significant generalization of the original SAGE certificates for both signomials and
polynomials. The formal name for these
generalizations are *conditional SAGE signomials* and *conditional SAGE polynomials*. This concept is introduced in a
2019 paper by Murray, Chandrasekaran, and Wierman, titled `Signomial and Polynomial Optimization via Relative Entropy
and Partial Dualization <https://arxiv.org/abs/1907.00814>`_.
Because the generalization follows so transparently from the original idea of SAGE certificates, we say simply say
"SAGE certificates" in reference to the most general idea.
