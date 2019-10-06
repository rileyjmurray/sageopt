.. _MCW2019: https://arxiv.org/abs/1907.00814

.. _MCW2018: https://arxiv.org/abs/1810.01614

.. _SDSOS: https://arxiv.org/abs/1706.02586


Background
==========

This page gives some brief [historical] background on the mathematics implemented by sageopt. Later on, we might add
some more technical mathematical background here as well.

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

The appendix of MCW2018_ describes a python package called "sigpy", which implements SAGE relaxations
for both signomial and polynomial optimization problems. Sageopt supercedes sigpy by implementing a significant
generalization of the original SAGE certificates for both signomials and polynomials. The formal name for these
generalizations are *conditional SAGE signomials* and *conditional SAGE polynomials*. This concept is introduced in a
2019 paper by Murray, Chandrasekaran, and Wierman, titled `Signomial and Polynomial Optimization via Relative Entropy
and Partial Dualization <https://arxiv.org/abs/1907.00814>`_.
Because the generalization follows so transparently from the original idea of SAGE certificates, we say simply say
"SAGE certificates" in reference to the most general idea.
