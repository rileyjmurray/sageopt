.. _gpkit:

========================
Using sageopt with GPKit
========================

GPKit is a python package which makes it easy for engineers to use signomial and geometric
programming models in their design workflow. Sageopt provides basic functionality to
interact with GPKit.
At present, this functionality is contained in a single function:
``sageopt.interop.gpkit.gpkit_model_to_sageopt_model``.
That function accepts a GPKit Model object, and returns a dict of the form::

    so_mod = {
        'vkmap': vkmap,
        'f': f,
        'gp_eqs': gp_eqs,
        'sp_eqs': sp_eqs,
        'gp_gts': gp_gts,
        'sp_gts': sp_gts
    }

To interpret this dict it is important to consider a few ways that sageopt differs from GPKit.

 * Sageopt does not have natively have "Variable" objects for building signomial programs. The
   closest sageopt approximation is :func:`sageopt.standard_sig_monomials`, which takes a
   parameter :math:`n`, and returns a length-:math:`n` array ``y`` of Signomial objects, where
   ``y[i]`` represents the signomial ``y[i](x) = exp(x[i])``.

 * Sageopt works with a single vectorized decision variable, while GPKit allows users to declare
   many named variables of different shapes for use in the same model. The return value
   ``so_mod['vkmap']`` helps reconcile this difference. Specifically, ``vkmap`` is a dict which maps
   a GPKit VarKey object into a numpy array of indices where the GPKit Variable occurs in sageopt's
   implicit vectorized model.

 * GPKit models are stated with "geometric form" signomials -- i.e. expressions like
   :math:`y_1 \sqrt{y_2} - y_3^{-2/3} + y_4^2 - y_2`, while sageopt refers to signomials in exponential form.
   It is easy to move back and forth between these two conventions; if you use sageopt to find a
   solution vector ``x`` to a signomial program, then you can map that into a format GPKit expects
   by working with ``y = np.exp(x)``.

For the remaining key-value pairs in ``so_mod``:

 * :math:`f` is the objective function to minimize.
 * :math:`\phi \in \mathtt{gp{\_}eqs}` represents a convex constraint
   :math:`\phi(x) = 0`. These constraint functions have exactly one positive term and
   exactly one negative term.
 * :math:`\phi \in \mathtt{sp{\_}eqs}` represents a nonconvex constraint
   :math:`\phi(x) = 0`.
 * :math:`g \in \mathtt{gp{\_}gts}` represents a convex constraint
   :math:`g(x) \geq 0`. Each function :math:`g` has exactly one positive term,
   with remaining terms being negative.
 * :math:`g \in \mathtt{gp{\_}gts}` represents a constraint
   :math:`g(x) \geq 0` which doesn't fall into the above category.

The prefix ``gp`` indicates compatibility with geometric programming, and the prefix ``sp`` indicates
that general signomial programming is required.
The suffix ``eqs`` refers to equality constraints, while ``gts`` refers to constraint functions
which must be greater than or equal to zero.

The :ref:`Examples page <gpkit_ex>` demonstrates how to use this data in an existing
GPKit workflow.
