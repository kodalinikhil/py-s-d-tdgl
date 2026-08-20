py-s-d-TDGL: single- and multi-component GL simulations
========================================================

.. image:: images/logo-transparent-large.png
  :width: 300
  :alt: py-s-d-TDGL logo.
  :align: center

.. image:: https://img.shields.io/github/actions/workflow/status/kodalinikhil/py-s-d-tdgl/lint-and-test.yml?branch=main
   :target: https://github.com/kodalinikhil/py-s-d-tdgl/actions
   :alt: GitHub Workflow Status

.. image:: https://img.shields.io/github/license/kodalinikhil/py-s-d-tdgl
   :target: https://github.com/kodalinikhil/py-s-d-tdgl/blob/main/LICENSE
   :alt: GitHub

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

`py-s-d-TDGL <https://github.com/kodalinikhil/py-s-d-tdgl>`_ is an
experimental Python framework for two-dimensional Ginzburg--Landau simulations
of thin-film superconductors. It extends pyTDGL's geometry, finite-volume,
transport, screening, post-processing, and visualization stack to four
order-parameter models:

* :class:`tdgl.SingleBandModel` for standard Kramer--Watts--Tobin TDGL;
* :class:`tdgl.SPlusDModel` for mixed s+d superconductivity;
* :class:`tdgl.DPlusDPrimeModel` for chiral d+d' states; and
* :class:`tdgl.SPlusSModel` for isotropic s+s and s+is systems.

Two backends cover different physical domains. :func:`tdgl.solve` evolves a
finite :class:`tdgl.Device` of arbitrary shape on an unstructured triangular
mesh, with holes, terminals, probes, disorder, and applied drives.
:func:`tdgl.solve_magnetic_periodic` evolves a rectangular vortex-cell torus
on a structured grid, with magnetic translations and an exactly fixed integer
flux sector. See :doc:`framework` for a capability map and :doc:`models` for
the model conventions and limitations.

Start with :doc:`installation`, then use the :doc:`single-band quickstart
<notebooks/quickstart>` or the examples in :doc:`models` and
:doc:`magnetic_periodic`. The :doc:`background` chapter describes the
inherited single-band formulation and shared finite-volume numerics.

.. tip::

   The original pyTDGL finite-device and single-band implementation is
   described in:

     pyTDGL: Time-dependent Ginzburg-Landau in Python,
     Computer Physics Communications **291**, 108799 (2023),
     DOI: `10.1016/j.cpc.2023.108799 <https://doi.org/10.1016/j.cpc.2023.108799>`_.

   The accepted version of the paper can also be found on arXiv: `arXiv:2302.03812 <https://doi.org/10.48550/arXiv.2302.03812>`_.


Attribution and citation
------------------------

If you use this framework in research, cite the original pyTDGL paper for the
inherited numerical framework and the relevant source listed in :doc:`models`
for the selected extended model.

.. code-block::

   % BibTeX citation
   @article{
       Bishop-Van_Horn2023-wr,
       title    = "{pyTDGL}: Time-dependent {Ginzburg-Landau} in Python",
       author   = "Bishop-Van Horn, Logan",
       journal  = "Comput. Phys. Commun.",
       volume   =  291,
       pages    = "108799",
       month    =  may,
       year     =  2023,
       url      = "http://dx.doi.org/10.1016/j.cpc.2023.108799",
       issn     = "0010-4655",
       doi      = "10.1016/j.cpc.2023.108799"
   }

The inherited pyTDGL implementation also adapts work from
`SuperDetectorPy <https://github.com/afsa/super-detector-py>`_ and
`SuperScreen <https://superscreen.readthedocs.io/en/latest/>`_. See the
:doc:`background` and :doc:`about/references` pages for the underlying
numerical references.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation.rst
   framework.rst
   models.rst
   notebooks/quickstart.ipynb
   background.rst
   magnetic_periodic.rst

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/screening.ipynb
   notebooks/polygons.ipynb
   notebooks/mesh.ipynb
   notebooks/py-mesh.ipynb
   notebooks/logo.ipynb


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/device.rst
   api/solver.rst
   api/solution.rst
   api/magnetic-periodic.rst
   api/finite-volume.rst
   api/visualization.rst

.. toctree::
   :maxdepth: 2
   :caption: About py-s-d-TDGL

   about/changelog.rst
   about/contributing.rst
   about/license.rst
   about/references.rst

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
