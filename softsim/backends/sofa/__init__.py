"""SOFA FEM backend (requires a SOFA binary release with SofaPython3; set ``SOFA_ROOT``).

The capability declaration is importable without SOFA, so ``softsim check``
can report what SOFA supports even where it is not installed.
"""

from softsim.backends.sofa.backend import SofaBackend, force_field_args, lame, rigid_mass_properties

__all__ = ["SofaBackend", "force_field_args", "lame", "rigid_mass_properties"]
