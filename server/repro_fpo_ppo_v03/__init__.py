"""Server-owned v0.3 production helpers.

The package intentionally avoids importing command modules eagerly.  This
keeps ``python -m server.repro_fpo_ppo_v03.asset_binding`` single-loaded and
prevents ``runpy`` from executing a module that was already imported while
initialising its package.
"""
