"""ComfyUI-H3LookSheets — look/outfit reference sheets from MiniMax-H3 ref2va.

Self-contained: no imports outside the standard library and what ComfyUI
already provides.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
