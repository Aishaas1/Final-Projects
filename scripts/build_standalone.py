"""Compatibility entry point: regenerate the final notebook and both old names."""
try:
    from .build_final_notebook import build
except ImportError:
    from build_final_notebook import build

if __name__ == "__main__":
    build()
