from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("blackwood")
except PackageNotFoundError:
    __version__ = "unknown"
