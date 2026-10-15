"""Portable JSON exports; original run files remain unchanged."""

import math


def portable_json(value, project):
    if isinstance(value, dict):
        return {portable_json(k, project): portable_json(v, project) for k, v in value.items()}
    if isinstance(value, list):
        return [portable_json(v, project) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        for prefix in {str(project), project.as_posix(), str(project).replace("/", "\\")}:
            value = value.replace(prefix, "<project>")
        if value.startswith("<project>"):
            value = value.replace("\\", "/")
    return value
