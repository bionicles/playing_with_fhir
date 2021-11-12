"quantify shape and depth diversity of FHIR data"
# conda create -n py39 python=3.9
# conda activate py39
# pip install rich, numpy
# python fhir.py

from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional, Tuple
import json
import os

from rich import print
import numpy as np

Array = np.ndarray

# data from https://synthetichealth.github.io/synthea/
VERSIONS = ("dstu2", "stu3", "r4")
N_PATIENTS = 100


def test_get_paths():
    item1 = {
        "a": "bion",
        "b": {"c": "is", "d": "cool"},
    }  # (("a",), ("b", "c"), ("b", "d"))
    paths1 = get_paths(item1)
    # print(item1, "\n", paths1)
    assert paths1 == (("a",), ("b", "c"), ("b", "d"))
    item2 = {"a": ("bion", "is", "cool")}  # (("a", 0), ("a", 1), ("a", 2))
    paths2 = get_paths(item2)
    # print(item2, "\n", paths2)
    assert paths2 == (("a", 0), ("a", 1), ("a", 2))
    item3 = {"a": {"b": ("bion", "is", "cool")}}
    paths3 = get_paths(item3)
    # print(item3, "\n", paths3)
    assert paths3 == (("a", "b", 0), ("a", "b", 1), ("a", "b", 2))
    assert get_paths("bion") == ((),)
    assert get_paths(b"cool") == ((),)
    assert get_paths(True) == ((),)
    assert get_paths(None) == ((),)
    assert get_paths(1) == ((),)
    shape1 = {"a": "bion", "b": {"c": "is", "d": "cool"}}
    shape2 = {"a": "stuff", "b": {"c": True, "d": False}}
    assert get_paths(shape1) == get_paths(shape2)


def get_paths(
    item: any, path: Tuple[any, ...] = ()
) -> Tuple[Tuple[any, ...]]:  # (()) or ((step0,), (step0, step1) ...)
    """
    given a PyTree / collection, returns a tuple of path tuples, one per leaf.
    leaves are non-collection types (str, int, float, bool, bytes, None)
    get_paths({"a": "bion", "b": {"c": "is", "d": "cool"}}) = (("a",), ("b", "c"), ("b", "d"))
    get_paths({"a": ("bion", "is", "cool")}) = (("a", 0), ("a", 1), ("a", 2))
    {"a": {"b": ("bion", "is", "cool")}} = (("a", "b", 0), ("a", "b", 1), ("a", "b", 2))
    """
    if isinstance(item, (str, int, float, bool, bytes, type(None))):
        return (path,)
    if isinstance(item, dict):
        nested = tuple(get_paths(value, path + (key,)) for key, value in item.items())
        # unnest nested tuples
        return tuple(chain.from_iterable(nested))
    if isinstance(item, (list, tuple)):
        nested = tuple(
            get_paths(value, path + (index,)) for index, value in enumerate(item)
        )
        # unnest nested tuples
        return tuple(chain.from_iterable(nested))
    raise TypeError(f"unsupported type: {type(item)}")


def get_data(folder: str, n_patients: Optional[int] = N_PATIENTS) -> List[dict]:
    "get data from a folder"

    data = []
    for file in os.listdir("./data" + folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file)) as f:
                data.append(json.load(f))
        if n_patients is not None and len(data) == n_patients:
            break
    return data


def group_by_resource_type(bundles: List[dict]) -> Dict[str, List[dict]]:
    "group data by resource type"

    grouped = {}
    for bundle in bundles:
        for entry in bundle["entry"]:
            resource = entry["resource"]
            resource_type = resource["resourceType"]
            if resource_type not in grouped:
                grouped[resource_type] = []
            grouped[resource_type].append(resource)
    return grouped


def get_shapes_and_depths(grouped: dict) -> Tuple[Dict[str, tuple], Dict[str, Array]]:
    "get shapes and leaf depths of resources"
    shapes, depths = {}, {}
    for resource_type, instances in grouped.items():
        if resource_type not in shapes:
            shapes[resource_type] = []
        if resource_type not in depths:
            depths[resource_type] = []
        for instance in instances:
            paths = get_paths(instance)
            leaf_depths = tuple(len(path) for path in paths)
            shapes[resource_type].append(len(paths))
            depths[resource_type].extend(leaf_depths)
    shapes = {k: tuple(set(v)) for k, v in shapes.items()}
    depths = {k: np.array(v) for k, v in depths.items()}
    return shapes, depths


@dataclass(frozen=True)
class VersionStats:
    "statistics for a FHIR version"
    version: str
    n_patients: int
    counts: Dict[str, int]
    depths: Dict[str, Array]
    shapes: Dict[str, tuple]


def get_version_stats(version: str, n_patients: int = N_PATIENTS) -> Dict[str, tuple]:
    "get shapes of resources"
    data = get_data(version, n_patients)
    grouped = group_by_resource_type(data)
    shapes, depths = get_shapes_and_depths(grouped)
    version_stats = VersionStats(
        version=version,
        n_patients=n_patients,
        counts={k: len(v) for k, v in grouped.items()},
        depths=depths,
        shapes=shapes,
    )
    return version_stats


def show_version(stats: VersionStats) -> None:
    "renders a VersionStats to stdout"
    print("FHIR Version {stats.version}")
    print("  n_patients: {stats.n_patients}")
    for key in sorted(stats.counts.keys()):
        print(f"    {key}: ")
        print(f"      count: {stats.counts[key]}")
        print(f"      n_shapes: {len(stats.shapes[key])}")
        print(f"      avg_depth: {stats.depths[key].mean()}")
        print(f"      max_depth: {stats.depths[key].max()}")


@dataclass(frozen=True)
class ResourceTypeStats:
    "statistics for a resource type"
    resource_type: str
    n_patients: int
    counts: Dict[str, int]  # {version: count}
    depths: Dict[str, Array]  # {version: depths}
    shapes: Dict[str, tuple]  # {version: shapes}


# group the resources by type across FHIR versions
def get_resource_stats(
    stats: Dict[str, VersionStats]  # {version: version_stats}
) -> Dict[str, ResourceTypeStats]:  # {resource_type: resource_type_stats}
    grouped = {}
    for version, version_stats in stats.items():
        for resource_type, instances in version_stats.counts.items():
            if resource_type not in grouped:
                grouped[resource_type] = ResourceTypeStats(
                    resource_type=resource_type,
                    n_patients=version_stats.n_patients,
                    counts={},
                    depths={},
                    shapes={},
                )
            grouped[resource_type].counts[version] = instances
            grouped[resource_type].depths[version] = version_stats.depths[resource_type]
            grouped[resource_type].shapes[version] = version_stats.shapes[resource_type]
    return grouped


def show_resource_stats(stats: ResourceTypeStats) -> None:
    "renders a ResourceTypeStats to stdout"
    print(f"Resource Type: {stats.resource_type}")
    print(f"  n_patients: {stats.n_patients}")
    for version in stats.counts.keys():
        print(f"    {version}: ")
        print(f"      count: {stats.counts[version]}")
        print(f"      n_shapes: {len(stats.shapes[version])}")
        print(f"      avg_depth: {stats.depths[version].mean()}")
        print(f"      max_depth: {stats.depths[version].max()}")


if __name__ == "__main__":
    version_stats = {
        version: get_version_stats(version, n_patients=N_PATIENTS)
        for version in VERSIONS
    }
    resource_stats = get_resource_stats(version_stats)
    for resource_type, resource_type_stats in resource_stats.items():
        show_resource_stats(resource_type_stats)
