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

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from rich import print
import numpy as np

Array = np.ndarray

# data from https://synthetichealth.github.io/synthea/
VERSIONS = ("dstu2", "stu3", "r4")
N_PATIENTS = 200


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
    path = os.path.join("data", folder)
    for file in os.listdir(path):
        if file.endswith(".json"):
            filepath = os.path.join("data", folder, file)
            with open(filepath) as f:
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
    counts: Dict[str, int]  # {resource_type: count}
    depths: Dict[str, Array]  # {resource_type: leaf_depths}
    shapes: Dict[str, tuple]  # {resource_type: (shape, ...)}


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


def get_resource_stats(
    stats: Dict[str, VersionStats]  # {version: version_stats}
) -> Dict[str, ResourceTypeStats]:  # {resource_type: resource_type_stats}
    "group the resources by type across FHIR versions"
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
    for version in stats.counts.keys():
        print(f"    {version}: ")
        print(f"      count: {stats.counts[version]}")
        print(f"      n_shapes: {len(stats.shapes[version])}")
        print(f"      avg_depth: {stats.depths[version].mean()}")
        print(f"      max_depth: {stats.depths[version].max()}")


def plot_lines_and_violins(
    all_version_stats: Dict[str, VersionStats],  # {version: version_stats}
    all_resource_type_stats: Dict[
        str, ResourceTypeStats
    ],  # {resource_type: resource_type_stats}
) -> go.Figure:
    """
    Plots 2 subfigures in 1 column
    top row: a (version, n_shapes) line per resource type
    top row: a (version, n_shapes) violin per version
    bottom row: a (version, depths) violin per version
    """
    # make a figure with 2 rows and 1 column
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("", ""),
    )
    # label the figure
    fig.update_layout(
        title_text="Resource Polymorphism & Nesting Of FHIR Versions",
        xaxis_title="--- FHIR Version ---> ",
        yaxis_title="Count",
        width=1000,
        height=800,
    )
    # make the top row
    # add a (version, n_shapes) line per resource_type
    for resource_type, resource_type_stats in all_resource_type_stats.items():
        fig.add_trace(
            go.Scatter(
                x=list(resource_type_stats.counts.keys()),
                y=list(
                    len(resource_type_stats.shapes[version])
                    for version in resource_type_stats.counts.keys()
                ),
                mode="lines",
                name=resource_type,
            ),
            row=1,
            col=1,
        )
    # label the top row
    fig.update_yaxes(
        title_text="Polymorphism / # Unique Shapes (lower is better)", row=1, col=1
    )
    # add a (version, n_shapes) violin per version
    colors = {"dstu2": "red", "stu3": "green", "r4": "blue"}
    for version, version_stats in all_version_stats.items():
        # group by version
        n_shapes = list(map(len, version_stats.shapes.values()))
        fig.add_trace(
            go.Violin(
                x=[version] * len(n_shapes),
                y=n_shapes,
                name=version,
                marker_color=colors[version],
                showlegend=False,
                legendgroup=version,
            ),
            row=1,
            col=1,
        )
    # make the bottom row
    # add a (version, depths) violin per version
    for version, version_stats in all_version_stats.items():
        # group all the depths for all the resource_types of this version
        depths = np.concatenate(
            [
                version_stats.depths[resource_type]
                for resource_type in version_stats.counts.keys()
            ]
        )
        fig.add_trace(
            go.Violin(
                x=[version] * len(depths),
                y=depths,
                name=version,
                marker_color=colors[version],
                legendgroup=version,
            ),
            row=2,
            col=1,
        )
    # label the bottom row
    fig.update_yaxes(title_text="Nesting / Leaf Depth (lower is better)", row=2, col=1)
    return fig


def plot_bars(
    all_resource_type_stats: Dict[
        str, ResourceTypeStats
    ],  # {resource_type: resource_type_stats}
) -> go.Figure:
    """
    Plots 2 subfigures in 1 column
    top row: a (resource_type, n_shapes) bar group per resource_type, one bar per version
    bottom row: a (resource_type, depths) box plot group per resource_type, one box per version
    """
    # make a figure with 2 rows and 1 column
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("", ""),
    )
    # label the figure
    fig.update_layout(
        title_text="Polymorphism & Nesting Of FHIR Resource Types",
        xaxis_title="Resource Type",
        yaxis_title="Count",
        barmode="group",
        width=1000,
        height=800,
    )
    colors = {"dstu2": "red", "stu3": "green", "r4": "blue"}
    # make the top row
    # add a (resource_type, n_shapes) bar group per resource_type, one bar per version
    for version in colors.keys():
        x = [
            resource_type
            for resource_type in all_resource_type_stats.keys()
            if version in all_resource_type_stats[resource_type].counts
        ]
        y = [
            len(resource_type_stats.shapes[version])
            for resource_type_stats in all_resource_type_stats.values()
            if version in resource_type_stats.counts
        ]
        fig.add_trace(
            go.Bar(
                name=version,
                x=x,
                y=y,
                marker_color=colors[version],
                legendgroup=version,
            ),
            row=1,
            col=1,
        )
    # label the top row
    fig.update_yaxes(
        title_text="Polymorphism / # Unique Shapes (lower is better)", row=1, col=1
    )
    # add a (resource_type, depths) bar group per resource_type, one box per version
    for version in colors.keys():
        x = [
            resource_type
            for resource_type in all_resource_type_stats.keys()
            if version in all_resource_type_stats[resource_type].counts
        ]
        y = [
            resource_type_stats.depths[version].mean()
            for resource_type_stats in all_resource_type_stats.values()
            if version in resource_type_stats.counts
        ]
        fig.add_trace(
            go.Bar(
                name=version,
                x=x,
                y=y,
                marker_color=colors[version],
                legendgroup=version,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    # label the bottom row
    fig.update_yaxes(
        title_text="Average Nesting / Leaf Depth (lower is better)", row=2, col=1
    )
    return fig


def find_worst_offenders(
    all_resource_type_stats: Dict[str, ResourceTypeStats],
    version: str,
) -> Dict[str, ResourceTypeStats]:
    """
    Finds the resource types with the worst polymorphing and nesting
    """
    # find the resource type with the most number of shapes
    most_polymorphic_resource_type = None
    deepest_resource_type_by_mean = None
    deepest_resource_type_by_max = None
    for resource_type, resource_type_stats in all_resource_type_stats.items():
        if version not in resource_type_stats.counts:
            continue
        shapes = resource_type_stats.shapes[version]
        depths = resource_type_stats.depths[version]
        if most_polymorphic_resource_type is None or len(shapes) > len(
            all_resource_type_stats[most_polymorphic_resource_type].shapes[version]
        ):
            most_polymorphic_resource_type = resource_type
        if (
            deepest_resource_type_by_mean is None
            or depths.mean()
            > all_resource_type_stats[deepest_resource_type_by_mean]
            .depths[version]
            .mean()
        ):
            deepest_resource_type_by_mean = resource_type
        if (
            deepest_resource_type_by_max is None
            or depths.max()
            > all_resource_type_stats[deepest_resource_type_by_max]
            .depths[version]
            .max()
        ):
            deepest_resource_type_by_max = resource_type
    return {
        "version": version,
        "most_polymorphic": all_resource_type_stats[most_polymorphic_resource_type],
        "deepest_by_mean": all_resource_type_stats[deepest_resource_type_by_mean],
        "deepest_by_max": all_resource_type_stats[deepest_resource_type_by_max],
    }


# - the resource type with the most inconsistent data:
# ImagingStudy with 177 different shapes in a sample of 977 ImagingStudy instances

# - the resource type with most deeply nested data (on average):
# ImagingStudy, which requires an average of 5.3 operations to access each leaf

# - the resource type with most deeply nested data (worst case):
# ExplanationOfBenefit has a leaf which requires 8 operations to access
def show_worst_offenders(worst_offenders: dict) -> None:
    version = worst_offenders["version"]
    print(f"\nworst offenders in FHIR {version}\n")
    most_polymorphic = worst_offenders["most_polymorphic"]
    resource_type = most_polymorphic.resource_type
    n_shapes = len(most_polymorphic.shapes[version])
    count = most_polymorphic.counts[version]
    print("the resource type with the most inconsistent data:")
    print(
        f"{resource_type}, with {n_shapes} unique shapes in a sample of {count} {resource_type} instances"
    )
    print()
    deepest_by_mean = worst_offenders["deepest_by_mean"]
    resource_type = deepest_by_mean.resource_type
    mean_depth = deepest_by_mean.depths[version].mean()
    print("the resource type with most deeply nested data (on average):")
    print(
        f"{resource_type}, which requires an average of {mean_depth} operations to access each leaf"
    )
    print()
    deepest_by_max = worst_offenders["deepest_by_max"]
    resource_type = deepest_by_max.resource_type
    max_depth = deepest_by_max.depths[version].max()
    print("the resource type with most deeply nested data (worst case):")
    print(
        f"{resource_type}, which has a leaf which requires {max_depth} operations to access"
    )


if __name__ == "__main__":
    version_stats = {
        version: get_version_stats(version, n_patients=N_PATIENTS)
        for version in VERSIONS
    }
    resource_stats = get_resource_stats(version_stats)

    # to make output.txt, uncomment this and run `python fhir.py > output.txt`
    for resource_type, resource_type_stats in resource_stats.items():
        show_resource_stats(resource_type_stats)

    # to make worst.txt, uncomment this and run `python fhir.py > worst.txt`
    worst_offenders = find_worst_offenders(resource_stats, "r4")
    show_worst_offenders(worst_offenders)

    # to make plots, uncomment this and run `python fhir.py`
    # warning: violin plots are slow if you have a lot of data
    lines_and_violins = plot_lines_and_violins(version_stats, resource_stats)
    lines_and_violins.show()
    # lines_and_violins.write_image("by_fhir_version.png")
    bars = plot_bars(resource_stats)
    bars.show()
    # bars.write_image("by_resource_type.png")