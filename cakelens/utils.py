import typing


def make_repr_attrs(items: typing.Sequence[typing.Tuple[str, typing.Any]]) -> str:
    return " ".join(map(lambda item: "=".join([item[0], str(item[1])]), items))


def format_percentage_values(values: list[float]) -> str:
    values_str = ", ".join(map(lambda v: f"{v * 100:.2f}%", values))
    return f"[{values_str}]"
