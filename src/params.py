from dataclasses import dataclass

import yaml
from marshmallow_dataclass import class_schema


@dataclass
class Target:
    latitude: float
    longitude: float
    altitude: float


@dataclass
class InitialPosition:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    julian_day: float


@dataclass
class Params:
    earth_target: Target
    satellite_initial_position_eci: InitialPosition


ParamsSchema = class_schema(Params)


def read_params(path: str) -> Params:
    with open(path, "r") as input_stream:
        schema = ParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
