import argparse
import enum
import os
from pathlib import Path


# With thanks to Tim for this stack overflow answer: https://stackoverflow.com/a/60750535
class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)

class Processor(enum.Enum):
    

def existing_dir_path(s: str) -> Path:
    if os.path.isdir(s):
        return Path(s)
    else:
        raise NotADirectoryError(s)

def preprocess(source: Path, dest: Path):
    pass


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('processor')
    arg_parser.add_argument('source',
                            'directory containing the input dataset, in the original g2net directory structure',
                            type=existing_dir_path)
    arg_parser.add_argument('dest',
                            'directory for the output dataset, in the original g2net directory structure',
                            type=Path)
    args = arg_parser.parse_args()
    preprocess(args.source, args.dest)
