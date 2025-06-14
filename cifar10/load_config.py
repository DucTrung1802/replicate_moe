from typing import Dict, Any
import argparse
import json
import supported


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON and override with command-line arguments if provided.

    Args:
        config_path (str): Path to the config.json file.

    Returns:
        Dict[str, Any]: Final merged configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(description="Unified config loader")

    for key, val in config.items():
        help_text = f"(default: {val})"

        if isinstance(val, bool):
            # Boolean flags: if True in config, allow disabling with --no-key
            parser.add_argument(
                f"--{key}", dest=key, default=val, action="store_true", help=help_text
            )
            parser.add_argument(
                f"--no-{key}",
                dest=key,
                default=val,
                action="store_false",
                help=help_text,
            )
            parser.set_defaults(**{key: val})

        elif key == "model":
            parser.add_argument(
                f"--{key}",
                type=str,
                choices=supported.model,
                default=val,
                help=help_text,
            )

        elif key == "strategy":
            parser.add_argument(
                f"--{key}",
                type=str,
                choices=supported.strategy,
                default=val,
                help=help_text,
            )

        elif val is None:
            # Handle None values (like wandb_name) safely by defaulting to str
            parser.add_argument(f"--{key}", type=str, default=None, help=help_text)

        else:
            parser.add_argument(f"--{key}", type=type(val), default=val, help=help_text)

    args = parser.parse_args()
    return vars(args)
