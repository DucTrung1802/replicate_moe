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

    parser = argparse.ArgumentParser(description="Training configuration loader")

    for key, val in config.items():
        # Explicit help texts for each parameter
        if key == "model":
            parser.add_argument(
                f"--{key}",
                type=str,
                choices=supported.model,
                default=val,
                help=f"Model architecture to use for training. Options: {supported.model}. Default: {val}",
            )

        elif key == "mixture":
            parser.add_argument(
                f"--{key}",
                dest=key,
                action="store_true",
                help="Enable Mixture of Experts (MoE) model. Default: True",
            )
            parser.add_argument(
                f"--no-{key}",
                dest=key,
                action="store_false",
                help="Disable Mixture of Experts (MoE) model.",
            )
            parser.set_defaults(**{key: val})

        elif key == "expert_num":
            parser.add_argument(
                f"--{key}",
                type=int,
                default=val,
                help=f"Number of expert networks in the MoE model. Default: {val}",
            )

        elif key == "strategy":
            parser.add_argument(
                f"--{key}",
                type=str,
                choices=supported.strategy,
                default=val,
                help=f"Gating strategy for routing in MoE. Options: {supported.strategy}. Default: {val}",
            )

        elif key == "degree":
            parser.add_argument(
                f"--{key}",
                type=int,
                default=val,
                help=f"Maximum rotation degree used in data augmentation. Default: {val}",
            )

        elif key == "crop":
            parser.add_argument(
                f"--{key}",
                type=int,
                default=val,
                help=f"Crop size for image preprocessing. Default: {val}",
            )

        elif key == "classes":
            parser.add_argument(
                f"--{key}",
                type=int,
                default=val,
                help=f"Number of target classes for classification. Default: {val}",
            )

        elif key == "clusters":
            parser.add_argument(
                f"--{key}",
                type=int,
                default=val,
                help=f"Number of clusters used in the model or dataset. Default: {val}",
            )

        elif key == "batch_size":
            parser.add_argument(
                f"--{key}",
                type=int,
                default=val,
                help=f"Training batch size. Default: {val}",
            )

        elif key == "learning_rate":
            parser.add_argument(
                f"--{key}",
                type=float,
                default=val,
                help=f"Initial learning rate for optimizer. Default: {val}",
            )

        elif key == "momentum":
            parser.add_argument(
                f"--{key}",
                type=float,
                default=val,
                help=f"Momentum factor for the optimizer (e.g., SGD). Default: {val}",
            )

        elif key == "weight_decay":
            parser.add_argument(
                f"--{key}",
                type=float,
                default=val,
                help=f"Weight decay (L2 regularization) value. Default: {val}",
            )

        elif key == "max_epoch":
            parser.add_argument(
                f"--{key}",
                type=int,
                default=val,
                help=f"Maximum number of training epochs. Default: {val}",
            )

        elif key == "patience":
            parser.add_argument(
                f"--{key}",
                type=int,
                default=val,
                help=f"Patience for early stopping. Training stops if no improvement after this many epochs. Default: {val}",
            )

        elif key == "early_stop":
            parser.add_argument(
                f"--{key}",
                dest=key,
                action="store_true",
                help="Enable early stopping during training. Default: False",
            )
            parser.add_argument(
                f"--no-{key}",
                dest=key,
                action="store_false",
                help="Disable early stopping.",
            )
            parser.set_defaults(**{key: val})

        elif key == "resume":
            parser.add_argument(
                f"--{key}",
                dest=key,
                action="store_true",
                help="Resume training from a checkpoint. Default: False",
            )
            parser.add_argument(
                f"--no-{key}",
                dest=key,
                action="store_false",
                help="Do not resume training.",
            )
            parser.set_defaults(**{key: val})

        elif key == "resume_from_file":
            parser.add_argument(
                f"--{key}",
                type=str,
                default=val,
                help=f"Path to the checkpoint file to resume from. Default: '{val}'",
            )

        elif key == "wandb_upload":
            parser.add_argument(
                f"--{key}",
                dest=key,
                action="store_true",
                help="Enable uploading logs to Weights & Biases (WandB). Default: True",
            )
            parser.add_argument(
                f"--no-{key}",
                dest=key,
                action="store_false",
                help="Disable WandB logging.",
            )
            parser.set_defaults(**{key: val})

        elif key == "wandb_resume_id":
            parser.add_argument(
                f"--{key}",
                type=str,
                default=None,
                help="WandB run ID to resume from (if applicable). Default: None",
            )

        elif key == "wandb_name":
            parser.add_argument(
                f"--{key}",
                type=str,
                default=None,
                help="Custom name for the WandB run. Default: None",
            )

        elif key == "note":
            parser.add_argument(
                f"--{key}",
                type=str,
                default=None,
                help="Optional note or tag to annotate the run. Default: None",
            )

        else:
            # Generic fallback for unknown types
            parser.add_argument(
                f"--{key}",
                type=type(val),
                default=val,
                help=f"{key.replace('_', ' ').capitalize()}. Default: {val}",
            )

    args = parser.parse_args()
    return vars(args)
