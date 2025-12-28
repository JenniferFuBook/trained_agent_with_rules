import argparse        # Handles command-line arguments
import random          # Used to randomly select nodes and templates
from pathlib import Path  # Safer path handling across OSes


def load_lines(path):
    """
    Load a text file and return a list of non-empty, stripped lines.

    Example:
        nodes.txt:
            api
            db
            cache

        → ["api", "db", "cache"]
    """
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def generate_examples(
    nodes,
    templates,
    output_template,
    count,
    output_file,
    multi_node=False
):
    """
    Generate training examples in the format:

    ### Input:
    <natural language>

    ### Output:
    <structured command>

    Args:
        nodes (list[str]):
            Available node names to sample from.

        templates (list[str]):
            Natural language templates. May contain:
              - {node}
              - {node_a} and {node_b}

        output_template (str):
            Command template, e.g.:
              "ADD_NODE {node}"
              "DELETE_NODE {node}"
              "CONNECT {node_a} {node_b}"

        count (int):
            Number of examples to generate.

        output_file (Path):
            Where the generated data will be written.

        multi_node (bool):
            If True, generate commands involving two nodes
            (CONNECT / DISCONNECT).
    """
    examples = []

    for _ in range(count):

        # ------------------------------------------
        # Multi-node case (CONNECT / DISCONNECT)
        # ------------------------------------------
        if multi_node:
            # Randomly select two distinct nodes
            node_a, node_b = random.sample(nodes, 2)

            # Fill the natural language template
            text = random.choice(templates).format(
                node_a=node_a,
                node_b=node_b
            )

            # Construct training example (single line format)
            example = f"### Input: {text} ### Output: {output_template.format(node_a=node_a, node_b=node_b)}"

        # ------------------------------------------
        # Single-node case (ADD_NODE / DELETE_NODE)
        # ------------------------------------------
        else:
            # Randomly select one node
            node = random.choice(nodes)

            # Fill the natural language template
            text = random.choice(templates).format(node=node)

            # Construct training example (single line format)
            example = f"### Input: {text} ### Output: {output_template.format(node=node)}"

        examples.append(example)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write all examples to file
    output_file.write_text("\n".join(examples), encoding="utf-8")

    print(f"✅ Generated {count} examples → {output_file}")


def main():
    """
    Entry point for command-line execution.
    Parses arguments and dispatches dataset generation.
    """
    parser = argparse.ArgumentParser(
        description="Generate node command training data"
    )

    # Path to nodes.txt
    parser.add_argument(
        "--nodes",
        required=True,
        help="Path to nodes.txt"
    )

    # Path to natural language templates
    parser.add_argument(
        "--templates",
        required=True,
        help="Path to templates.txt"
    )

    # Output file path
    parser.add_argument(
        "--output",
        required=True,
        help="Output file path"
    )

    # Number of examples to generate
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of examples"
    )

    # Structured command template
    parser.add_argument(
        "--command-template",
        required=True,
        help=(
            'Output command template, e.g. '
            '"ADD_NODE {node}", '
            '"DELETE_NODE {node}", '
            '"CONNECT {node_a} {node_b}"'
        )
    )

    args = parser.parse_args()

    # Load nodes and templates from disk
    nodes = load_lines(Path(args.nodes))
    templates = load_lines(Path(args.templates))

    # Detect whether this is a multi-node command
    multi_node = (
        "{node_a}" in args.command_template and
        "{node_b}" in args.command_template
    )

    # Generate dataset
    generate_examples(
        nodes=nodes,
        templates=templates,
        output_template=args.command_template,
        count=args.count,
        output_file=Path(args.output),
        multi_node=multi_node
    )


# Only run main() if executed directly
if __name__ == "__main__":
    main()
