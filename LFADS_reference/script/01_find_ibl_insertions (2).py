"""Search IBL Brain-Wide Map for probe insertion IDs (PIDs).

Run from a terminal, not by copying shell commands into a notebook cell.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="MOp", help="Allen atlas acronym, e.g. MOp")
    parser.add_argument("--project", default="brainwide", help="IBL project")
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()
    from one.api import ONE

    one = ONE(base_url="https://openalyx.internationalbrainlab.org")
    insertions = one.search_insertions(
        atlas_acronym=args.region,
        datasets="spikes.times.npy",
        project=args.project,
    )
    print(f"Found {len(insertions)} insertions. Showing up to {args.limit}:")
    for idx, insertion in enumerate(insertions[: args.limit]):
        print(f"{idx + 1:3d}. {str(insertion)}")
    print("\nCopy one PID into 02_ibl_to_h5.py via --pid.")


if __name__ == "__main__":
    main()
