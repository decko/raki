import click


@click.group()
@click.version_option(package_name="raki")
def main():
    """RAKI — Retrieval Assessment for Knowledge Impact"""


if __name__ == "__main__":
    main()
