#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import click
from models import ProcessorBertClassification


@click.command()
@click.option('--config', '-c', required=True, type=str)
def main(config):
    assert(os.path.exists(config))
    processor = ProcessorBertClassification(config)
    processor.process()
    return


if __name__ == "__main__":
    main()
