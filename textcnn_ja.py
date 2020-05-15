#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import click
from models import ConfigProcessorTextCNN, ProcessorTextCNN


@click.command()
@click.option('--config', '-c', required=True, type=str)
def main(config):
    assert(os.path.exists(config))
    config_processor = ConfigProcessorTextCNN(config)
    processor = ProcessorTextCNN(config_processor)
    processor.process()
    return


if __name__ == "__main__":
    main()
