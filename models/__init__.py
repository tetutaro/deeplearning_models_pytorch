#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .ProcessorTextCNN import ProcessorTextCNN
from .ProcessorGradCAM import ProcessorGradCAM
from .ProcessorBertClassification import ProcessorBertClassification
from .ProcessorBertSum import ProcessorBertSum
from .ProcessorCycleGAN import ProcessorCycleGAN

__all__ = [
    ProcessorTextCNN, ProcessorGradCAM,
    ProcessorBertClassification, ProcessorBertSum,
    ProcessorCycleGAN,
]
