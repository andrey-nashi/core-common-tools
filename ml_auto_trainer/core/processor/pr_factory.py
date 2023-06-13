import inspect

from .pr_seg_bin import ProcessorBinarySegmentation
from .pr_clf_multi import ProcessorMultilabelClassification

class ProcessorFactory:

    _LIST_PROCESSORS = [
        ProcessorBinarySegmentation,
        ProcessorMultilabelClassification
     ]

    _TABLE_PROCESSORS = {m.__name__:m for m in _LIST_PROCESSORS}

    @staticmethod
    def create_processor(processor_name, processor_methods, args):
        if processor_name in ProcessorFactory._TABLE_PROCESSORS:
            return ProcessorFactory._TABLE_PROCESSORS[processor_name](methods=processor_methods, **args)

        else:
            raise NotImplemented
