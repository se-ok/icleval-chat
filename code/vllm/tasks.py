from pathlib import Path

from .base import ICLEvalTask


class CopyNumber(ICLEvalTask):
    filename = Path("copy_dict_search_number.json")


class CopyString(ICLEvalTask):
    filename = Path("copy_dict_search_string.json")


class CopyHash(ICLEvalTask):
    filename = Path("copy_natural_language_string.json")


class CheckOrder(ICLEvalTask):
    filename = Path("classifier_order.json")


class GenerateOrder(ICLEvalTask):
    filename = Path("generate_order.json")


class CheckDuplication(ICLEvalTask):
    filename = Path("classifier_duplication.json")


class GenerateDuplication(ICLEvalTask):
    filename = Path("generate_duplication.json")


class RelationAnalysis(ICLEvalTask):
    filename = Path("generate_relation_analysis.json")


class CountNavigation(ICLEvalTask):
    filename = Path("generate_count_or_navigation.json")


class CheckFormat(ICLEvalTask):
    filename = Path("classifier_format.json")


class GenerateFormat(ICLEvalTask):
    filename = Path("generate_output_format.json")


class ConvertFormat(ICLEvalTask):
    filename = Path("generate_format_conversion.json")


class ListNumber(ICLEvalTask):
    filename = Path("generate_list_number.json")
