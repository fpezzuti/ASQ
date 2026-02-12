
from dataclasses import dataclass
import re


def extract_between_tags(text, start_tag, end_tag):
    pattern = re.compile(rf"{re.escape(start_tag)}(.*?){re.escape(end_tag)}", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return None


@dataclass(frozen=True)
class ControlTAGs:
    QUERY: str = "search"
    ANSWER: str = "answer"
    THINK: str = "think"
    RANKEDLIST: str = "information"
    OPEN_TAG_TEMPLATE: str = "<{tag_name}>"
    CLOSE_TAG_TEMPLATE: str = "</{tag_name}>"

    def get_open_tag(self, tag_name: str) -> str:
        return self.OPEN_TAG_TEMPLATE.format(tag_name=tag_name)

    def get_close_tag(self, tag_name: str) -> str:
        return self.CLOSE_TAG_TEMPLATE.format(tag_name=tag_name)

    def extract(self, tag: str, text: str) -> str | None:
        return extract_between_tags(
            text,
            self.get_open_tag(tag),
            self.get_close_tag(tag),
        )


class RegexExtractor:
    def __init__(self, ctrl_tags: ControlTAGs):
        self.control_tags = ctrl_tags

    def get_query(self, text):
        return self.control_tags.extract(self.control_tags.QUERY, text)

    def get_answer(self, text):
        return self.control_tags.extract(self.control_tags.ANSWER, text)

    def get_thought(self, text):
        return self.control_tags.extract(self.control_tags.THINK, text)


_DEFAULT_TAGS = ControlTAGs()
_DEFAULT_EXTRACTOR = RegexExtractor(_DEFAULT_TAGS)


def get_query(text):
    return _DEFAULT_EXTRACTOR.get_query(text)


def get_answer(text):
    return _DEFAULT_EXTRACTOR.get_answer(text)


def get_thought(text):
    return _DEFAULT_EXTRACTOR.get_thought(text)